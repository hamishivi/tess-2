from transformers import Trainer
from transformers.utils import is_apex_available
from typing import Dict, Union, Any, Optional, List, Tuple
import torch
from torch import nn
from sdlm.utils import convert_to_simplex, scale
from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_utils import EvalLoopOutput, has_length, denumpify_detensorize
from sdlm.utils import EvalPrediction
from transformers.trainer_pt_utils import nested_numpify
from transformers.utils import logging
from torch.utils.data import DataLoader
from transformers.deepspeed import deepspeed_init
from transformers.trainer_pt_utils import find_batch_size, nested_concat, nested_truncate
from sdlm.pipelines.simplex_ddpm import SimplexDDPMPipeline

if is_apex_available():
    from apex import amp


IS_SAGEMAKER_MP_POST_1_10 = False


logger = logging.get_logger(__name__)


class DiffusionTrainer(Trainer):
    def __init__(self, noise_scheduler, inference_noise_scheduler, diffusion_args, data_args, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_scheduler = noise_scheduler
        self.diffusion_args = diffusion_args
        self.data_args = data_args
        self.vocab_size = self.model.config.vocab_size
        self.inference_noise_scheduler = inference_noise_scheduler

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.
        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Creates the noisy simplex and timesteps.
        simplex = convert_to_simplex(inputs["input_ids"], self.diffusion_args.simplex_value, self.vocab_size)
        noise = self.diffusion_args.simplex_value * torch.randn(simplex.shape, device=simplex.device, dtype=simplex.dtype)
        bsz = simplex.shape[0]
        # Sample a random timestep for each simplex token representation.
        timesteps = torch.randint(0, len(self.noise_scheduler), (bsz,), device=simplex.device, dtype=torch.int64)
        # Adds noise to each simplex representation (Forward diffusion process).
        noisy_simplex = self.noise_scheduler.add_noise(simplex, noise, timesteps)
        timesteps = scale(timesteps, len(self.noise_scheduler))
        inputs.update({"timesteps": timesteps, "simplex": noisy_simplex})

        with self.autocast_smart_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    def prediction_step(
        self, inputs: Dict[str, Union[torch.Tensor, Any]], pipeline
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                outputs = pipeline(
                    batch_size=self.args.per_device_eval_batch_size,
                    seq_length=self.data_args.max_seq_length,
                    batch=inputs,
                    guidance_scale=self.diffusion_args.guidance_scale,
                )
        logits = nested_detach(outputs.logits)
        simplex = nested_detach(outputs.simplex)
        return (simplex, logits)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.
        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only
        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(self, num_training_steps=0, resume_from_checkpoint=None, inference=True)
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        pipeline = SimplexDDPMPipeline(
            model=model,
            scheduler=self.inference_noise_scheduler,
            simplex_value=self.diffusion_args.simplex_value,
            top_p=self.diffusion_args.top_p,
            sampling_type=self.diffusion_args.sampling_type,
            span_infilling=self.data_args.span_infilling,
            tokenizer=self.tokenizer,
            classifier_free_uncond_input=self.diffusion_args.classifier_free_uncond_input,
            classifier_free_guided_prev_outputs=self.diffusion_args.classifier_free_guided_prev_outputs,
        )

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        # Initialize containers
        # preds/simplex/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        preds_host = None
        simplex_host = None
        labels_host = None
        inputs_host = None

        # preds/simplex/labels on CPU (final containers)
        all_preds = None
        all_simplex = None
        all_labels = None
        all_inputs = None
        observed_num_examples = 0

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            simplex, logits = self.prediction_step(inputs, pipeline=pipeline)
            inputs_decode = self._prepare_input(inputs["input_ids"])

            # Update containers on host
            if inputs_decode is not None:
                inputs_decode = self._pad_across_processes(inputs_decode)
                inputs_decode = self._nested_gather(inputs_decode)
                inputs_host = (
                    inputs_decode if inputs_host is None else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if simplex is not None:
                simplex = self._pad_across_processes(simplex)
                simplex = self._nested_gather(simplex)
                if self.preprocess_logits_for_metrics is not None:
                    simplex = self.preprocess_logits_for_metrics(simplex)
                simplex_host = simplex if simplex_host is None else nested_concat(simplex_host, simplex, padding_index=-100)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

        # Gather all remaining tensors and put them back on the CPU
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if simplex_host is not None:
            simplex = nested_numpify(simplex_host)
            all_simplex = simplex if all_simplex is None else nested_concat(all_simplex, simplex, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )

        # Number of samples
        num_samples = len(eval_dataset)
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_simplex is not None:
            all_simplex = nested_truncate(all_simplex, num_samples)
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_inputs is not None:
            all_inputs = nested_truncate(all_inputs, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None and all_simplex is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        # if all_losses is not None:
        #    metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
