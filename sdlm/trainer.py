from transformers import Trainer
from transformers.utils import is_apex_available
from typing import Dict, Union, Any, Optional, List, Tuple, NamedTuple
from torch.utils.data import Dataset
import torch
import pdb
import math
import time
import numpy as np
from torch import nn
import torch.nn.functional as F
from sdlm.utils import convert_to_simplex, scale
from transformers.trainer_pt_utils import nested_detach, nested_numpify, find_batch_size, nested_concat, nested_truncate
from transformers.integrations import is_fairscale_available, TensorBoardCallback
from transformers.trainer_utils import has_length, denumpify_detensorize, speed_metrics, ShardedDDPOption
from transformers.utils import logging
from torch.utils.data import DataLoader
from transformers.deepspeed import deepspeed_init
from sdlm.pipelines.simplex_ddpm import SimplexDDPMPipeline
from sdlm.inference.inference_utils import predict_conditional_generated, logits_projection
from sdlm.utils import self_condition_preds


if is_apex_available():
    from apex import amp

if is_fairscale_available():
    dep_version_check("fairscale")
    from fairscale.optim import OSS

IS_SAGEMAKER_MP_POST_1_10 = False


logger = logging.get_logger(__name__)


class EvalLoopOutput(NamedTuple):
    logits: Union[np.ndarray, Tuple[np.ndarray]]
    simplex: Union[np.ndarray, Tuple[np.ndarray]]
    input_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    results: Optional[Dict[str, List[str]]]
    num_samples: Optional[int]


class DiffusionTrainer(Trainer):
    def __init__(
        self,
        causal_model,
        causal_tokenizer,
        noise_scheduler,
        inference_noise_scheduler,
        diffusion_args,
        data_args,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.noise_scheduler = noise_scheduler
        self.diffusion_args = diffusion_args
        self.data_args = data_args
        self.vocab_size = self.model.config.vocab_size
        self.inference_noise_scheduler = inference_noise_scheduler
        self.causal_model = causal_model
        self.causal_tokenizer = causal_tokenizer
        self.tb_writer = self.get_tb_writer()

    def get_tb_writer(self):
        for cb in self.callback_handler.callbacks:
            if isinstance(cb, TensorBoardCallback):
                return cb
        return None

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

        if self.diffusion_args.self_condition is not None:
            previous_pred = None
            if np.random.rand(1) > 0.5:
                outputs = model(
                    simplex=noisy_simplex,
                    timesteps=timesteps,
                    input_ids=inputs["input_ids"],
                    span_mask=inputs["span_mask"] if self.data_args.span_infilling else None,
                    previous_pred=previous_pred,
                )
                logits_projection_fct = lambda x: logits_projection(
                    x, self.diffusion_args.sampling_type, self.diffusion_args.top_p, self.diffusion_args.simplex_value
                )
                previous_pred = self_condition_preds(
                    self.diffusion_args.self_condition, outputs.logits, logits_projection_fct
                )
                inputs.update({"previous_pred": previous_pred})

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
        # logits/simplex/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        logits_host = None
        simplex_host = None
        inputs_host = None
        masks_host = None

        # logits/simplex/labels on CPU (final containers)
        all_logits = None
        all_simplex = None
        all_inputs = None
        all_masks = None
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
            masks = self._prepare_input(inputs["span_mask"]) if self.data_args.span_infilling else None

            # Update containers on host
            if inputs_decode is not None:
                inputs_decode = self._pad_across_processes(inputs_decode)
                inputs_decode = self._nested_gather(inputs_decode)
                inputs_host = (
                    inputs_decode if inputs_host is None else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if masks is not None:
                # TODO: check pad for masks.
                masks = self._pad_across_processes(masks)
                masks = self._nested_gather(masks)
                masks_host = masks if masks_host is None else nested_concat(masks_host, masks, padding_index=-100)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits)
                logits_host = logits if logits_host is None else nested_concat(logits_host, logits, padding_index=-100)
            if simplex is not None:
                simplex = self._pad_across_processes(simplex)
                simplex = self._nested_gather(simplex)
                # TODO: note that this is no more a simplex, but the processed one.
                simplex = F.softmax(simplex, dim=-1)
                if self.preprocess_logits_for_metrics is not None:
                    simplex = self.preprocess_logits_for_metrics(simplex)
                simplex_host = simplex if simplex_host is None else nested_concat(simplex_host, simplex, padding_index=-100)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

        # Gather all remaining tensors and put them back on the CPU
        if logits_host is not None:
            logits = nested_numpify(logits_host)
            all_logits = logits if all_logits is None else nested_concat(all_logits, logits, padding_index=-100)
        if simplex_host is not None:
            simplex = nested_numpify(simplex_host)
            all_simplex = simplex if all_simplex is None else nested_concat(all_simplex, simplex, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if masks_host is not None:
            masks = nested_numpify(masks)
            all_masks = masks if all_masks is None else nested_concat(all_masks, masks, padding_index=-100)

        # Number of samples
        num_samples = len(eval_dataset)
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_simplex is not None:
            all_simplex = nested_truncate(all_simplex, num_samples)
        if all_logits is not None:
            all_logits = nested_truncate(all_logits, num_samples)
        if all_inputs is not None:
            all_inputs = nested_truncate(all_inputs, num_samples)

        # Generates the texts.
        results = {}
        if self.data_args.span_infilling:
            # We predict the masked tokens only. Here, we compute the masked tokens.
            results.update(
                predict_conditional_generated(all_masks, all_inputs, self.tokenizer, all_simplex, "pred_texts_from_simplex")
            )
            results.update(
                predict_conditional_generated(all_masks, all_inputs, self.tokenizer, all_logits, "pred_texts_from_logits")
            )
        else:
            results.update({"pred_texts_from_simplex": self.tokenizer.batch_decode(all_simplex, skip_special_tokens=False)})
            results.update({"pred_texts_from_logits": self.tokenizer.batch_decode(all_logits, skip_special_tokens=False)})

        if self.data_args.span_infilling:
            # Adds the decoded original texts to the final results.
            results.update({"gold_texts": self.tokenizer.batch_decode(all_inputs, skip_special_tokens=False)})

        # Metrics!
        # TODO: make sure causal model is going through the same stuff as the model.
        # TODO: we need to make sure metric for checkpoint is selected.
        metrics = self.compute_metrics(results, self.causal_model, self.causal_tokenizer, self.data_args.span_infilling)

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        # if all_losses is not None:
        #    metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(
            logits=all_logits,
            simplex=all_simplex,
            input_ids=all_inputs,
            metrics=metrics,
            num_samples=num_samples,
            results=results,
        )

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.
        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).
        You can also subclass and override this method to inject custom behavior.
        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)
        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        # Adds the generated texts to tensorboard.
        if self.args.log_generated_texts:
            self.log_results_to_tensorboard(self.state, output)

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )
        self.log(output.metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)
        return output.metrics

    def log_results_to_tensorboard(self, state, output):
        for i in range(len(output.logits)):
            total_text = ""
            for k, v in output.results.items():
                total_text += f"*** {k} ***: {v[i]}" + "  \n"
            self.tb_writer.tb_writer.add_text(f"sample_{i}", total_text, state.global_step)

    '''
    # TODO: check with and without this.
    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model

        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight", "timestep_embed.weight", "timestep_embed.bias"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in opt_model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in opt_model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")

        return self.optimizer
    '''
