from transformers import Trainer
from transformers.utils import is_apex_available
from typing import Dict, Union, Any, Optional, List, Tuple, NamedTuple
from torch.utils.data import Dataset
import torch
import pdb
import math
import time
import os
import numpy as np
from torch import nn
import torch.nn.functional as F
from sdlm.utils import convert_to_simplex, scale
from transformers.trainer_pt_utils import (
    nested_detach,
    nested_numpify,
    find_batch_size,
    nested_concat,
    nested_truncate,
    IterableDatasetShard,
)
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from torch.utils.data import RandomSampler
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
import sys
from transformers.trainer_callback import TrainerState
from transformers.integrations import TensorBoardCallback, hp_params
from transformers.trainer_utils import (
    has_length,
    denumpify_detensorize,
    speed_metrics,
    seed_worker,
    HPSearchBackend,
    TrainOutput,
    ShardedDDPOption,
)
from transformers.utils import logging, is_datasets_available, is_torch_tpu_available
from transformers.pytorch_utils import is_torch_less_than_1_11
from torch.utils.data import DataLoader
from transformers.deepspeed import deepspeed_init
from sdlm.pipelines.simplex_ddpm import SimplexDDPMPipeline
from sdlm.inference.inference_utils import predict_conditional_generated, logits_projection
from sdlm.utils import self_condition_preds, pad_data
from torch.nn import CrossEntropyLoss
from transformers.utils import is_sagemaker_mp_enabled
from transformers import AdamW
import shutil
from transformers.trainer import TRAINER_STATE_NAME

if is_apex_available():
    from apex import amp

if is_datasets_available():
    import datasets

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

IS_SAGEMAKER_MP_POST_1_10 = False
GENERATION_RESULTS = "generated"


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
        self.tb_writer = self.get_tb_writer()
        self.eos_token_id = self.tokenizer.eos_token_id
        self.classifier_free_guidance = diffusion_args.guidance_scale > 1.0 and data_args.conditional_generation is not None

    def annotated_split(self, split):
        return f"{split}_top_p_{self.diffusion_args.top_p}_temperature_{self.diffusion_args.temperature}_seed_{self.args.seed}_guidance_scale_{self.diffusion_args.guidance_scale}"

    def save_metrics(self, split, metrics, combined=True):
        super().save_metrics(self.annotated_split(split), metrics, combined)

    def log_metrics(self, split, metrics):
        super().log_metrics(self.annotated_split(split), metrics)


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

        # Truncate the length if needed.
        if self.data_args.truncation_length > 0:
            inputs["input_ids"] = inputs["input_ids"][:, : -self.data_args.truncation_length]
            inputs["span_mask"] = inputs["span_mask"][:, : -self.data_args.truncation_length]

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
        if self.diffusion_args.self_condition is not None:
            previous_pred = None
            if np.random.rand(1) > 0.5:
                outputs = model(**inputs, previous_pred=previous_pred)
                logits_projection_fct = lambda x: logits_projection(
                    x,
                    self.diffusion_args.sampling_type,
                    self.diffusion_args.top_p,
                    self.diffusion_args.simplex_value,
                    self.diffusion_args.temperature,
                )
                previous_pred = self_condition_preds(
                    self.diffusion_args.self_condition, outputs.logits, logits_projection_fct
                )
            inputs.update({"previous_pred": previous_pred})
        # NOTE: we do this after computation of self-conditioning to not affect that one.
        inputs.update({"classifier_free_guidance_in_train": self.classifier_free_guidance})
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
        is_conditional_generation = True if "span_mask" in inputs else False
        with torch.no_grad():
            with self.compute_loss_context_manager():
                outputs = pipeline(
                    batch_size=inputs["input_ids"].shape[0]
                    if is_conditional_generation
                    else self.args.per_device_eval_batch_size,
                    seq_length=self.data_args.max_seq_length - self.data_args.truncation_length,
                    batch=inputs,
                    guidance_scale=self.diffusion_args.guidance_scale,
                    generator=torch.Generator(device=self.args.device).manual_seed(self.args.seed) if self.diffusion_args.generate_with_seed else None,
                )
                if is_conditional_generation:
                    loss = outputs.loss.mean().detach()
                else:
                    loss = None
        logits = nested_detach(outputs.logits)
        simplex = nested_detach(outputs.simplex)

        # Computes the evaluation loss from the simplex values.
        if self.args.compute_eval_loss_with_simplex and is_conditional_generation:
            loss_fct = CrossEntropyLoss()
            labels = torch.where(inputs["span_mask"], inputs["input_ids"], -100)
            loss = loss_fct(simplex.view(-1, self.model.config.vocab_size), labels.view(-1)).detach()

        return (simplex, logits, loss)

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
        is_conditional_generation = self.data_args.conditional_generation is not None
        save_prefixes = is_conditional_generation  # and self.data_args.conditional_generation != "seq2seq"

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
            is_conditional_generation=is_conditional_generation,
            tokenizer=self.tokenizer,
            classifier_free_uncond_input=self.diffusion_args.classifier_free_uncond_input,
            temperature=self.diffusion_args.temperature,
            guidance_softmax_combination=self.diffusion_args.guidance_softmax_combination
        )

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        # Initialize containers
        # logits/simplex/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        logits_host = None
        simplex_host = None
        inputs_host = None
        masks_host = None
        prefixes_host = None

        # logits/simplex/labels on CPU (final containers)
        all_losses = None
        all_logits = None
        all_simplex = None
        all_inputs = None
        all_masks = None
        all_prefixes = None
        observed_num_examples = 0

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            has_mask = True if "span_mask" in inputs else False

            # Truncate the length if needed.
            if self.data_args.truncation_length > 0:
                inputs["input_ids"] = inputs["input_ids"][:, : -self.data_args.truncation_length]
                inputs["span_mask"] = inputs["span_mask"][:, : -self.data_args.truncation_length]
                max_seq_length = self.data_args.max_seq_length - self.data_args.truncation_length
                assert self.data_args.eval_context_size < max_seq_length

            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            simplex, logits, loss = self.prediction_step(inputs, pipeline=pipeline)
            inputs_decode = self._prepare_input(inputs["input_ids"])
            masks = self._prepare_input(inputs["span_mask"]) if has_mask else None
            if save_prefixes:
                prefixes = (
                    pad_data([input[~mask] for input, mask in zip(inputs_decode, masks)], self.tokenizer)
                    if has_mask
                    else None
                )
                prefixes = self._prepare_input(prefixes)
            else:
                prefixes = None
            # Update containers on host
            if prefixes is not None:
                prefixes = self._pad_across_processes(prefixes, pad_index=self.eos_token_id)
                prefixes = self._nested_gather(prefixes)
                prefixes_host = (
                    prefixes
                    if prefixes_host is None
                    else nested_concat(prefixes_host, prefixes, padding_index=self.eos_token_id)
                )
            if inputs_decode is not None:
                inputs_decode = self._pad_across_processes(inputs_decode, pad_index=self.eos_token_id)
                inputs_decode = self._nested_gather(inputs_decode)
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=self.eos_token_id)
                )
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            # Note that this block should be before masks block, since we need masks here.
            if simplex is not None:
                # In case of having a mask softmax is applied over the simplex non-masked values.
                if has_mask:
                    mask_value = torch.finfo(simplex.dtype).min
                    mask_value = torch.tensor(mask_value, dtype=simplex.dtype, device=simplex.device)
                    simplex = torch.where(masks[:, :, None], simplex, mask_value)
                simplex = F.softmax(simplex, dim=-1)
                if self.preprocess_logits_for_metrics is not None:
                    simplex = self.preprocess_logits_for_metrics(simplex)
                simplex = self._pad_across_processes(simplex, pad_index=self.eos_token_id)
                simplex = self._nested_gather(simplex)
                # TODO: note that this is no more a simplex, but the processed one.
                simplex_host = (
                    simplex
                    if simplex_host is None
                    else nested_concat(simplex_host, simplex, padding_index=self.eos_token_id)
                )
            if masks is not None:
                masks = self._pad_across_processes(masks, pad_index=0)
                masks = self._nested_gather(masks)
                # We pad masks with False tokens.
                masks_host = masks if masks_host is None else nested_concat(masks_host, masks, padding_index=0)
            if logits is not None:
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits)
                logits = self._pad_across_processes(logits, pad_index=self.eos_token_id)
                logits = self._nested_gather(logits)
                logits_host = (
                    logits if logits_host is None else nested_concat(logits_host, logits, padding_index=self.eos_token_id)
                )

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if logits_host is not None:
            logits = nested_numpify(logits_host)
            all_logits = logits if all_logits is None else nested_concat(all_logits, logits, padding_index=self.eos_token_id)
        if simplex_host is not None:
            simplex = nested_numpify(simplex_host)
            all_simplex = (
                simplex if all_simplex is None else nested_concat(all_simplex, simplex, padding_index=self.eos_token_id)
            )
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode
                if all_inputs is None
                else nested_concat(all_inputs, inputs_decode, padding_index=self.eos_token_id)
            )
        if masks_host is not None:
            masks = nested_numpify(masks_host)
            all_masks = masks if all_masks is None else nested_concat(all_masks, masks, padding_index=0)
        if prefixes_host is not None:
            prefixes = nested_numpify(prefixes_host)
            all_prefixes = (
                prefixes if all_prefixes is None else nested_concat(all_prefixes, prefixes, padding_index=self.eos_token_id)
            )

        # Number of samples
        num_samples = len(eval_dataset)
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_masks is not None:
            all_masks = nested_truncate(all_masks, num_samples)
        if all_simplex is not None:
            all_simplex = nested_truncate(all_simplex, num_samples)
        if all_logits is not None:
            all_logits = nested_truncate(all_logits, num_samples)
        if all_inputs is not None:
            all_inputs = nested_truncate(all_inputs, num_samples)
        if all_prefixes is not None:
            all_prefixes = nested_truncate(all_prefixes, num_samples)

        # Generates the texts.
        results = {}
        if is_conditional_generation:

            # We predict the masked tokens only. Here, we compute the masked tokens.
            results.update(
                predict_conditional_generated(
                    all_masks,
                    all_inputs,
                    self.tokenizer,
                    all_simplex,
                    "pred_texts_from_simplex",
                    self.data_args.skip_special_tokens,
                )
            )
            results.update(
                predict_conditional_generated(
                    all_masks,
                    all_inputs,
                    self.tokenizer,
                    all_logits,
                    "pred_texts_from_logits",
                    self.data_args.skip_special_tokens,
                )
            )
        else:
            results.update(
                {
                    "pred_texts_from_simplex": self.tokenizer.batch_decode(
                        all_simplex, skip_special_tokens=self.data_args.skip_special_tokens
                    )
                }
            )
            results.update(
                {
                    "pred_texts_from_logits": self.tokenizer.batch_decode(
                        all_logits, skip_special_tokens=self.data_args.skip_special_tokens
                    )
                }
            )
        if is_conditional_generation:

            results.update(
                {
                    "gold_texts_masked": [
                        self.tokenizer.decode(input[mask], skip_special_tokens=self.data_args.skip_special_tokens)
                        for mask, input in zip(all_masks, all_inputs)
                    ]
                }
            )
            if save_prefixes:
                results.update(
                    {
                        "prefixes": [
                            self.tokenizer.decode(x, skip_special_tokens=True)  # self.data_args.skip_special_tokens)
                            for x in all_prefixes
                        ]
                    }
                )

        # Metrics.
        if self.compute_metrics is not None:
            metrics = self.compute_metrics(results)
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

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

        output = self.evaluation_loop(
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

        # Save the results
        self.save_metrics(GENERATION_RESULTS + "_" + metric_key_prefix, output.results)
        logger.info("Results are saved now")

        return output.metrics

    def log_results_to_tensorboard(self, state, output):
        # TODO: we need to fix this which happens during the only eval option.
        if self.tb_writer.tb_writer is None:
            return
        for i in range(len(output.logits)):
            total_text = ""
            for k, v in output.results.items():
                total_text += f"*** {k} ***: {v[i]}" + "  \n"
            self.tb_writer.tb_writer.add_text(f"sample_{i}", total_text, state.global_step)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].
        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.
        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator("train")
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
        )

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].
        Subclass and override this method if you want to inject some custom behavior.
        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator("eval")

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if not self.args.ssdlm_optimizer:
            return super().create_optimizer()

        # SDDLM optimizer.
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
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
        return self.optimizer

    def _rotate_checkpoints(self, use_mtime=False, output_dir=None) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime, output_dir=output_dir)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        # If save_total_limit=1 with load_best_model_at_end=True, we could end up deleting the last checkpoint, which
        # we don't do to allow resuming.
        save_total_limit = self.args.save_total_limit
        if (
            self.state.best_model_checkpoint is not None
            and self.args.save_total_limit == 1
            and checkpoints_sorted[-1] != self.state.best_model_checkpoint
        ):
            save_total_limit = 2

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
            if self.args.save_checkpoints_on_s3:
                # TODO: we need to remove this from the codes.
                command = f"AWS_ACCESS_KEY_ID=AKIA5BJLZJPW5VNOE6SI AWS_SECRET_ACCESS_KEY=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 aws s3 sync {checkpoint} s3://ai2-s2-research/rabeehk/{checkpoint}"
                os.system(command)
            shutil.rmtree(checkpoint)

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
            self.sharded_ddp is not None
            and self.sharded_ddp != ShardedDDPOption.SIMPLE
            or is_sagemaker_mp_enabled()
            or self.fsdp is not None
        )
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(f"  Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator) if len_dataloader is not None else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            step = -1
            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                if (
                    ((step + 1) % args.gradient_accumulation_steps != 0)
                    and args.local_rank != -1
                    and args._no_sync_in_gradient_accumulation
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss_step = self.training_step(model, inputs)
                else:
                    tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if self.do_grad_scaling:
                            # Reduce gradients first for XLA
                            if is_torch_tpu_available():
                                gradients = xm._fetch_gradients(self.optimizer)
                                xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    elif is_torch_tpu_available():
                        if self.do_grad_scaling:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            xm.optimizer_step(self.optimizer)
                    elif self.do_grad_scaling:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()

                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint.
        if self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            if self.is_world_process_zero():
                for checkpoint in checkpoints_sorted:
                    if checkpoint != self.state.best_model_checkpoint:
                        logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                        if self.args.save_checkpoints_on_s3:
                            # TODO: we need to remove this from the codes.
                            command = f"AWS_ACCESS_KEY_ID=AKIA5BJLZJPW5VNOE6SI AWS_SECRET_ACCESS_KEY=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 aws s3 sync {checkpoint} s3://ai2-s2-research/rabeehk/{checkpoint}"
                            os.system(command)
                        shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)
