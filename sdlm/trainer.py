import math
import os
import shutil
import sys
import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import Trainer
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import deepspeed_init, deepspeed_load_checkpoint
from transformers.integrations import TensorBoardCallback, hp_params
from transformers.trainer import TRAINER_STATE_NAME
from transformers.trainer_callback import TrainerState
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    find_batch_size,
    nested_concat,
    nested_detach,
    nested_numpify,
)
from transformers.trainer_utils import (  # ShardedDDPOption,
    HPSearchBackend,
    TrainOutput,
    denumpify_detensorize,
    has_length,
    seed_worker,
    speed_metrics,
)
from transformers.training_args import ParallelMode
from transformers.utils import (
    is_accelerate_available,
    is_apex_available,
    is_datasets_available,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    logging,
)

from .inference.inference_utils import logits_projection, predict_conditional_generated
from .models.utils import is_cdcd_check, is_tokenwise_cdcd_check
from .pipelines.simplex_ddpm import SimplexDDPMPipeline
from .utils import convert_to_simplex, pad_data, scale, self_condition_preds

if is_apex_available():
    from apex import amp

if is_datasets_available():
    import datasets

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

skip_first_batches = None
if is_accelerate_available():
    from accelerate import __version__ as accelerate_version

    if version.parse(accelerate_version) >= version.parse("0.16"):
        from accelerate import skip_first_batches

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
        inference_noise_schedulers,
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
        self.inference_noise_schedulers = inference_noise_schedulers
        self.inference_timesteps = diffusion_args.num_inference_diffusion_steps
        self.tb_writer = self.get_tb_writer()
        self.eos_token_id = self.tokenizer.eos_token_id
        self.classifier_free_guidance = (
            diffusion_args.guidance_scale > 1.0
            and data_args.conditional_generation is not None
        )
        self.counter = 0

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

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
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
            inputs["input_ids"] = inputs["input_ids"][
                :, : -self.data_args.truncation_length
            ]
            inputs["span_mask"] = inputs["span_mask"][
                :, : -self.data_args.truncation_length
            ]

        # Creates the noisy simplex and timesteps.
        simplex = convert_to_simplex(
            inputs["input_ids"], self.diffusion_args.simplex_value, self.vocab_size
        )
        noise = self.diffusion_args.simplex_value * torch.randn(
            simplex.shape, device=simplex.device, dtype=simplex.dtype
        )
        bsz = simplex.shape[0]
        # Sample a random timestep for each simplex token representation.
        timesteps = torch.randint(
            0,
            len(self.noise_scheduler),
            (bsz, inputs["input_ids"].shape[1])
            if is_tokenwise_cdcd_check(self.model)
            else (bsz,),
            device=simplex.device,
            dtype=torch.int64,
        )
        # expand out timesteps to match tokenwise setup
        if not is_tokenwise_cdcd_check(self.model):
            timesteps = timesteps[:, None].expand(-1, inputs["input_ids"].shape[1])

        # if we're not doing token warping, just set all relative positions to 1.
        norm_relative_position = torch.ones_like(inputs["input_ids"])
        # save original timesteps for warping
        original_timesteps = timesteps
        # warp timesteps according to cdf
        # we re-scale the timesteps to the correct range.
        # the -1 is due to the timestep should be in range [0, 5000)
        if is_tokenwise_cdcd_check(self.model):
            timesteps = self.model.warp_timesteps(
                timesteps, t_max=len(self.noise_scheduler) - 1
            )
        # Adds noise to each simplex representation (Forward diffusion process).
        noisy_simplex = self.noise_scheduler.add_noise(
            simplex, noise, timesteps, norm_relative_position
        )
        # the warper model will scale the timesteps to the correct range.
        timesteps = scale(timesteps, len(self.noise_scheduler))
        # original_timesteps_scaled = scale(original_timesteps, len(self.noise_scheduler))
        inputs.update(
            {"original_timesteps": scale(original_timesteps, len(self.noise_scheduler))}
        )

        inputs.update(
            {
                "timesteps": timesteps,
                "simplex": noisy_simplex,
                "token_rel_positions": norm_relative_position,
            }
        )
        inputs.update({"max_timestep": len(self.noise_scheduler)})
        if self.diffusion_args.self_condition is not None:
            previous_pred = None
            previous_hidden = None
            if np.random.rand(1) > 0.5:
                next_timestep = inputs.pop("timesteps")
                next_simplex = inputs.pop("simplex")
                timesteps = torch.clamp(
                    (next_timestep * len(self.noise_scheduler)) + 1,
                    max=len(self.noise_scheduler) - 1,
                )
                if is_tokenwise_cdcd_check(self.model):
                    timesteps = self.model.warp_timesteps(
                        timesteps, t_max=len(self.noise_scheduler) - 1
                    )
                noisy_simplex = self.noise_scheduler.add_noise(
                    simplex, noise, timesteps, norm_relative_position
                )
                timesteps = scale(timesteps, len(self.noise_scheduler))
                inputs.update(
                    {
                        "timesteps": timesteps,
                        "simplex": noisy_simplex,
                    }
                )
                outputs = model(**inputs, previous_pred=previous_pred)
                logits_projection_fct = lambda x: logits_projection(  # noqa: E731
                    x,
                    self.diffusion_args.sampling_type,
                    self.diffusion_args.top_p,
                    self.diffusion_args.simplex_value,
                    self.diffusion_args.temperature,
                )
                previous_pred = self_condition_preds(
                    self.diffusion_args.self_condition,
                    outputs.logits,
                    logits_projection_fct,
                )
                # following rest of self-conditioning, don't backprop through.
                previous_hidden = outputs.hidden_states.detach()
                # pop timestep/simplex and put the old ones back.
                inputs.update(
                    {
                        "timesteps": next_timestep,
                        "simplex": next_simplex,
                    }
                )
            inputs.update({"previous_pred": previous_pred})
            inputs.update({"previous_hidden": previous_hidden})
        else:
            inputs.update({"previous_pred": None})
            inputs.update({"previous_hidden": None})
            previous_hidden = None
        # NOTE: we do this after computation of self-conditioning to not affect that one.
        inputs.update(
            {"classifier_free_guidance_in_train": self.classifier_free_guidance}
        )
        # re-warp based on previous hidden state
        if is_cdcd_check(self.model):
            timesteps = self.model.warp_timesteps(
                original_timesteps,
                t_max=len(self.noise_scheduler) - 1,
                previous_hidden=previous_hidden,
            )
            noisy_simplex = self.noise_scheduler.add_noise(
                simplex, noise, timesteps, norm_relative_position
            )
            timesteps = scale(timesteps, len(self.noise_scheduler))
            inputs.update(
                {
                    "timesteps": timesteps,
                    "simplex": noisy_simplex,
                    "token_rel_positions": norm_relative_position,
                }
            )
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        # HACK: transformer update
        # if self.do_grad_scaling:
        #     self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)
        return loss.detach() / self.args.gradient_accumulation_steps

    def light_prediction_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        with torch.no_grad():
            inputs = self._prepare_inputs(inputs)
            # Truncate the length if needed.
            if self.data_args.truncation_length > 0:
                inputs["input_ids"] = inputs["input_ids"][
                    :, : -self.data_args.truncation_length
                ]
                inputs["span_mask"] = inputs["span_mask"][
                    :, : -self.data_args.truncation_length
                ]
            # Creates the noisy simplex and timesteps.
            simplex = convert_to_simplex(
                inputs["input_ids"], self.diffusion_args.simplex_value, self.vocab_size
            )
            noise = self.diffusion_args.simplex_value * torch.randn(
                simplex.shape, device=simplex.device, dtype=simplex.dtype
            )
            bsz = simplex.shape[0]
            # Sample a random timestep for each simplex token representation.
            # we use the train timesteps to be consistent with the training process.
            timesteps = torch.randint(
                0,
                len(self.noise_scheduler),
                (bsz, inputs["input_ids"].shape[1])
                if is_tokenwise_cdcd_check(self.model)
                else (bsz,),
                device=simplex.device,
                dtype=torch.int64,
            )
            # original_timesteps = timesteps
            if not is_tokenwise_cdcd_check(self.model):
                timesteps = timesteps[:, None].expand(-1, inputs["input_ids"].shape[1])

            # if we're not doing token warping, just set all relative positions to 1.
            norm_relative_position = torch.ones_like(inputs["input_ids"])

            # if cdcd, we need to wrap the timesteps in a cdf.
            # make sure we scale the timesteps to the correct range!
            if is_cdcd_check(self.model):
                timesteps = self.model.warp_timesteps(
                    timesteps, t_max=len(self.noise_scheduler) - 1
                )

            # Adds noise to each simplex representation (Forward diffusion process).
            noisy_simplex = self.noise_scheduler.add_noise(
                simplex, noise, timesteps, norm_relative_position
            )

            timesteps = scale(timesteps, len(self.noise_scheduler))
            # original_timesteps_scaled = scale(
            #     original_timesteps, len(self.noise_scheduler)
            # )
            # inputs.update({"original_timesteps": original_timesteps_scaled})

            inputs.update(
                {
                    "timesteps": timesteps,
                    "simplex": noisy_simplex,
                    "token_rel_positions": norm_relative_position,
                }
            )
            inputs.update({"max_timestep": len(self.noise_scheduler)})
            if self.diffusion_args.self_condition is not None:
                previous_pred = None
                last_hidden_state = None
                if np.random.rand(1) > 0.5:
                    outputs = model(**inputs, previous_pred=previous_pred)
                    logits_projection_fct = lambda x: logits_projection(  # noqa: E731
                        x,
                        self.diffusion_args.sampling_type,
                        self.diffusion_args.top_p,
                        self.diffusion_args.simplex_value,
                        self.diffusion_args.temperature,
                    )
                    previous_pred = self_condition_preds(
                        self.diffusion_args.self_condition,
                        outputs.logits,
                        logits_projection_fct,
                    )
                    last_hidden_state = outputs.hidden_states
                inputs.update(
                    {
                        "previous_pred": previous_pred,
                        "previous_hidden": last_hidden_state,
                    }
                )
            # NOTE: we do this after computation of self-conditioning to not affect that one.
            inputs.update(
                {"classifier_free_guidance_in_train": self.classifier_free_guidance}
            )
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            return (
                loss.detach()
            )  # no division by gradient accumulation steps for eval. we want per-sample avg loss.

    # TODO: argument for doing one step.
    def prediction_step(
        self,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        model: nn.Module,
        pipeline: List[SimplexDDPMPipeline],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)
        is_conditional_generation = True if "span_mask" in inputs else False
        # full inference.
        with torch.no_grad():
            with self.compute_loss_context_manager():
                for i, x in enumerate(
                    pipeline(
                        batch_size=inputs["input_ids"].shape[0]
                        if is_conditional_generation
                        else self.args.per_device_eval_batch_size,
                        seq_length=self.data_args.max_seq_length
                        - self.data_args.truncation_length,
                        batch=inputs,
                        guidance_scale=self.diffusion_args.guidance_scale,
                        generator=torch.Generator(device=self.args.device).manual_seed(
                            self.args.seed
                        )
                        if self.diffusion_args.generate_with_seed
                        else None,
                        is_generator=False,
                    )
                ):
                    outputs = x
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
        noise_scheduler=None,
        light_eval_dataloader=None,
        do_light_eval=False,
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.
        Works both with or without labels.
        """
        args = self.args
        is_conditional_generation = self.data_args.conditional_generation is not None
        save_prefixes = is_conditional_generation

        prediction_loss_only = (
            prediction_loss_only
            if prediction_loss_only is not None
            else args.prediction_loss_only
        )
        # if eval is called w/o train handle model prep here
        if self.is_deepspeed_enabled and self.model_wrapped is self.model:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

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
            scheduler=noise_scheduler,
            simplex_value=self.diffusion_args.simplex_value,
            top_p=self.diffusion_args.top_p,
            sampling_type=self.diffusion_args.sampling_type,
            is_conditional_generation=is_conditional_generation,
            tokenizer=self.tokenizer,
            classifier_free_uncond_input=self.diffusion_args.classifier_free_uncond_input,
            temperature=self.diffusion_args.temperature,
            guidance_softmax_combination=self.diffusion_args.guidance_softmax_combination,
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

        # light evaluation loop.
        if light_eval_dataloader is not None and do_light_eval:
            for step, inputs in enumerate(light_eval_dataloader):
                # Truncate the length if needed.
                if self.data_args.truncation_length > 0:
                    inputs["input_ids"] = inputs["input_ids"][
                        :, : -self.data_args.truncation_length
                    ]
                    inputs["span_mask"] = inputs["span_mask"][
                        :, : -self.data_args.truncation_length
                    ]
                    max_seq_length = (
                        self.data_args.max_seq_length - self.data_args.truncation_length
                    )
                    assert self.data_args.eval_context_size < max_seq_length
                # predict loss mimicking training.
                loss = self.light_prediction_step(model, inputs)

                if loss is not None:
                    losses = self._nested_gather(loss.repeat(batch_size))
                    losses_host = (
                        losses
                        if losses_host is None
                        else torch.cat((losses_host, losses), dim=0)
                    )
                if (
                    args.eval_accumulation_steps is not None
                    and (step + 1) % args.eval_accumulation_steps == 0
                ):
                    if losses_host is not None:
                        losses = nested_numpify(losses_host)
                        all_losses = (
                            losses
                            if all_losses is None
                            else np.concatenate((all_losses, losses), axis=0)
                        )
                    losses_host = None

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            has_mask = True if "span_mask" in inputs else False

            # Truncate the length if needed.
            if self.data_args.truncation_length > 0:
                inputs["input_ids"] = inputs["input_ids"][
                    :, : -self.data_args.truncation_length
                ]
                inputs["span_mask"] = inputs["span_mask"][
                    :, : -self.data_args.truncation_length
                ]
                max_seq_length = (
                    self.data_args.max_seq_length - self.data_args.truncation_length
                )
                assert self.data_args.eval_context_size < max_seq_length

            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            simplex, logits = self.prediction_step(inputs, model, pipeline=pipeline)
            inputs_decode = self._prepare_input(inputs["input_ids"])
            masks = self._prepare_input(inputs["span_mask"]) if has_mask else None
            if save_prefixes:
                prefixes = (
                    pad_data(
                        [input[~mask] for input, mask in zip(inputs_decode, masks)],
                        self.tokenizer,
                    )
                    if has_mask
                    else None
                )
                prefixes = self._prepare_input(prefixes)
            else:
                prefixes = None
            # Update containers on host
            if prefixes is not None:
                prefixes = self.accelerator.pad_across_processes(
                    prefixes, dim=1, pad_index=self.eos_token_id
                )
                prefixes = self._nested_gather(prefixes)
                prefixes_host = (
                    prefixes
                    if prefixes_host is None
                    else nested_concat(
                        prefixes_host, prefixes, padding_index=self.eos_token_id
                    )
                )
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(
                    inputs_decode, dim=1, pad_index=self.eos_token_id
                )
                inputs_decode = self._nested_gather(inputs_decode)
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(
                        inputs_host, inputs_decode, padding_index=self.eos_token_id
                    )
                )
            # Note that this block should be before masks block, since we need masks here.
            if simplex is not None:
                # In case of having a mask softmax is applied over the simplex non-masked values.
                if has_mask:
                    mask_value = torch.finfo(simplex.dtype).min
                    mask_value = torch.tensor(
                        mask_value, dtype=simplex.dtype, device=simplex.device
                    )
                    simplex = torch.where(masks[:, :, None], simplex, mask_value)
                simplex = F.softmax(simplex, dim=-1)
                if self.preprocess_logits_for_metrics is not None:
                    simplex = self.preprocess_logits_for_metrics(simplex)
                simplex = self.accelerator.pad_across_processes(
                    simplex, dim=1, pad_index=self.eos_token_id
                )
                simplex = self._nested_gather(simplex)
                # TODO: note that this is no more a simplex, but the processed one.
                simplex_host = (
                    simplex
                    if simplex_host is None
                    else nested_concat(
                        simplex_host, simplex, padding_index=self.eos_token_id
                    )
                )
            if masks is not None:
                masks = self.accelerator.pad_across_processes(masks, dim=1, pad_index=0)
                masks = self._nested_gather(masks)
                # We pad masks with False tokens.
                masks_host = (
                    masks
                    if masks_host is None
                    else nested_concat(masks_host, masks, padding_index=0)
                )
            if logits is not None:
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits)
                logits = self.accelerator.pad_across_processes(
                    logits, dim=1, pad_index=self.eos_token_id
                )
                logits = self._nested_gather(logits)
                logits_host = (
                    logits
                    if logits_host is None
                    else nested_concat(
                        logits_host, logits, padding_index=self.eos_token_id
                    )
                )

            self.control = self.callback_handler.on_prediction_step(
                args, self.state, self.control
            )

        # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
        if (
            args.eval_accumulation_steps is not None
            and (step + 1) % args.eval_accumulation_steps == 0
        ):
            if logits_host is not None:
                logits = nested_numpify(logits_host)
                all_logits = (
                    logits
                    if all_logits is None
                    else nested_concat(
                        all_logits, logits, padding_index=self.eos_token_id
                    )
                )
            if simplex_host is not None:
                simplex = nested_numpify(simplex_host)
                all_simplex = (
                    simplex
                    if all_simplex is None
                    else nested_concat(
                        all_simplex, simplex, padding_index=self.eos_token_id
                    )
                )
            if inputs_host is not None:
                inputs_decode = nested_numpify(inputs_host)
                all_inputs = (
                    inputs_decode
                    if all_inputs is None
                    else nested_concat(
                        all_inputs, inputs_decode, padding_index=self.eos_token_id
                    )
                )
            if masks_host is not None:
                masks = nested_numpify(masks_host)
                all_masks = (
                    masks
                    if all_masks is None
                    else nested_concat(all_masks, masks, padding_index=0)
                )
            if prefixes_host is not None:
                prefixes = nested_numpify(prefixes_host)
                all_prefixes = (
                    prefixes
                    if all_prefixes is None
                    else nested_concat(
                        all_prefixes, prefixes, padding_index=self.eos_token_id
                    )
                )

            # Set back to None to begin a new accumulation
            logits_host, simplex_host, inputs_host, masks_host, prefixes_host = (
                None,
                None,
                None,
                None,
                None,
            )

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            all_losses = nested_numpify(losses_host)
        if logits_host is not None:
            all_logits = nested_numpify(logits_host)
        if simplex_host is not None:
            all_simplex = nested_numpify(simplex_host)
        if inputs_host is not None:
            all_inputs = nested_numpify(inputs_host)
        if masks_host is not None:
            all_masks = nested_numpify(masks_host)
        if prefixes_host is not None:
            all_prefixes = nested_numpify(prefixes_host)

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif (
            isinstance(eval_dataset, IterableDatasetShard)
            and getattr(eval_dataset, "num_examples", 0) > 0
        ):
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

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
                        all_simplex,
                        skip_special_tokens=self.data_args.skip_special_tokens,
                    )
                }
            )
            results.update(
                {
                    "pred_texts_from_logits": self.tokenizer.batch_decode(
                        all_logits,
                        skip_special_tokens=self.data_args.skip_special_tokens,
                    )
                }
            )
        if is_conditional_generation:
            results.update(
                {
                    "gold_texts_masked": [
                        self.tokenizer.decode(
                            input[mask],
                            skip_special_tokens=self.data_args.skip_special_tokens,
                        )
                        for mask, input in zip(all_masks, all_inputs)
                    ]
                }
            )
            if save_prefixes:
                results.update(
                    {
                        "prefixes": [
                            self.tokenizer.decode(
                                x, skip_special_tokens=True
                            )  # self.data_args.skip_special_tokens)
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
        light_eval_dataloader = self.get_light_eval_dataloader(eval_dataset)
        start_time = time.time()

        outputs = []
        timesteps = self.inference_timesteps
        for timestep, noise_scheduler in zip(
            timesteps, self.inference_noise_schedulers
        ):
            output = self.evaluation_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if self.compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
                noise_scheduler=noise_scheduler,
                light_eval_dataloader=light_eval_dataloader,
                do_light_eval=timestep
                == timesteps[
                    0
                ],  # we only need the loss once, since it is the same for all timesteps
            )
            outputs.append(output)
            key_prefix = f"inference_{timestep}_"
            metrics = {key_prefix + k: v for k, v in output.metrics.items()}
            results = {key_prefix + k: v for k, v in output.results.items()}
            # reset output with new metrics / results
            output = EvalLoopOutput(
                logits=output.logits,
                simplex=output.simplex,
                input_ids=output.input_ids,
                metrics=metrics,
                num_samples=output.num_samples,
                results=results,
            )

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
            self.control = self.callback_handler.on_evaluate(
                self.args, self.state, self.control, output.metrics
            )
            self._memory_tracker.stop_and_update_metrics(output.metrics)

            # Save the results
            self.save_metrics(
                GENERATION_RESULTS + "_" + key_prefix + metric_key_prefix,
                output.results,
            )
            logger.info("Results are saved now")

        # log outside so we can group generations together
        if self.args.log_generated_texts:
            length = len(outputs[0].logits)
            results = {
                f"{k}_inference_{i}": v
                for o, i in zip(outputs, timesteps)
                for k, v in o.results.items()
            }
            self.log_results_to_tensorboard(self.state, length, results)

        return output.metrics

    def log_results_to_tensorboard(self, state, length, results):
        # TODO: we need to fix this which happens during the only eval option.
        if self.tb_writer.tb_writer is None:
            return
        for i in range(length):
            total_text = ""
            for k, v in results.items():
                total_text += f"*** {k} ***: {v[i]}" + "  \n"
            self.tb_writer.tb_writer.add_text(
                f"sample_{i}", total_text, state.global_step
            )

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
            train_dataset = self._remove_unused_columns(
                train_dataset, description="training"
            )
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="training"
            )

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

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
            eval_dataset = self._remove_unused_columns(
                eval_dataset, description="evaluation"
            )
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="evaluation"
            )

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last

        return self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))

    def get_light_eval_dataloader(
        self, eval_dataset: Optional[Dataset] = None
    ) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].
        Used for the light evaluation, which matches masking with training.
        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator("train")

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(
                eval_dataset, description="evaluation"
            )
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="evaluation"
            )

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last

        return self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))

    def _inner_training_loop(
        self,
        batch_size=None,
        args=None,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = (
            args.train_batch_size * args.gradient_accumulation_steps * args.world_size
        )

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = (
                len_dataloader // args.gradient_accumulation_steps
            )
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
                max_steps = math.ceil(
                    args.num_train_epochs * num_update_steps_per_epoch
                )
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = (
                    self.num_examples(train_dataloader) * args.num_train_epochs
                )
        elif (
            args.max_steps > 0
        ):  # Rely on max_steps when dataloader does not have a working size
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

        # delay_optimizer_creation = (
        #     self.sharded_ddp is not None
        #     and self.sharded_ddp != ShardedDDPOption.SIMPLE
        #     or is_sagemaker_mp_enabled()
        #     or self.fsdp is not None
        # )
        # HACK: transformer version update
        delay_optimizer_creation = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps
            )

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # Fairscale Sharded DDP, FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(
                        self.model, self.optimizer
                    )
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # deepspeed ckpt loading
        if resume_from_checkpoint is not None and self.is_deepspeed_enabled:
            deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
        )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(
            f"  Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (
                    num_update_steps_per_epoch
                )
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step"
            )
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(
                f"  Continuing training from global step {self.state.global_step}"
            )
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(
                        total=steps_trained_in_current_epoch
                    )
                    steps_trained_progress_bar.set_description(
                        "Skipping the first batches"
                    )

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
            assignments = (
                trial.assignments
                if self.hp_search_backend == HPSearchBackend.SIGOPT
                else trial
            )
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

        self.control = self.callback_handler.on_train_begin(
            args, self.state, self.control
        )

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                for _ in train_dataloader:
                    break

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(
                args, self.state, self.control
            )

            if (
                epoch == epochs_trained
                and resume_from_checkpoint is not None
                and steps_trained_in_current_epoch == 0
            ):
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            if skip_first_batches is not None and steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(
                    epoch_iterator, steps_trained_in_current_epoch
                )
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

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
                    self.control = self.callback_handler.on_step_begin(
                        args, self.state, self.control
                    )

                with self.accelerator.accumulate(model):
                    tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (
                        1 + self.state.global_step - self._globalstep_last_logged
                    )
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # should this be under the accumulate context manager?
                # the `or` condition of `steps_in_epoch <= args.gradient_accumulation_steps` is not covered
                # in accelerate
                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # HACK: transformer update
                        # deepspeed does its own clipping
                        # if self.do_grad_scaling:
                        #     # Reduce gradients first for XLA
                        #     if is_torch_tpu_available():
                        #         gradients = xm._fetch_gradients(self.optimizer)
                        #         xm.all_reduce(
                        #             "sum", gradients, scale=1.0 / xm.xrt_world_size()
                        #         )
                        #     # AMP: gradients need unscaling
                        #     self.scaler.unscale_(self.optimizer)

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if is_torch_tpu_available():
                        if self.do_grad_scaling:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            xm.optimizer_step(self.optimizer)
                    # HACK: transformer update
                    # elif self.do_grad_scaling:
                    #     scale_before = self.scaler.get_scale()
                    #     self.scaler.step(self.optimizer)
                    #     self.scaler.update()
                    #     scale_after = self.scaler.get_scale()
                    #     optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()
                        optimizer_was_run = (
                            not self.accelerator.optimizer_step_was_skipped
                        )

                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(
                            self.lr_scheduler,
                            torch.optim.lr_scheduler.ReduceLROnPlateau,
                        ):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(
                        args, self.state, self.control
                    )

                    self._maybe_log_save_evaluate(
                        tr_loss, model, trial, epoch, ignore_keys_for_eval
                    )
                else:
                    self.control = self.callback_handler.on_substep_end(
                        args, self.state, self.control
                    )

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(
                args, self.state, self.control
            )
            self._maybe_log_save_evaluate(
                tr_loss, model, trial, epoch, ignore_keys_for_eval
            )

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

        logger.info(
            "\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n"
        )
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(
            use_mtime=False, output_dir=run_dir
        )

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if (
            self.args.should_save
            and self.state.best_model_checkpoint is not None
            and self.args.save_total_limit == 1
        ):
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(
                        f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit"
                    )
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(
            args, self.state, self.control
        )

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def create_optimizer(self):
        from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
        from transformers.trainer_pt_utils import get_parameter_names

        # overriden
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )

            # only training warping parameters...
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (
                            n in decay_parameters
                            and p.requires_grad
                            and not ("cdf" in n or "linear_l" in n or "position_l" in n)
                        )
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": optimizer_kwargs["lr"],
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (
                            n not in decay_parameters
                            and p.requires_grad
                            and not ("cdf" in n or "linear_l" in n or "position_l" in n)
                        )
                    ],
                    "weight_decay": 0.0,
                    "lr": optimizer_kwargs["lr"],
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (
                            ("cdf" in n or "linear_l" in n or "position_l" in n)
                            and p.requires_grad
                        )
                    ],
                    "weight_decay": 0.0,
                    "lr": 1e-3,
                },
            ]

            optimizer_kwargs.pop("lr")

            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )

        return self.optimizer
