import math
import time
import warnings
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer

with warnings.catch_warnings():
    warnings.simplefilter(action="ignore", category=FutureWarning)
    from transformers.deepspeed import deepspeed_init

from transformers.integrations import TensorBoardCallback
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    find_batch_size,
    nested_concat,
    nested_detach,
    nested_numpify,
)
from transformers.trainer_utils import denumpify_detensorize, has_length, speed_metrics
from transformers.utils import (
    is_apex_available,
    is_datasets_available,
    is_sagemaker_mp_enabled,
    logging,
)

from sdlm.inference.inference_utils import (
    logits_projection,
    predict_conditional_generated,
)
from sdlm.models.utils import is_cdcd_check
from sdlm.pipelines.simplex_ddpm import SimplexDDPMClassifierGuidancePipeline
from sdlm.utils import convert_to_simplex, pad_data, scale, self_condition_preds

if is_apex_available():
    from apex import amp

if is_datasets_available():
    import datasets

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
        self.original_data_collator = self.data_collator
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
        # TODO: control seed.
        self.self_cond_generator = np.random.default_rng(42)

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
        # testing just sampling the same place. This better matches reality.
        if True:  # np.random.rand(1) > 0.5:
            timesteps = torch.randint(
                0,
                len(self.noise_scheduler),
                (bsz, inputs["input_ids"].shape[1])
                if False  # is_tokenwise_cdcd_check(self.model)
                else (bsz,),
                device=simplex.device,
                dtype=torch.int64,
            )
            timesteps = timesteps[:, None].expand(-1, inputs["input_ids"].shape[1])
        else:
            timesteps = torch.randint(
                0,
                len(self.noise_scheduler),
                (bsz, inputs["input_ids"].shape[1])
                if True  # is_tokenwise_cdcd_check(self.model)
                else (bsz,),
                device=simplex.device,
                dtype=torch.int64,
            )
        # expand out timesteps to match tokenwise setup
        # if True:  # not is_tokenwise_cdcd_check(self.model):
        #     timesteps = timesteps[:, None].expand(-1, inputs["input_ids"].shape[1])

        # save original timesteps for warping
        original_timesteps = timesteps
        # warp timesteps according to cdf
        # we re-scale the timesteps to the correct range.
        # the -1 is due to the timestep should be in range [0, 5000)
        if is_cdcd_check(self.model):
            input_ids = inputs["input_ids"]
            span_mask = inputs["span_mask"]
            token_input = torch.where(
                (input_ids * span_mask) > 1, self.tokenizer.pad_token_id, input_ids
            )
            timesteps = self.model.warp_timesteps(
                timesteps,
                token_input=token_input,
                span_mask=span_mask,
                t_max=len(self.noise_scheduler) - 1,
            )
        # Adds noise to each simplex representation (Forward diffusion process).
        noisy_simplex = self.noise_scheduler.add_noise(simplex, noise, timesteps)
        # the warper model will scale the timesteps to the correct range.
        timesteps = scale(timesteps, len(self.noise_scheduler))
        # original_timesteps_scaled = scale(original_timesteps, len(self.noise_scheduler))
        # inputs.update(
        #     {"original_timesteps": scale(original_timesteps, len(self.noise_scheduler))}
        # )

        inputs.update(
            {
                "timesteps": timesteps,
                "simplex": noisy_simplex,
            }
        )
        # inputs.update({"max_timestep": len(self.noise_scheduler)})
        if self.diffusion_args.self_condition is not None:
            previous_pred = None
            # previous_hidden = None
            if self.self_cond_generator.random(1) > 0.5:
                next_timestep = inputs.pop("timesteps")
                next_simplex = inputs.pop("simplex")
                timesteps = torch.clamp(
                    (next_timestep * len(self.noise_scheduler)) + 1,
                    max=len(self.noise_scheduler) - 1,
                )
                if is_cdcd_check(self.model):
                    input_ids = inputs["input_ids"]
                    span_mask = inputs["span_mask"]
                    token_input = torch.where(
                        (input_ids * span_mask) > 1,
                        self.tokenizer.pad_token_id,
                        input_ids,
                    )
                    timesteps = self.model.warp_timesteps(
                        timesteps,
                        token_input=token_input,
                        span_mask=span_mask,
                        t_max=len(self.noise_scheduler) - 1,
                    )
                noisy_simplex = self.noise_scheduler.add_noise(
                    simplex, noise, timesteps
                )
                timesteps = scale(timesteps, len(self.noise_scheduler))
                inputs.update(
                    {
                        "timesteps": timesteps,
                        "simplex": noisy_simplex,
                    }
                )
                # we don't backprop through this.
                with torch.no_grad():
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
                ).detach()
                # following rest of self-conditioning, don't backprop through.
                # previous_hidden = outputs.hidden_states.detach()
                # pop timestep/simplex and put the old ones back.
                inputs.update(
                    {
                        "timesteps": next_timestep,
                        "simplex": next_simplex,
                    }
                )
            inputs.update({"previous_pred": previous_pred})
            # inputs.update({"previous_hidden": previous_hidden})
        else:
            inputs.update({"previous_pred": None})
            # inputs.update({"previous_hidden": None})
            # previous_hidden = None
        # NOTE: we do this after computation of self-conditioning to not affect that one.
        # inputs.update(
        #     {"classifier_free_guidance_in_train": self.classifier_free_guidance}
        # )
        # re-warp based on previous hidden state
        if is_cdcd_check(self.model):
            # replace masked tokens with <mask> token.
            input_ids = inputs["input_ids"]
            span_mask = inputs["span_mask"]
            token_input = torch.where(
                (input_ids * span_mask) > 1, self.tokenizer.pad_token_id, input_ids
            )
            timesteps = self.model.warp_timesteps(
                original_timesteps,
                t_max=len(self.noise_scheduler) - 1,
                token_input=token_input,
                span_mask=span_mask,
            )
            noisy_simplex = self.noise_scheduler.add_noise(simplex, noise, timesteps)
            timesteps = scale(timesteps, len(self.noise_scheduler))
            inputs.update(
                {
                    "timesteps": timesteps,
                    "simplex": noisy_simplex,
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
            # randomly flip between random batchwise and tokenwise timesteps.
            if True:
                timesteps = torch.randint(
                    0,
                    len(self.noise_scheduler),
                    (bsz, inputs["input_ids"].shape[1])
                    if False  # is_tokenwise_cdcd_check(self.model)
                    else (bsz,),
                    device=simplex.device,
                    dtype=torch.int64,
                )
                timesteps = timesteps[:, None].expand(-1, inputs["input_ids"].shape[1])
            else:
                timesteps = torch.randint(
                    0,
                    len(self.noise_scheduler),
                    (bsz, inputs["input_ids"].shape[1])
                    if True  # is_tokenwise_cdcd_check(self.model)
                    else (bsz,),
                    device=simplex.device,
                    dtype=torch.int64,
                )
            # original_timesteps = timesteps

            # if cdcd, we need to wrap the timesteps in a cdf.
            # make sure we scale the timesteps to the correct range!
            if is_cdcd_check(self.model):
                input_ids = inputs["input_ids"]
                span_mask = inputs["span_mask"]
                token_input = torch.where(
                    (input_ids * span_mask) > 1, self.tokenizer.pad_token_id, input_ids
                )
                timesteps = self.model.warp_timesteps(
                    timesteps,
                    t_max=len(self.noise_scheduler) - 1,
                    token_input=token_input,
                    span_mask=span_mask,
                )

            # Adds noise to each simplex representation (Forward diffusion process).
            noisy_simplex = self.noise_scheduler.add_noise(simplex, noise, timesteps)

            timesteps = scale(timesteps, len(self.noise_scheduler))
            # original_timesteps_scaled = scale(
            #     original_timesteps, len(self.noise_scheduler)
            # )
            # inputs.update({"original_timesteps": original_timesteps_scaled})

            inputs.update(
                {
                    "timesteps": timesteps,
                    "simplex": noisy_simplex,
                }
            )
            # inputs.update({"max_timestep": len(self.noise_scheduler)})
            if self.diffusion_args.self_condition is not None:
                previous_pred = None
                # last_hidden_state = None
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
                    # last_hidden_state = outputs.hidden_states
                inputs.update(
                    {
                        "previous_pred": previous_pred,
                        # "previous_hidden": last_hidden_state,
                    }
                )
            # NOTE: we do this after computation of self-conditioning to not affect that one.
            # inputs.update(
            #     {"classifier_free_guidance_in_train": self.classifier_free_guidance}
            # )
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
        pipeline: List[SimplexDDPMClassifierGuidancePipeline],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)
        # full inference.
        with torch.no_grad():
            with self.compute_loss_context_manager():
                for i, x in enumerate(
                    pipeline(
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
                        use_gumbel_softmax=self.diffusion_args.use_gumbel_softmax,
                        do_hard_sample=self.diffusion_args.do_hard_sample,
                        softmax_temperature=self.diffusion_args.softmax_temperature,
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

        pipeline = SimplexDDPMClassifierGuidancePipeline(
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
            classifier_model_name_or_path=self.diffusion_args.classifier_model_name_or_path,
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
        self.data_collator = self.original_data_collator("train")
        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        self.data_collator = self.original_data_collator("eval")
        return super().get_eval_dataloader(eval_dataset)

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
        data_collator = self.original_data_collator("train")

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

    def create_optimizer(self):
        from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
        from transformers.trainer_pt_utils import get_parameter_names

        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is not None:
            return self.optimizer

        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args
        )

        # override to apply higher lr to timestep_embed and cdcd cdf
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in opt_model.named_parameters()
                    if (
                        n in decay_parameters
                        and p.requires_grad
                        and not ("timestep_embed" in n or "cdf" in n)
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
                        and not ("timestep_embed" in n or "cdf" in n)
                    )
                ],
                "weight_decay": 0.0,
                "lr": optimizer_kwargs["lr"],
            },
            {
                "params": [
                    p
                    for n, p in opt_model.named_parameters()
                    if (("timestep_embed" in n) and p.requires_grad)
                ],
                "weight_decay": 0.0,
                "lr": self.args.timestep_embed_lr or self.args.learning_rate,
            },
        ]
        # check cdcd
        cdf_params = [
            p
            for n, p in opt_model.named_parameters()
            if (("cdf" in n) and p.requires_grad)
        ]
        if cdf_params:
            optimizer_grouped_parameters.append(
                {
                    "params": cdf_params,
                    "weight_decay": 0.0,
                    "lr": 1e-3,
                }
            )

        optimizer_kwargs.pop("lr")

        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer
