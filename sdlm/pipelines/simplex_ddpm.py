from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput

from sdlm.inference.inference_utils import logits_projection
from sdlm.models.utils import check_tokenizer_equal, is_cdcd_check, load_classifier
from sdlm.utils import scale, self_condition_preds, convert_to_simplex


@dataclass
class SimplexDiffusionPipelineOutput(BaseOutput):
    """
    Output class for simplex diffusion pipelines.
    Args:
        simplex (`np.ndarray`)
            numpy array showing the denoised simplex representation.
        logits (`np.ndarray`) final generated logits before applying the projection.
    """

    simplex: np.ndarray
    logits: np.ndarray
    loss: np.ndarray


def yield_func(x):
    yield x


class SimplexDDPMPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    Parameters:
        model: Model architecture to denoise the latents (encoded token ids).
        scheduler ([`SchedulerMixin`]): A scheduler to denoise the encoded latent.
    """

    def __init__(
        self,
        model,
        scheduler,
        simplex_value,
        top_p,
        sampling_type,
        is_conditional_generation,
        tokenizer,
        classifier_free_uncond_input,
        temperature,
        guidance_softmax_combination,
    ):
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler)
        self.simplex_value = simplex_value
        self.top_p = top_p
        self.sampling_type = sampling_type
        self.is_conditional_generation = is_conditional_generation
        self.tokenizer = tokenizer
        self.classifier_free_uncond_input = classifier_free_uncond_input
        self.temperature = temperature
        self.guidance_softmax_combination = guidance_softmax_combination

    @torch.inference_mode()
    def __call__(
        self,
        seq_length: int = 512,
        generator: Optional[torch.Generator] = None,
        batch: Optional[torch.FloatTensor] = None,
        guidance_scale: float = 1.0,
        is_generator: bool = False,
    ) -> Union[SimplexDiffusionPipelineOutput, Tuple]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            seq_length: (`int`), sequence length for the generated samples.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            batch (`torch.FloatTensor`): batch of input data, mostly used in the conditional generation setting.
        Returns:
            [`~pipeline_utils.SimplexDiffusionPipelineOutput`]: returns the generated simplex.
        """
        # Classifier_free guidance works only in the conditional generation case.
        classifier_free_guidance = (
            guidance_scale > 1.0 and self.is_conditional_generation
        )
        """
        if classifier_free_guidance:
            # Makes unconditional input for max sequence length, later we truncate it.
            uncond_input = self.tokenizer(
                [""] * batch_size, padding="max_length", max_length=seq_length, return_tensors="pt"
            ).to(self.device)
            # Converts this to a simplex (batch_size, max_seq, vocab_size)
            uncond_simplex = convert_to_simplex(uncond_input["input_ids"], self.simplex_value, self.model.config.vocab_size)
        """
        # Sample gaussian noise to begin loop
        vocab_size = self.model.config.vocab_size
        if batch is not None:
            # TODO(rabeeh): is giving the length cheating for this setting?
            # Adapts the sequence length to the given `span_mask`'s length.
            seq_length = batch["input_ids"].shape[1]
        # idk why i have the bsz argument.
        batch_size = batch["input_ids"].shape[0]
        simplex_shape = (batch_size, seq_length, vocab_size)
        simplex = self.simplex_value * torch.randn(
            simplex_shape, generator=generator, device=self.device
        )
        if self.model.config.self_condition is not None:
            previous_pred = torch.zeros(
                (batch_size, seq_length, vocab_size), device=self.device
            )
        logits_projection_fct = lambda x: logits_projection(  # noqa: E731
            x, self.sampling_type, self.top_p, self.simplex_value, self.temperature
        )
        losses = []
        previous_hidden = None

        warped_steps = []
        prev_t = 0
        for t in self.progress_bar(self.scheduler.timesteps):
            original_t = torch.tensor([t], device=self.device).expand(
                batch_size, seq_length
            )
            if is_cdcd_check(self.model):
                # warp timesteps based on cdf
                # we are in inference mode, anything in span_mask is to gen.
                token_inputs = torch.where(
                    batch["span_mask"], self.tokenizer.pad_token_id, batch["input_ids"]
                )
                t = self.model.warp_timesteps(
                    original_t,
                    t_min=0,
                    t_max=len(self.scheduler) - 1,
                    token_input=token_inputs,
                    span_mask=batch["span_mask"],
                )
            else:
                t = original_t
            t_scaled = scale(t, len(self.scheduler))
            warped_steps.append(t)
            """
            if classifier_free_guidance:
                if self.classifier_free_uncond_input == "empty_token":
                    uncond_input = uncond_simplex[:, : batch["input_ids"].shape[1], :]
                elif self.classifier_free_uncond_input == "noisy_simplex":
                    uncond_input = self.simplex_value * torch.randn(simplex.shape, generator=generator, device=self.device)
                else:
                    raise NotImplementedError
            """
            # 1. predict noise model_output. Note we need not to pass the input_ids in case of
            # unconditional generation since the loss would be computed and it should not.
            model_output = self.model(
                input_ids=batch["input_ids"]
                if self.is_conditional_generation
                else None,
                span_mask=batch["span_mask"]
                if self.is_conditional_generation
                else None,
                simplex=simplex,
                timesteps=t_scaled,
                previous_pred=previous_pred
                if self.model.config.self_condition
                else None,
                classifier_free_guidance=classifier_free_guidance,
                reduce_loss="none",
                max_timestep=len(self.scheduler),
                previous_hidden=previous_hidden,
            )
            model_output_logits = model_output.logits
            previous_hidden = model_output.hidden_states

            # Performs classifier-free guidance.
            if classifier_free_guidance:
                logits_uncond, logits_pred = model_output_logits.chunk(2)
                if self.guidance_softmax_combination:
                    model_output_logits = F.softmax(
                        logits_uncond, dim=-1
                    ) + guidance_scale * (
                        F.softmax(logits_pred, dim=-1)
                        - F.softmax(logits_uncond, dim=-1)
                    )
                else:
                    model_output_logits = logits_uncond + guidance_scale * (
                        logits_pred - logits_uncond
                    )

            if self.model.config.self_condition is not None:
                if classifier_free_guidance:
                    prev_output_logits = model_output.logits.chunk(2)[1]
                else:
                    prev_output_logits = model_output_logits

                previous_pred = self_condition_preds(
                    self.model.config.self_condition,
                    prev_output_logits,
                    logits_projection_fct,
                )

            # Projection.
            projected_logits = logits_projection_fct(model_output_logits)

            old_simplex = simplex

            # 2. compute previous logits: x_t -> x_t-1
            noise = self.simplex_value * torch.randn(
                simplex_shape, generator=generator, device=self.device
            )
            if is_cdcd_check(self.model):
                # warp timesteps based on cdf
                token_inputs = torch.where(
                    batch["span_mask"], self.tokenizer.pad_token_id, batch["input_ids"]
                )
                prev_t = self.model.warp_timesteps(
                    original_t - 1,
                    t_min=0,
                    t_max=len(self.scheduler) - 1,
                    token_input=token_inputs,
                    span_mask=batch["span_mask"],
                ).long()
                # since the tokenwise can do some wild stuff.
                prev_t = torch.clamp(prev_t, min=0, max=len(self.scheduler) - 1)
            else:
                prev_t = original_t - 1
            simplex = self.scheduler.step(
                projected_logits,
                t,
                prev_t,
                noise,
                generator=generator,
            ).prev_sample

            # keep loss for logging
            losses.append(model_output.loss.detach().cpu())

            # yield over it. (prolly not optimal, but whatever)
            yield SimplexDiffusionPipelineOutput(
                simplex=old_simplex, logits=model_output_logits, loss=losses[-1]
            )
        # we take the mean loss over all timesteps
        loss = torch.stack(losses, dim=0)
        # from matplotlib import pyplot as plt
        # warped_steps = torch.stack(warped_steps, dim=0)
        # for i in range(warped_steps.shape[1]):
        #     plt.plot(warped_steps[:, i, 256:].cpu())
        #     plt.savefig(f"warps_prefix_tokenwise/warped_{i}.png")
        #     plt.clf()
        return SimplexDiffusionPipelineOutput(
            simplex=simplex, logits=model_output_logits, loss=loss
        )


class SimplexDDPMClassifierGuidancePipeline(SimplexDDPMPipeline):
    def __init__(
        self,
        model,
        scheduler,
        simplex_value,
        top_p,
        sampling_type,
        is_conditional_generation,
        tokenizer,
        classifier_free_uncond_input,
        temperature,
        guidance_softmax_combination,
        classifier_model_name_or_path,
    ) -> None:
        super().__init__(
            model,
            scheduler,
            simplex_value,
            top_p,
            sampling_type,
            is_conditional_generation,
            tokenizer,
            classifier_free_uncond_input,
            temperature,
            guidance_softmax_combination,
        )
        self.classifier = None
        if classifier_model_name_or_path is not None:
            classifier_tokenizer, classifier = load_classifier(
                classifier_model_name_or_path
            )
            check_tokenizer_equal(self.tokenizer, classifier_tokenizer)
            self.classifier = classifier.to(self.device)

    def get_reward(
        self,
        logits: torch.FloatTensor,
        use_gumbel_softmax: bool,
        do_hard_sample: bool,
        softmax_temperature: float,
        one_hot: Optional[torch.Tensor] = None,
        span_mask: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        logits = logits.to(torch.bfloat16)
        logits.requires_grad = True
        if use_gumbel_softmax:
            simplex = F.gumbel_softmax(
                logits, tau=softmax_temperature, hard=do_hard_sample, dim=-1
            )
        else:
            simplex = torch.softmax(logits / softmax_temperature, dim=-1)
        # mask out context
        if span_mask is not None:
            simplex = torch.where(span_mask.unsqueeze(-1), simplex, one_hot)
        # forcibly add eos token to the simplex
        # eos_token = torch.nn.functional.one_hot(
        #     torch.tensor(self.tokenizer.eos_token_id),
        #     num_classes=self.classifier.config.vocab_size,
        # ).to(simplex.device)
        # eos_token = eos_token.unsqueeze(0).unsqueeze(0).expand_as(simplex)
        # simplex = torch.cat([simplex, eos_token], dim=1)
        inputs_embeds = F.linear(
            simplex, self.classifier.model.get_input_embeddings().weight.data.T
        )
        # forward pass through reward model
        reward = self.classifier(inputs_embeds=inputs_embeds).logits
        return reward
    

    @torch.no_grad()
    def __call__(
        self,
        seq_length: int = 512,
        generator: Optional[torch.Generator] = None,
        batch: Optional[torch.FloatTensor] = None,
        guidance_scale: float = 1.0,
        is_generator: bool = False,
        use_gumbel_softmax: bool = False,
        do_hard_sample: bool = False,
        softmax_temperature: float = 1.0,
        use_ddim_sampling: bool = False,
        num_guidance_steps: int = 5,
    ) -> Union[SimplexDiffusionPipelineOutput, Tuple]:
        # check for classifier guidance
        use_classifier_guidance = self.classifier is not None and guidance_scale > 0.0

        # NOTE: copied from SimplexDDPMPipeline
        # Sample gaussian noise to begin loop
        vocab_size = self.model.config.vocab_size
        if batch is not None:
            # TODO(rabeeh): is giving the length cheating for this setting?
            # Adapts the sequence length to the given `span_mask`'s length.
            seq_length = batch["input_ids"].shape[1]
        # idk why i have the bsz argument.
        batch_size = batch["input_ids"].shape[0]
        simplex_shape = (batch_size, seq_length, vocab_size)
        # simplex ~ N(0, kI)
        simplex = self.simplex_value * torch.randn(
            simplex_shape, generator=generator, device=self.device
        )
        if self.model.config.self_condition is not None:
            previous_pred = torch.zeros(
                (batch_size, seq_length, vocab_size), device=self.device
            )
        # logits -> hard sampled k / -k
        logits_projection_fct = lambda x: logits_projection(  # noqa: E731
            x, self.sampling_type, self.top_p, self.simplex_value, self.temperature
        )
        losses = []
        previous_hidden = None

        warped_steps = []
        prev_t = 0
        all_rewards = []
        for t in self.progress_bar(self.scheduler.timesteps):
            original_t = torch.tensor([t], device=self.device).expand(
                batch_size, seq_length
            )
            if is_cdcd_check(self.model):
                # warp timesteps based on cdf
                # we are in inference mode, anything in span_mask is to gen.
                token_inputs = torch.where(
                    batch["span_mask"], 50264, batch["input_ids"]
                )
                t = self.model.warp_timesteps(
                    original_t,
                    t_min=0,
                    t_max=len(self.scheduler) - 1,
                    token_input=token_inputs,
                    span_mask=batch["span_mask"],
                )
            else:
                t = original_t
            t_scaled = scale(t, len(self.scheduler))
            warped_steps.append(t)

            # 1. predict noise model_output. Note we need not to pass the input_ids in case of
            # unconditional generation since the loss would be computed and it should not.
            model_output = self.model(
                input_ids=batch["input_ids"]
                if self.is_conditional_generation
                else None,
                span_mask=batch["span_mask"]
                if self.is_conditional_generation
                else None,
                simplex=simplex,
                timesteps=t_scaled,
                previous_pred=previous_pred
                if self.model.config.self_condition
                else None,
                classifier_free_guidance=False,
                reduce_loss="none",
                max_timestep=len(self.scheduler),
                previous_hidden=previous_hidden,
            )
            model_output_logits = model_output.logits
            previous_hidden = model_output.hidden_states

            # NOTE: classifier guidance!
            # compute one_hot
            span_mask = batch["span_mask"]
            one_hot = F.one_hot(batch["input_ids"], len(self.tokenizer)).to(
                torch.bfloat16
            )
            model_output_logits = model_output_logits.to(torch.bfloat16)

            if use_classifier_guidance:
                # use torch.optim api
                model_output_logits = torch.nn.Parameter(model_output_logits)
                optimizer = torch.optim.SGD([model_output_logits], lr=guidance_scale)
                 # guidance
                with torch.enable_grad():
                    for _ in range(num_guidance_steps):
                        reward = self.get_reward(
                            logits=model_output_logits,
                            use_gumbel_softmax=use_gumbel_softmax,
                            do_hard_sample=do_hard_sample,
                            softmax_temperature=softmax_temperature,
                            one_hot=one_hot,
                            span_mask=span_mask,
                        )
                        # all_rewards.append(reward.detach().cpu())
                        reward = reward.sum().neg()
                        reward.backward()
                        optimizer.step()
                        
                model_output_logits = model_output_logits.data

            if self.model.config.self_condition is not None:
                prev_output_logits = model_output_logits
                previous_pred = self_condition_preds(
                    self.model.config.self_condition,
                    prev_output_logits,
                    logits_projection_fct,
                )

            old_simplex = simplex

            # 2. compute previous logits: x_t -> x_t-1
            if is_cdcd_check(self.model):
                # warp timesteps based on cdf
                token_inputs = torch.where(
                    batch["span_mask"], 50264, batch["input_ids"]
                )
                prev_t = self.model.warp_timesteps(
                    original_t - 1,
                    t_min=0,
                    t_max=len(self.scheduler) - 1,
                    token_input=token_inputs,
                    span_mask=batch["span_mask"],
                ).long()
                # since the tokenwise can do some wild stuff.
                prev_t = torch.clamp(prev_t, min=0, max=len(self.scheduler) - 1)
            else:
                prev_t = original_t - 1

            if not use_ddim_sampling:
                # normal tess
                noise = self.simplex_value * torch.randn(
                    simplex_shape, generator=generator, device=self.device
                )
                # Projection.
                projected_logits = logits_projection_fct(model_output_logits)
                simplex = self.scheduler.step(
                    projected_logits,
                    t,
                    prev_t,
                    noise,
                    generator=generator,
                ).prev_sample
            else:
                # input: noisy k / -k
                # output: clean k / -k
                x_t = old_simplex
                x_0_hat = (
                    2 * self.simplex_value * torch.softmax(model_output_logits, dim=-1)
                    - self.simplex_value
                )
                alpha_prod_t = self.scheduler.alphas_cumprod[t[0, 0].item()]
                sqrt_alpha_prod_t = torch.sqrt(alpha_prod_t)
                sqrt_one_minus_alpha_prod_t = torch.sqrt(1 - alpha_prod_t)
                noise = (
                    x_t - sqrt_alpha_prod_t * x_0_hat
                ) / sqrt_one_minus_alpha_prod_t
                simplex = self.scheduler.step(
                    x_0_hat,
                    t,
                    prev_t,
                    noise,
                    generator=generator,
                ).prev_sample

            # keep loss for logging
            losses.append(model_output.loss.detach().cpu())

            # yield over it. (prolly not optimal, but whatever)
            yield SimplexDiffusionPipelineOutput(
                simplex=old_simplex, logits=model_output_logits, loss=losses[-1]
            )
        # we take the mean loss over all timesteps
        loss = torch.stack(losses, dim=0)
        # from matplotlib import pyplot as plt
        # all_rewardst = torch.cat(all_rewards, dim=-1)
        # plt.plot(all_rewardst.to(torch.float32).T)
        # plt.savefig("tmp.png")
        # import pdb; pdb.set_trace()
        return SimplexDiffusionPipelineOutput(
            simplex=simplex, logits=model_output_logits, loss=loss
        )


# A variant of the SimplexDDPMPipeline that is used for evaluation.
# Main difference is that we assume that you pass the ground truth, and
# want to compute the loss.
class SimplexDDPMPipelineForEvaluation(SimplexDDPMPipeline):
    @torch.inference_mode()
    def __call__(
        self,
        seq_length: int = 512,
        generator: Optional[torch.Generator] = None,
        batch: Optional[torch.FloatTensor] = None,
        guidance_scale: float = 1.0,
        is_generator: bool = False,
    ) -> Union[SimplexDiffusionPipelineOutput, Tuple]:
        # Classifier_free guidance works only in the conditional generation case.
        classifier_free_guidance = (
            guidance_scale > 1.0 and self.is_conditional_generation
        )
        # Sample gaussian noise to begin loop
        vocab_size = self.model.config.vocab_size
        if batch is not None:
            seq_length = batch["input_ids"].shape[1]
        # idk why i have the bsz argument.
        batch_size = batch["input_ids"].shape[0]
        simplex_shape = (batch_size, seq_length, vocab_size)
        # simplex here is the simplex of the actual input!
        simplex = convert_to_simplex(
            batch["input_ids"], self.simplex_value, self.model.config.vocab_size
        )
        noise = self.simplex_value * torch.randn(
            simplex_shape, generator=generator, device=self.device
        )
        if self.model.config.self_condition is not None:
            previous_pred = torch.zeros(
                (batch_size, seq_length, vocab_size), device=self.device
            )
        logits_projection_fct = lambda x: logits_projection(  # noqa: E731
            x, self.sampling_type, self.top_p, self.simplex_value, self.temperature
        )
        losses = []
        previous_hidden = None

        warped_steps = []
        prev_t = 0
        for t in self.progress_bar(self.scheduler.timesteps):
            original_t = torch.tensor([t], device=self.device).expand(
                batch_size, seq_length
            )
            if is_cdcd_check(self.model):
                # warp timesteps based on cdf
                # we are in inference mode, anything in span_mask is to gen.
                token_inputs = torch.where(
                    batch["span_mask"], self.tokenizer.pad_token_id, batch["input_ids"]
                )
                t = self.model.warp_timesteps(
                    original_t,
                    t_min=0,
                    t_max=len(self.scheduler) - 1,
                    token_input=token_inputs,
                    span_mask=batch["span_mask"],
                )
            else:
                t = original_t
            t_scaled = scale(t, len(self.scheduler))
            warped_steps.append(t)
            noisy_simplex = self.scheduler.add_noise(simplex, noise, t)

            attention_mask = batch["input_ids"] != self.tokenizer.pad_token_id

            # TODO: do we care about self-conditioning...?
            model_output = self.model(
                input_ids=batch["input_ids"],
                span_mask=batch["span_mask"],
                attention_mask=attention_mask,
                simplex=noisy_simplex,
                timesteps=t_scaled,
                classifier_free_guidance=classifier_free_guidance,
                reduce_loss="none",
                previous_pred=previous_pred,
                max_timestep=len(self.scheduler),
                previous_hidden=previous_hidden,
            )
            model_output_logits = model_output.logits
            previous_hidden = model_output.hidden_states
            losses.append(model_output.loss.detach().cpu())

            if self.model.config.self_condition is not None:
                prev_output_logits = model_output_logits
                previous_pred = self_condition_preds(
                    self.model.config.self_condition,
                    prev_output_logits,
                    logits_projection_fct,
                )

            old_simplex = simplex
            # no output stuff here, since all we care about is the loss.
            # yield over it. (prolly not optimal, but whatever)
            yield SimplexDiffusionPipelineOutput(
                simplex=noisy_simplex, logits=model_output_logits, loss=losses[-1]
            )
        # we take the mean loss over all timesteps
        loss = torch.stack(losses, dim=0)
        return SimplexDiffusionPipelineOutput(
            simplex=simplex, logits=model_output_logits, loss=loss
        )