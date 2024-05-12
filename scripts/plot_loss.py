import logging
import os
import sys

import torch
from matplotlib import pyplot as plt
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)

from sdlm.arguments import DiffusionArguments, ModelArguments
from sdlm.models import TokenwiseCDCDRobertaConfig, TokenwiseCDCDRobertaForDiffusionLM
from sdlm.pipelines.simplex_ddpm import SimplexDDPMPipeline
from sdlm.schedulers import TokenWiseSimplexDDPMScheduler

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def main():
    parser = HfArgumentParser((ModelArguments, DiffusionArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, diffusion_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, diffusion_args = parser.parse_args_into_dataclasses()

    # Set seed before initializing model.
    set_seed(42)
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    config = TokenwiseCDCDRobertaConfig.from_pretrained(
        model_args.model_name_or_path,
        self_condition=diffusion_args.self_condition,
        self_condition_zeros_after_softmax=diffusion_args.self_condition_zeros_after_softmax,
        deepmind_conditional=diffusion_args.deepmind_conditional,
        classifier_free_simplex_inputs=diffusion_args.classifier_free_simplex_inputs,
        classifier_free_uncond_input=diffusion_args.classifier_free_uncond_input,
        self_condition_mlp_projection=diffusion_args.self_condition_mlp_projection,
        self_condition_mix_before_weights=diffusion_args.self_condition_mix_before_weights,
        self_condition_mix_logits_before_weights=diffusion_args.self_condition_mix_logits_before_weights,
        empty_token_be_mask=diffusion_args.empty_token_be_mask,
        **config_kwargs,
    )
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, **tokenizer_kwargs
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        model = TokenwiseCDCDRobertaForDiffusionLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        raise RuntimeError("You need to load a pretrained model")

    # We resize the xs only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    vocab_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    # for some insane reason some of the model is not correctly loaded using from_pretrained...
    state_dict = torch.load(
        os.path.join(model_args.model_name_or_path, "pytorch_model.bin"),
        map_location="cpu",
    )
    # for some insane reason the word embeddings dont get loaded
    model.roberta.embeddings.word_embeddings.weight = torch.nn.Parameter(
        state_dict["roberta.embeddings.word_embeddings.weight"]
    )
    model.tie_weights()
    # make sure loading is entirely correct.
    assert (
        len(
            [k for k in state_dict if torch.any(state_dict[k] != model.state_dict()[k])]
        )
        == 0
    )

    def generate(
        inputs,
        simplex_value=5.0,
        top_p=0.99,
        temperature=1.0,
        diffusion_steps=2500,
        beta_schedule="squaredcos_improved_ddpm",
        clip_sample=False,
        guidance_scale=1.0,
        generated_sequence_length=256,
        use_model="cdcd",
    ):
        tokenized_input = tokenizer(
            [inputs], add_special_tokens=False, return_tensors="pt"
        ).input_ids
        tokenized_input_len = tokenized_input.shape[-1]
        span_mask = torch.cat(
            [
                torch.zeros((1, tokenized_input_len // 2)),
                torch.ones((1, tokenized_input_len - tokenized_input_len // 2)),
            ],
            axis=-1,
        ).bool()
        inputs = {"input_ids": tokenized_input.cuda(), "span_mask": span_mask.cuda()}

        model.eval()

        pipeline = SimplexDDPMPipeline(
            model=model.cuda(),
            scheduler=TokenWiseSimplexDDPMScheduler(
                num_train_timesteps=diffusion_steps,
                beta_schedule=beta_schedule,
                simplex_value=simplex_value,
                clip_sample=clip_sample,
                device=torch.device("cuda", 0),
            ),
            simplex_value=simplex_value,
            top_p=top_p,
            sampling_type="top_p",  # currently only this is supported
            is_conditional_generation=True,
            tokenizer=tokenizer,
            classifier_free_uncond_input="empty_token",
            temperature=temperature,
            guidance_softmax_combination=True,
        )
        # pipeline.progress_bar = progress.tqdm
        pipeline_args = {
            "batch_size": 1,
            "seq_length": generated_sequence_length,
            "batch": inputs,
            "guidance_scale": guidance_scale,
            "is_generator": True,
        }
        for output in pipeline(**pipeline_args):
            yield output.loss.item()

    generator = generate(
        inputs="bounded to KDE, whereas I think Gaim is not so quite involved with Gnome.\nSo we can reduce this redundancy problem to the problem of two desktops, if it is a problem. I'm not sure it is, after all competition is good and drives development. Is it that much of a problem that some coders have spent time and effort re-inventing the wheel anyway?\nWell I think it is a problem, unless you are heavily camped in either desktop. I'm not. When I develop a GUI application, I have to make a decision about which environment suits me best. I know there's more to it than that - Gnome has all that other stuff like Glib and KDE isn't just QT - but the GUI is a major consideration.\nSo if I opt for (say) GTK+, will some QT zealot prefer to code an alternative to using my version?\nI run a mixture of Gnome and KDE applications. I use Konqueror as a web browser, Gimp for some graphic stuff, etc. The two toolkits are somewhat annoying because they do not interact particularly well.\nWhen I write a GUI app, I want the interface code to be as separate from the logic as possible, to make swapping them in and out as easy as possible. Don't code an alternative, write a new interface for me!\nGlade is a GTK+ interface builder. You use a drag-and-drop program to compose a UI, and it can generate the corresponding GTK+ code. You can save the project seperately as an XML application.\nWhat is really exciting however is the companion libglade. This library reads in the XML that Glade writes at runtime and constructs the UI from that.\nYou therefore don't even need to necessarily use the UI builder to generate the XML if you don't want to. Perhaps you'll just translate a UI described in a different XML application.\nSo why do I think this is exciting?\nAlter/tweak the UI at runtime, without a recompilation.\nTranslate (via XSL) an interface description in a different XML application. I haven't looked at the design of the XML glade dialect much myself, it might be a nightmare; but at least that nightmare could be hidden behind a translation.\nWant a KDE version? Write a runtime, XML-interpreting UI builder for KDE!\nArgh! so this rant has gone",
        diffusion_steps=100,
    )
    losses = list(generator)
    # now, we need to warp timesteps
    from sdlm.models.cdcd.cdf import LossCDF

    module = LossCDF(100)
    weights = torch.load(
        model_args.model_name_or_path + "/pytorch_model.bin", map_location="cpu"
    )
    l_t = weights["cdf.l_t"]
    l_u = weights["cdf.l_u"]
    module.load_state_dict({"l_t": l_t, "l_u": l_u})

    t = torch.linspace(0, 1, 100)
    u = module(t=t, normalized=True)
    u = torch.clamp((u * 100).long(), 0, 99)
    plt.clf()

    plt.clf()
    plt.plot(losses)
    plt.savefig("losses.jpg")


if __name__ == "__main__":
    main()
