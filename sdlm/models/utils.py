from .longformer.configuration_longformer import LongformerDiffusionConfig
from .longformer.modeling_longformer import LongformerForDiffusionLM
from .roberta.configuration_roberta import RobertaDiffusionConfig
from .roberta.modeling_roberta import RobertaForDiffusionLM
from .xlm_roberta.configuration_xlm_roberta import XLMRobertaDiffusionConfig
from .xlm_roberta.modeling_xlm_roberta import XLMRobertaForDiffusionLM


def model_config_helper(model_name_or_path):
    if "longformer" in model_name_or_path:
        return LongformerDiffusionConfig, LongformerForDiffusionLM
    elif "xlm" in model_name_or_path:
        return XLMRobertaDiffusionConfig, XLMRobertaForDiffusionLM
    return RobertaDiffusionConfig, RobertaForDiffusionLM
