from .h3.configuration_h3 import H3DiffusionConfig
from .h3.modeling_h3 import H3ForDiffusionLM
from .longformer.configuration_longformer import LongformerDiffusionConfig
from .longformer.modeling_longformer import LongformerForDiffusionLM
from .roberta.configuration_roberta import RobertaDiffusionConfig
from .roberta.modeling_roberta import RobertaForDiffusionLM
from .xlm_roberta.configuration_xlm_roberta import XLMRobertaDiffusionConfig
from .xlm_roberta.modeling_xlm_roberta import XLMRobertaForDiffusionLM


def model_config_helper(model_name_or_path):
    if "longformer" in model_name_or_path:
        return LongformerDiffusionConfig, LongformerForDiffusionLM
    elif "h3" in model_name_or_path:
        return H3DiffusionConfig, H3ForDiffusionLM
    elif "xlm" in model_name_or_path:
        return XLMRobertaDiffusionConfig, XLMRobertaForDiffusionLM
    return RobertaDiffusionConfig, RobertaForDiffusionLM
