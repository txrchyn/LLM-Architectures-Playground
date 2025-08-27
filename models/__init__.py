from .GPT_2_modern.ModernGPT import ModernGPT
from .GPT_2_vanilla.VanillaGPT import VanillaGPT

from .GPT_2_modern.gpt_2_modern_config import gpt_2_modern_config
from .GPT_2_vanilla.gpt_2_vanilla_config import gpt_2_vanilla_config

MODEL_REGISTRY = {
    'modern': ModernGPT,
    'vanilla': VanillaGPT,
}

CONFIG_REGISTRY = {
    'modern': gpt_2_modern_config,
    'vanilla': gpt_2_vanilla_config,
}

def create_model(model_name: str, config: dict):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found in registry. Available: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[model_name]
    model = model_class(config)
    return model

def create_config(model_name, config_dict):
    """
    Creates a config object from a dictionary based on the model name.
    """
    if model_name not in CONFIG_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name}")
    
    config_class = CONFIG_REGISTRY[model_name]
    return config_class(**config_dict)