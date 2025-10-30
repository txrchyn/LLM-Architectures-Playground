# registry.py

from .modern_gpt import ModernGPT, ModernGPTConfig
from .vanilla_gpt import VanillaGPT, VanillaGPTConfig
from .gqa_gpt import GQAGPT, GQAGPTConfig
from .StateSpace import StateSpaceLM, StateSpaceConfig
from .Linformer import LinformerLM, LinformerConfig

# === MoE imports ===
from .MoE.moe_model import MoEGPT
from .MoE.moe_config import MoEConfig

MODEL_REGISTRY = {
    'modern': ModernGPT,
    'vanilla': VanillaGPT,
    'gqa': GQAGPT,
    'linformer': LinformerLM,
    'ssm': StateSpaceLM,
    'moe': MoEGPT,          # <-- добавили
}

CONFIG_REGISTRY = {
    'modern': ModernGPTConfig,
    'vanilla': VanillaGPTConfig,
    'gqa': GQAGPTConfig,
    'linformer': LinformerConfig,
    'ssm': StateSpaceConfig,
    'moe': MoEConfig,       # <-- добавили
}

def create_model(model_name: str, config):
    """
    Создаёт модель по имени. Аргумент `config` должен быть ИМЕННО объектом конфига
    соответствующего класса (а не dict).
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found in registry. Available: {list(MODEL_REGISTRY.keys())}")

    model_class = MODEL_REGISTRY[model_name]
    return model_class(config)

def create_config(model_name: str, config_dict: dict):
    """
    Создаёт объект конфига из dict на основе имени модели.
    """
    if model_name not in CONFIG_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name}. Available: {list(CONFIG_REGISTRY.keys())}")

    config_class = CONFIG_REGISTRY[model_name]
    return config_class(**config_dict)
