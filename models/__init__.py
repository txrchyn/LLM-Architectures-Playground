from .GPT_2_modern.ModernGPT import ModernGPT
from .GPT_2_vanilla.VanillaGPT import VanillaGPT

MODEL_REGISTRY = {
    'modern': ModernGPT,
    'vanilla': VanillaGPT,
}

def create_model(model_name: str, config: dict):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found in registry. Available: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[model_name]
    model = model_class(config)
    return model