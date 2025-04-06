from typing import Optional
from app.models.qwen_omni_wrapper import QwenOmniWrapper

# Global variable to hold the model instance
model_instance: Optional[QwenOmniWrapper] = None


def get_model_instance():
    """Dependency function to get the initialized model instance"""
    if model_instance is None or not model_instance.is_ready:
        raise RuntimeError("Model is not initialized or ready")
    return model_instance
