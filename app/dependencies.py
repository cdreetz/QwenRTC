from typing import Optional, Union
from app.models.qwen_omni_wrapper import QwenOmniWrapper
from app.models.phi4_multimodal_wrapper import Phi4MultimodalWrapper
import os
import logging

logger = logging.getLogger(__name__)

# Get default model type from environment
DEFAULT_MODEL_TYPE = os.getenv("MODEL_TYPE", "qwen")  # Default to Qwen

# Global variable to hold a single model instance
model_instance: Optional[Union[QwenOmniWrapper, Phi4MultimodalWrapper]] = None
current_model_type: Optional[str] = None

def get_model_instance():
    """Dependency function to get the initialized model instance"""
    global model_instance, current_model_type
    
    if model_instance is None or not model_instance.is_ready:
        raise RuntimeError("Model is not initialized or ready")
    
    return model_instance

def initialize_model(model_type: str = DEFAULT_MODEL_TYPE):
    """Initialize the model with the specified type"""
    global model_instance, current_model_type
    
    # If we already have this model type loaded and ready, do nothing
    if model_instance is not None and model_instance.is_ready and current_model_type == model_type:
        return
    
    # Unload existing model if any
    if model_instance is not None:
        logger.info(f"Unloading current model of type {current_model_type}")
        model_instance = None
        current_model_type = None
    
    # Initialize the requested model
    try:
        if model_type == "qwen":
            model_name = os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen2.5-Omni-7B")
            logger.info(f"Initializing Qwen model: {model_name}")
            model_instance = QwenOmniWrapper(model_name=model_name)
        elif model_type == "phi":
            model_name = os.getenv("PHI_MODEL_NAME", "microsoft/Phi-4-multimodal-instruct")
            logger.info(f"Initializing Phi model: {model_name}")
            model_instance = Phi4MultimodalWrapper(model_name=model_name)
        else:
            logger.error(f"Unsupported model type: {model_type}")
            raise ValueError(f"Unsupported model type: {model_type}")
        
        current_model_type = model_type
        logger.info(f"Model {model_type} initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model of type {model_type}: {str(e)}")
        raise RuntimeError(f"Model initialization failed: {str(e)}")
