import logging
import torch
from typing import Dict, Any, Optional, Tuple, List, Union
from transformers import Qwen2_5OmniProcessor, Qwen2TokenizerFast
from transformers import Qwen2_5OmniModel

logger = logging.getLogger(__name__)

class QwenOmniWrapper:
    """
    Wrapper for Qwen 2.5 Omni model providing simplified interface
    for inference and resource management.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Omni-7B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        enable_audio_output: bool = True,
    ):
        """
        Initialize the Qwen 2.5 Omni model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to load the model on ('cuda', 'cpu')
            enable_audio_output: Whether to enable audio generation
        """
        self.device = device
        self.model_name = model_name
        self.enable_audio_output = enable_audio_output
        self.is_ready = False
        
        logger.info(f"Loading model {model_name} on {device}...")
        try:
            # Initialize tokenizer and processor
            self.processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
            self.tokenizer = Qwen2TokenizerFast.from_pretrained(model_name) 
            
            # Load model with audio output if enabled
            self.model = Qwen2_5OmniModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
            )
            
            if device == "cpu":
                self.model.float()  # Use float32 on CPU
            
            # If audio output is enabled, ensure talker is loaded
            if enable_audio_output and not self.model.has_talker:
                logger.info("Enabling talker module...")
                self.model.enable_talker()
                
            logger.info(f"Model loaded successfully. Available speakers: {list(self.model.speaker_map.keys())}")
            self.is_ready = True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}")
    
    def get_available_speakers(self) -> List[str]:
        """Return list of available speakers for audio output"""
        return list(self.model.speaker_map.keys())
    
    async def generate_response(
        self,
        input_text: str,
        images: Optional[List[str]] = None,
        videos: Optional[List[str]] = None,
        audios: Optional[List[str]] = None,
        speaker: str = None,
        return_audio: bool = None,
        max_new_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """
        Generate a response from the model for a given input.
        
        Args:
            input_text: Input text prompt
            images: Optional list of image paths or URLs
            videos: Optional list of video paths or URLs
            audios: Optional list of audio paths or URLs
            speaker: Speaker identifier for audio output
            return_audio: Whether to return audio output
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Dictionary containing response text and optionally audio data
        """
        if not self.is_ready:
            raise RuntimeError("Model is not ready for inference")
        
        # Process inputs with the processor
        inputs = self.processor(
            text=input_text,
            images=images,
            videos=videos,
            audios=audios,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Set return_audio based on parameters or defaults
        if return_audio is None:
            return_audio = self.enable_audio_output
            
        # Verify speaker if audio is requested
        if return_audio and speaker not in self.model.speaker_map:
            available_speakers = self.get_available_speakers()
            logger.warning(f"Requested speaker '{speaker}' not available. Using default speaker.")
            speaker = available_speakers[0] if available_speakers else None
            
        # Generate response
        try:
            if return_audio and speaker:
                generated_text, audio_waveform = self.model.generate(
                    **inputs,
                    spk=speaker,
                    thinker_max_new_tokens=max_new_tokens,
                    use_audio_in_video=False,  # Default to False for simplicity
                    return_audio=True
                )
                
                # Convert tensors to Python-native types
                response = {
                    "text": self.tokenizer.decode(generated_text[0], skip_special_tokens=True),
                    "has_audio": True,
                    "audio_sample_rate": 24000,  # Default sample rate for Qwen Omni
                    "audio_waveform": audio_waveform.detach().cpu().numpy().tolist()
                }
            else:
                generated_text = self.model.generate(
                    **inputs,
                    thinker_max_new_tokens=max_new_tokens,
                    return_audio=False
                )
                
                response = {
                    "text": self.tokenizer.decode(generated_text[0], skip_special_tokens=True),
                    "has_audio": False
                }
                
            return response
            
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            raise RuntimeError(f"Inference failed: {str(e)}")
