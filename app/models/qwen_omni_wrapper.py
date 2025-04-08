import logging
import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
from transformers import Qwen2_5OmniProcessor, Qwen2TokenizerFast
from transformers import Qwen2_5OmniModel
from qwen_omni_utils import process_mm_info

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
            use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

            if use_bf16:
                torch_dtype = torch.bfloat16
                logger.info("Using bfloat16 precission for model loading")
            else:
                torch_dtype = torch.float16
                logger.info("Using float16 precission for model loading")

            self.model_dtype = torch_dtype

            # Initialize  processor
            self.processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
            
            # Load model with audio output if enabled
            self.model = Qwen2_5OmniModel.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto" if device == "cuda" else None,
                attn_implementation="flash_attention_2",
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
        system_prompt: str = "You are Qwen, a helpful AI assistant, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
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

        conversation = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": input_text}],
            },
        ]

        if images and len(images) > 0:
            for image_path in images:
                conversation[1]["content"].append({"type": "image", "image": image_path})

        if videos and len(videos) > 0:
            for video_path in videos:
                conversation[1]["content"].append({"type": "video", "video": video_path})

        if audios and len(audios) > 0:
            for audio_path in audios:
                conversation[1]["content"].append({"type": "audio", "audio": audio_path})

        USE_AUDIO_IN_VIDEO = True

        print("conversation before processor:", conversation)

        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        
        print("text after processor:", text)

        try:
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)

            #if audios and len(audios) > 0:
            #    processed_audios = []
            #    for audio in audios:
            #        if isinstance(audio, np.ndarray):
            #            audio_tensor = torch.tensor(audio, dtype=torch.float32)
            #            if hasattr(self, 'model_dtype'):
            #                audio_tensor = audio_tensor.to(dtype=self.model_dtype)

            #            audio_tensor = audio_tensor.to(self.device)
            #            processed_audios.append(audio_tensor)
            #        elif isinstance(audio, torch.Tensor):
            #            if hasattr(self, 'model_dtype') and audio.dtype != self.model_dtype:
            #                audio = audio.to(dtype=self.model_dtype)

            #            audio = audio.to(self.device)
            #            processed_audios.append(audio)

            #    audios = processed_audios

            #else:
            #    audios = []

            #if not audios or len(audios) == 0:
            #    logger.warning("No valid audio data processed, running in text only mode")
            #else:
            #    logger.info(f"Successfully processed {len(audios)} audio files into tensors")

        except Exception as e:
            logger.error(f"Error processing multimodal inputs: {str(e)}", exc_info=True)
            # fallback
            audios, images, videos = [], [], []
        
        # Process inputs with the processor
        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            audios=audios,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=USE_AUDIO_IN_VIDEO
        ).to(self.device).to(self.model_dtype)
        
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
                    use_audio_in_video=USE_AUDIO_IN_VIDEO,  # Default to False for simplicity
                    return_audio=True
                )
                decoded_text = self.processor.batch_decode(generated_text, skip_special_tokens=True)
                text_response = decoded_text[0] if isinstance(decoded_text, list) else decoded_text

                
                # Convert tensors to Python-native types
                response = {
                    "text": text_response.split("\n")[-1],
                    "has_audio": True,
                    "audio_sample_rate": 24000,  # Default sample rate for Qwen Omni
                    "audio_waveform": audio_waveform.detach().cpu().numpy().tolist()
                }
            else:
                generated_text = self.model.generate(
                    **inputs,
                    thinker_max_new_tokens=max_new_tokens,
                    return_audio=False,
                    use_audio_in_video=USE_AUDIO_IN_VIDEO
                )

                decoded_text = self.processor.batch_decode(generated_text, skip_special_tokens=True)
                print("decoded text: ", decoded_text)

                if isinstance(decoded_text, list):
                    text_response = decoded_text[0] if decoded_text else ""
                else:
                    text_response = decoded_text

                print("text response: ",text_response)

                final_text = text_response.split("\n")[-1] if isinstance(text_response, str) and "\n" in text_response else text_response
                
                response = {
                    "text": final_text,
                    "has_audio": False
                }
                
            return response
            
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            raise RuntimeError(f"Inference failed: {str(e)}")
