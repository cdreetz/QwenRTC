
import os
import io
import torch
import logging
import requests
from PIL import Image
from typing import Optional, List
from urllib.request import urlopen

import soundfile as sf
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Phi4MultimodalWrapper:
    def __init__(
        self,
        model_name: str = "microsoft/Phi-4-multimodal-instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the Phi-4 multimodal model

        Args:
            model_name: HF model id
            device: Device to laod the model on 'cpu' or 'cuda'
        """
        self.device = device
        self.model_name = model_name
        self.is_ready = False

        logger.info(f"Loading model {model_name} on {device}...")
        try:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            attn_implementation = 'flash_attention_2' if (
                torch.cuda.is_available() and
                torch.cuda.get_device_capability()[0] >= 8 
            ) else 'eager'
            logger.info(f"Using attention impl: {attn_implementation}")

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                device_map="cuda", 
                torch_dtype="auto", 
                trust_remote_code=True,
                _attn_implementation=attn_implementation,
            ).to(self.device)

            self.generation_config = GenerationConfig.from_pretrained(model_name)
            logger.info(f"Model loaded sucessfully.")
            self.is_ready = True

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}")

    def load_audio(self, audio_path: str):
        """Load audio file from path or URL"""
        try:
            if audio_path.startswith(('http://', 'https://')):
                # Handle URL
                response = requests.get(audio_path)
                audio_bytes = io.BytesIO(response.content)
                audio, samplerate = sf.read(audio_bytes)
            else:
                # Handle local file
                audio, samplerate = sf.read(audio_path)
            
            return audio, samplerate
        except Exception as e:
            logger.error(f"Failed to load audio: {str(e)}")
            raise RuntimeError(f"Audio loading failed: {str(e)}")

    def get_available_speakers(self) -> List[str]:
        """Phi 4 doesnt support audio output"""
        return []



    async def generate_response(
        self,
        input_text: str,
                images: Optional[List[str]] = None,
        videos: Optional[List[str]] = None,
        audios: Optional[List[str]] = None,
        speaker: str = None,  # Not used for Phi-4 but kept for API compatibility
        return_audio: bool = False,  # Not used for Phi-4 but kept for API compatibility
        max_new_tokens: int = 1024,
        system_prompt: str = "You are Phi-4, a helpful AI assistant that can understand images and audio."
    ):
        """
        Generate a response from the model for a given input.
        
        Args:
            input_text: Input text prompt
            images: Optional list of image paths or URLs
            videos: Optional list of video paths or URLs (not supported by Phi-4)
            audios: Optional list of audio paths or URLs
            speaker: Speaker identifier (not used by Phi-4)
            return_audio: Whether to return audio (not supported by Phi-4)
            max_new_tokens: Maximum number of tokens to generate
            system_prompt: System prompt for the model
            
        Returns:
            Dictionary containing response text
        """
        if not self.is_ready:
            raise RuntimeError("Model is not properly initialized")

        user_prompt = '<|user|>'
        assistant_prompt = '<|assistant|>'
        prompt_suffix = '<|end|>'
        prompt = f'{user_prompt}{input_text}{prompt_suffix}{assistant_prompt}'

        input_dict = {"text": prompt, "return_tensors": "pt"}

        audio_objects = []
        if audios and len(audios) > 0:
            for i, audio_path in enumerate(audios):
                try:
                    audio, samplerate = self.load_audio(audio_path)
                    audio_objects.append((audio, samplerate))
                    prompt = f'{user_prompt}<|audio_{i+1}|>{input_text}{prompt_suffix}{assistant_prompt}'
                except Exception as e:
                    logger.error(f"Failed to load audio {audio_path}: {str(e)}")

            input_dict["text"] = prompt
            input_dict["audios"] = audio_objects

        if videos:
            logger.warning("Phi-4 does not support videos. Videos will be ignored.")



        input = self.processor(**input_dict).to(self.device)

        generate_ids = self.model.generate(
            **input,
            max_new_tokens=max_new_tokens,
            generation_config=self.generation_config,
        )
        generate_ids = generate_ids[:, input['input_ids'].shape[1]:]

        response = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return {
            "text": response[0]
        }

    def __call__(self, input_text: str, audio_path: Optional[str] = None, **kwargs):
        import asyncio
        return asyncio.run(self.generate_response(input_text, audio_path, **kwargs))



