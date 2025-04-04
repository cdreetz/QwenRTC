import logging
import time
import asyncio
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)

class LlamaService:
    def __init__(self, model_id="meta-llama/Llama-3-8b-chat-hf", device=None):
        self.model_id = model_id
        
        # Determine the device
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.torch_dtype = torch.float16 if "cuda" in self.device else torch.float32
        
        logging.info(f"Using device for LLaMA: {self.device}, dtype: {self.torch_dtype}")
        self.llm_pipeline = None
        
        # Initialize the model
        self.initialize_model()
        
    def initialize_model(self):
        """Initialize the LLaMA model and tokenizer"""
        try:
            logging.info(f"Loading LLaMA model: {self.model_id}")
            start_time = time.time()
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                device_map=self.device
            )
            
            # Create pipeline
            self.llm_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=self.torch_dtype,
                device=self.device,
            )
            
            logging.info(f"LLaMA model loaded in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logging.error(f"Error initializing LLaMA model: {e}")
            raise
    
    async def generate_response(self, prompt, max_length=512, temperature=0.7):
        """
        Generate a response using the LLaMA model
        
        Args:
            prompt: The user's input text
            max_length: Maximum length of the response
            temperature: Controls randomness (higher = more random)
        
        Returns:
            Dict with generated response
        """
        try:
            if self.llm_pipeline is None:
                self.initialize_model()
                
            start_time = time.time()
            
            # Format prompt according to LLaMA chat template
            formatted_prompt = f"<|system|>\nYou are a helpful, harmless, and precise assistant.\n<|user|>\n{prompt}\n<|assistant|>\n"
            
            # Run inference
            result = await asyncio.to_thread(
                self.llm_pipeline,
                formatted_prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=0.95,
                do_sample=True,
                num_return_sequences=1
            )
            
            process_time = time.time() - start_time
            logging.info(f"LLaMA inference completed in {process_time:.2f} seconds")
            
            # Extract the response and remove the prompt
            generated_text = result[0]['generated_text']
            
            # Extract just the assistant's response from the generated text
            assistant_response = generated_text.split("<|assistant|>")[-1].strip()
            
            return {
                "response": assistant_response,
                "process_time": process_time
            }
            
        except Exception as e:
            logging.error(f"Error in LLaMA inference: {str(e)}")
            return {"error": str(e)}
