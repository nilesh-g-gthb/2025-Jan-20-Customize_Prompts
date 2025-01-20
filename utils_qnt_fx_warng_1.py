import sys
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from transformers import BitsAndBytesConfig

# Centralized model configuration
MODEL_ID = "microsoft/Phi-3.5-mini-instruct" # Change your LLM Model here
DEVICE = "auto"

# Quantization configuration
QUANTIZATION_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

class LLMHandler:
    def __init__(self):
        self.pipe = None

    def initialize_llm(self) -> None:
        """Initialize the LLM model with quantization."""
        try:
            print(f"Loading model {MODEL_ID} with 4-bit quantization...")
            
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                quantization_config=QUANTIZATION_CONFIG,
                trust_remote_code=True,
                device_map=DEVICE
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_ID,
                padding_side="left"
            )
            
            # Set pad token to eos token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            self.pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )
            
            print("Quantized model initialized successfully!")
            print("Using 4-bit quantization")
            
        except Exception as e:
            print(f"Error initializing LLM: {str(e)}")
            sys.exit(1)

    def get_llm_response(self, prompt: str) -> Optional[str]:
        """Get response from LLM with error handling."""
        try:
            if self.pipe is None:
                self.initialize_llm()

            # Updated pipeline parameters to match the correct API
            response = self.pipe(
                prompt,
                max_new_tokens=100,  # Changed from max_length to max_new_tokens
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.pipe.tokenizer.pad_token_id,
                return_full_text=True
            )
            
            response_text = response[0]['generated_text']
            
            if response_text.startswith(prompt):
                response_text = response_text[len(prompt):].strip()
            
            if 'Output:' in response_text:
                response_text = response_text.split('Output:')[-1].strip()
            
            valid_types = ['QuoteRequest', 'BondRequest', 'GENERAL']
            for type_ in valid_types:
                if type_ in response_text:
                    return type_
            
            return "GENERAL"
            
        except Exception as e:
            print(f"Error getting LLM response: {str(e)}")
            return "GENERAL"