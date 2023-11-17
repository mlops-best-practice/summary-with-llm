import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from typing import List, Dict

DEVICE = "cuda"
TRANSFORMERS_CACHE="/datadrive04/temp_huggingface/models"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.1"
CHECKPOINT_PATH = "/home/ai4code/summary-llm/mistral-summary-finetune/checkpoint-50"


class GPTCustomChat:
    def __init__(
        self, 
        max_new_tokens = 1024, 
        model_max_length = 512,
        checkpoint_path: str = None):
        self.max_new_tokens = max_new_tokens 
        self.model_max_length = model_max_length
        self.checkpoint_path = checkpoint_path
        
    
    def load_tokenizer(self):
        self. tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            model_max_length=self.model_max_length,
            padding_side="left",
            add_eos_token=True, 
            cache_dir =TRANSFORMERS_CACHE)
        self.tokenizer.pad_token = self.tokenizer.eos_token



    def load_model(self):
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        _base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, # Mistral, same as before
            quantization_config=bnb_config, # Same quantization config as before
            device_map="auto",
            trust_remote_code=True,
            cache_dir =TRANSFORMERS_CACHE,
            use_auth_token=True, 
            local_files_only=True

        )
        self.model = PeftModel.from_pretrained(_base_model, self.checkpoint_path)
        self.model.eval()
    
    def encode(self, conversation: List[Dict[str, str]]):
        return self.tokenizer.apply_chat_template(
            conversation, 
            return_tensors="pt").to(DEVICE)

    def decode(self, output):
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
        
    
    def generate(self, conversation: List[Dict[str, str]]):
        input_ids = self.encode(conversation)
        with torch.no_grad():
           _generated = self.model.generate(
                input_ids = input_ids, 
                max_new_tokens=self.max_new_tokens,  
                do_sample=True)
           output = self.decode(_generated)
        return output