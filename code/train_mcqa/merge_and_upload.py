import os
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import dotenv
# Load environment variables from .env file

huggingface_username = dotenv.dotenv_values(".env").get("HF_USERNAME")
access_token = dotenv.dotenv_values(".env").get("HF_ACCESS_TOKEN")

model_name = "./qwen3-0.6b-base-sft-final"
tokenizer = AutoTokenizer.from_pretrained(model_name)   
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", torch_dtype="bfloat16")
# Load the LoRA checkpoint
lora_model = PeftModel.from_pretrained(model, 
        "./qwen3-0.6b-base-grpo-final",
        torch_dtype=torch.bfloat16)

merged_model = lora_model.merge_and_unload()
merged_model = merged_model.to(torch.bfloat16)
print("LoRA weights successfully merged with base model")
model.push_to_hub(huggingface_username + "/MNLP_M3_mcqa_model", token=access_token)
tokenizer.push_to_hub(huggingface_username + "/MNLP_M3_mcqa_model", token=access_token)