import datasets_utils
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
import torch
import dotenv
import wandb
# Load environment variables from .env file
wandb_api_key = dotenv.dotenv_values(".env").get("WANDB_API_KEY")
wandb.login(key=wandb_api_key)


model_name = "Qwen/Qwen3-0.6B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
).to("cuda")


model.gradient_checkpointing_enable()

sft_dataset = datasets_utils.sft_dataset(tokenizer, max_length=750, number_of_processes=4)

torch.cuda.empty_cache()
training_args = SFTConfig(
    output_dir="qwen3-0.6b-base-sft",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=4e-6,
    num_train_epochs=2,
    logging_steps=100,      
    save_steps=5000,
    warmup_steps=50,
    weight_decay=0.01,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    seed=3407,
    bf16=True,                   
    fp16=False,
    log_level="info",            
    report_to="wandb",           
    dataset_num_proc=4,
    max_length=770,
    packing=False,
    gradient_checkpointing = True
)
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=sft_dataset,
)

trainer.train()
trainer.save_model("qwen3-0.6b-base-sft-final")