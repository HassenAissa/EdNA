import os
import wandb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv, find_dotenv


def load_model(
        model_name="HAissa/last-sft-40000",
        load_in_8bit=False,
        bf16_dtype=True,
        max_length=256
        ):
    
    dtype = torch.bfloat16 if bf16_dtype else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto",
        torch_dtype=dtype,
        )
    
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    tokenizer.model_max_length = max_length
    # if no pad_token, set it to eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("loaded model and tokenizer")
    return model, tokenizer


def get_dataset(path="levinius/MNLP_M3_dpo_dataset"):
    dataset = load_dataset(path)
    train_ds = dataset["train"]
    val_ds = dataset["validation"]
    print("Train dataset: ",train_ds)
    return train_ds, val_ds


def get_training_args(hf_user):
    training_args = DPOConfig(
        output_dir="MNLP_M3_dpo_model_reproduction",
        num_train_epochs=2,
        report_to="wandb",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        eval_accumulation_steps=2,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        learning_rate=3e-6,
        lr_scheduler_type="linear",
        weight_decay=0.01,
        warmup_ratio=0.03,
        bf16=True,
        fp16=False, # Changed from bf16 to fp16
        max_grad_norm=2.0,
        logging_steps=200,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        load_best_model_at_end=True,
        metric_for_best_model="eval_rewards/accuracies",
        save_total_limit=3,
        optim="adamw_torch",
        seed=42,
        push_to_hub=True,
        hub_model_id=f"{hf_user}/MNLP_M3_dpo_model_reproduction",
        hub_strategy="checkpoint",
        beta=0.5
    )
    print("Created training args")
    return training_args


def main():
    load_dotenv(find_dotenv())
    login(token=os.getenv("HF_TOKEN"))
    hf_user = os.getenv("HF_USERNAME")
    wandb_user = os.getenv("WANDB_USERNAME")
    model, tokenizer = load_model()
    train_ds, val_ds = get_dataset()
    training_args = get_training_args(hf_user)

    wandb.login()
    
    wandb.init(
        project="MNLP",
        entity=wandb_user,
        name="MNLP_M3_dpo_model_reproduction",
        config=training_args
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer
    )
    print("Created trainer start training...")
    trainer.train()

if __name__ == "__main__":
    main()