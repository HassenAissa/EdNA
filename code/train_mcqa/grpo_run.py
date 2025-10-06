import datasets_utils
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import dotenv
import wandb
# Load environment variables from .env file
wandb_api_key = dotenv.dotenv_values(".env").get("WANDB_API_KEY")
wandb.login(key=wandb_api_key)

model_name = "qwen3-0.6b-base-sft-final"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype="bfloat16",
)
model.gradient_checkpointing_enable()

rl_dataset = datasets_utils.prepare_RL_mcqs_dataset(
    tokenizer,
    max_length=128,
)



GRPO_config = GRPOConfig(
    output_dir="qwen3-0.6b-base-grpo",
    per_device_train_batch_size=200,
    gradient_accumulation_steps=3,
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_steps=50,      
    save_steps=100,
    warmup_steps=50,
    weight_decay=0.01,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    seed=3407,
    bf16=True,                   
    fp16 = False,
    log_level="info",            
    report_to="wandb",           
    num_generations=6,
    max_grad_norm=0.5,
    max_prompt_length=128,
    max_completion_length=35,
    gradient_checkpointing=True,

)

rank = 16
peft_config = LoraConfig(
    use_dora=True,
    r=rank,
    lora_alpha=rank*2,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
    bias='none',
    lora_dropout=0.05,
)

model.config.pad_token_id = tokenizer.pad_token_id

model = get_peft_model(model, peft_config)

grpo_trainer = GRPOTrainer(
    model=model,
    args=GRPO_config,
    train_dataset=rl_dataset,
    reward_funcs=[datasets_utils.correctness_reward_function, datasets_utils.format_reward_function],

)
grpo_trainer.train()
grpo_trainer.save_model("qwen3-0.6b-base-grpo-final")