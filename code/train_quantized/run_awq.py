from datasets import load_dataset
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import torch
import os
import dotenv

huggingface_username = dotenv.dotenv_values(".env").get("HF_USERNAME")
access_token = dotenv.dotenv_values(".env").get("HF_ACCESS_TOKEN")

NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 256
LETTER_INDICES = ["A", "B", "C", "D"]

raw = load_dataset("cais/mmlu", "auxiliary_train")
ds = raw["train"]

ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))


def mmlu_prompt_with_answer(example):
    rec = example["train"] if "train" in example else example
    q = rec["question"]
    choices = rec["choices"]
    gold_ix = rec["answer"]
    topic = "knowledge and skills in advanced master-level STEM courses"
    prompt = (
        f"The following are multiple choice questions (with answers) "
        f"about {topic}.\n\n"
        f"{q}\n"
    )
    for letter, choice in zip(LETTER_INDICES, choices):
        prompt += f"{letter}. {choice}\n"
    prompt += "Answer:"
    gold_letter = LETTER_INDICES[gold_ix]
    return {"text": prompt + " " + gold_letter}

ds = ds.map(mmlu_prompt_with_answer, remove_columns=ds.column_names)

calib_texts = ds["text"] 

tokenizer = AutoTokenizer.from_pretrained("HAissa/MNLP_M3_mcqa_model")

model = AutoAWQForCausalLM.from_pretrained(
    "HAissa/MNLP_M3_mcqa_model",
    device_map="auto",
    trust_remote_code=True)

quant_config = {
    "w_bit": 4,
    "q_group_size": 64,
    "zero_point": True
}

model.quantize(
    tokenizer=tokenizer,
    quant_config=quant_config,
    calib_data=calib_texts, 
    max_calib_samples=NUM_CALIBRATION_SAMPLES 
)

model.save_quantized("awq_model_4bit")
tokenizer.save_pretrained("awq_model_4bit")


    