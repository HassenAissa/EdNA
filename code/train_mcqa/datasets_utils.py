from datasets import load_dataset
import json
import random
import re


def sft_dataset(tokenizer, max_length=200, number_of_processes = 4):
    """
    Load the SFT dataset from Hugging Face.
    """
    # Load the dataset
    dataset = load_dataset("HAissa/MNLP_M3_mcqa_dataset")["train"]
    dataset = dataset.filter(
        lambda x: len(tokenizer.encode(x["question"])) + len(tokenizer.encode(x["answer"])) < max_length, num_proc = number_of_processes
    )
    dataset = dataset.rename_column("question", "prompt")
    dataset = dataset.rename_column("answer", "completion")
    return dataset

def prepare_RL_mcqs_dataset(tokenizer, max_length=200, number_of_processes=4):
    """
    Load the RL MCQs dataset from Hugging Face.
    """
    dataset = load_dataset("HAissa/MCQ_dataset")["train"]
    dataset = dataset.map(
        lambda x: {
            "prompt": x["question"],
            "final_answer": x["answer"],
        },
        remove_columns=dataset.column_names,
        num_proc=number_of_processes,
    )

    dataset = dataset.filter(
        lambda x: len(tokenizer.encode(x["prompt"])) + len(tokenizer.encode(x["final_answer"])) < max_length, num_proc = number_of_processes
    )
    dataset = dataset.shuffle(seed=42)

    return dataset


def format_reward_function(completions, **kwargs):
    """
    Format the reward function for the RL dataset.
    """
    pattern = re.compile(r"[ABCDE]\.\s*.+")
    matches = [re.search(pattern, completion) for completion in completions]
    return [0.5 if match else 0 for match in matches]

def correctness_reward_function(completions, final_answer, **kwargs):    
    """
    Format the correctness reward function for the RL dataset.
    """
    rewards = []
    for completion, correct_answer in zip(completions, final_answer):
        completion = completion.split("\n")[0]
        correct_answer = correct_answer.split("\n")[0]
        if completion.strip() == correct_answer.strip():
            rewards.append(1.0)
        else:
            rewards.append(-1.0)
    return rewards