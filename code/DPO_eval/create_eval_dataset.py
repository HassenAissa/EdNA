import os
from datasets import load_dataset, DatasetDict, concatenate_datasets
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import login


def extract_chosen_rejected(example):
    if example["overall_preference"] < 0:
        chosen   = example["response1"]
        rejected = example["response2"]
    else:
        chosen   = example["response2"]
        rejected = example["response1"]

    prompt_str = "\n".join(
        f"{turn['role']}: {turn['content']}"
        for turn in example["context"]
    )
    
    return {
        "prompt":  prompt_str,
        "chosen":   chosen,
        "rejected": rejected,
        "source":   "HelpSteer3"
    }


def process_reward_bench(example):
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"],
        "source": "reward-bench"
    }


def main():
    load_dotenv(find_dotenv())
    login(token=os.getenv("HF_ACCESS_TOKEN"))
    hf_user = os.getenv("HF_USERNAME")
    
    # Steer ds dataset
    steer_ds = load_dataset("nvidia/HelpSteer3")
    val_steer_ds = steer_ds["validation"]
    val_steer_stem = val_steer_ds.filter(lambda ex: ex["domain"] in ["code", "stem"])
    hs_chosen_rejected = val_steer_stem.map(
        extract_chosen_rejected,
        remove_columns=val_steer_stem.column_names
    )
    dataset_dict = DatasetDict({"test": hs_chosen_rejected})
    dataset_dict.push_to_hub(
        repo_id=f"{hf_user}/HelpSteer_stem_eval",
        private=False
    )

    # RewardBench dataset
    reward_bench_ds = load_dataset("allenai/reward-bench", split="filtered")
    rb_chosen_rejected = reward_bench_ds.map(process_reward_bench, remove_columns=reward_bench_ds.column_names)
    rb_data_dict = DatasetDict({"test": rb_chosen_rejected})
    rb_data_dict.push_to_hub(
        repo_id=f"{hf_user}/reward_bench_eval",
        private=False
    )

    # Complete evaluation dataset
    merged = concatenate_datasets([hs_chosen_rejected, rb_chosen_rejected])
    merged_data_dict = DatasetDict({"test": merged})
    merged_data_dict.save_to_disk("DPO_eval")
    merged_data_dict.push_to_hub(
        repo_id=f"{hf_user}/DPO_eval",
        private=False
    )
    

if __name__ == "__main__":
    main()