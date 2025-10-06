import random
import os
from datasets import DatasetDict, load_from_disk, load_dataset, concatenate_datasets
from dotenv import load_dotenv, find_dotenv


def parse_three_labels(ranking_str):
    if not isinstance(ranking_str, str):
        return None
    idx = ranking_str.find("Ranking:")
    if idx == -1:
        return None
    tail = ranking_str[idx:].strip().rstrip(".")
    parts = tail.split(":", 1)
    if len(parts) != 2:
        return None
    after_colon = parts[1]
    labels = [lbl.strip() for lbl in after_colon.split(">")]
    if len(labels) != 3:
        return None
    for L in labels:
        if L not in {"A", "B", "C"}:
            return None
    return labels


def keep_if_valid(ex):
    labels = parse_three_labels(ex.get("ranking", ""))
    return labels is not None


def create_preference_dataset(example):
    question = example["prompt"]
    options  = example["options"]
    if options and options != [""]:
        prompt_text = question + "\nOptions:\n" + "\n".join(options)
    else:
        prompt_text = question

    raw_rank = example["ranking"] or ""
    idx = raw_rank.find("Ranking:")
    if idx != -1:
        raw_rank = raw_rank[idx:]
    raw_rank = raw_rank.strip().rstrip(".")
        
    parts = raw_rank.split(":", 1)[1].split(">")
    labels = [p.strip() for p in parts]

    label_to_answer = {
        "A": example["Qwen/Qwen3-8B"],
        "B": example["Mistral-7B-Instruct-v0.2"],
        "C": example["Meta-Llama-3-8B-Instruct"],
    }

    chosen_text = label_to_answer[labels[0]]

    if random.random() < 0.9:
        rejected_label = labels[1]
    else:
        rejected_label = labels[2]
    rejected_text = label_to_answer[rejected_label]

    return {
        "prompt":  prompt_text,
        "chosen":  chosen_text,
        "rejected": rejected_text,
        "source":  "distilled",
    }


def remap_and_add_source(example):
    return {
        "prompt":   example["input"],
        "chosen":   example["accepted"],
        "rejected": example["rejected"],
        "source":   "code-preference-pairs",
    }


def map_python_ds(ex):
    return {
        "prompt": ex["prompt"],
        "chosen": ex["chosen"],
        "rejected": ex["rejected"],
        "source": "python-dpo"
    }


def map_math_ds(ex):
    return {
        "prompt": ex["question"],
        "chosen": ex["chosen"],
        "rejected": ex["rejected"],
        "source": "math-dpo"
    }


def main():
    complete_ds = load_from_disk("distill_questions_with_judgment")
    filtered_ds = complete_ds.filter(keep_if_valid)
    preference_ds = filtered_ds.map(create_preference_dataset, remove_columns=complete_ds.column_names)

    preference_ds.save_to_disk("Distilled_Preference_Data")

    code_preferences = load_dataset("Vezora/Code-Preference-Pairs", split="train")
    code_subset = code_preferences.select(range(5000))
    code_pref = code_subset.map(
        remap_and_add_source,
        remove_columns=code_subset.column_names,
        batched=False,
    )

    python_ds = load_dataset("jondurbin/py-dpo-v0.1", split="train")
    python_dpo = python_ds.map(map_python_ds, remove_columns=python_ds.column_names)
    python_dpo = python_dpo.select(range(7000))

    math_ds = load_dataset("prhegde/preference-data-math-stack-exchange", split="train")
    math_ds = math_ds.select(range(5000))
    math_dpo = math_ds.map(map_math_ds, remove_columns=math_ds.column_names)

    distill_code_math = concatenate_datasets([preference_ds, code_pref, python_dpo, math_dpo])
    distill_code_math_split = distill_code_math.train_test_split(test_size=0.1, seed=42)
    distill_code_math_dataset_dict = DatasetDict({
        "train": distill_code_math_split["train"],
        "validation": distill_code_math_split["test"]
    })
    print("Train size:", len(distill_code_math_dataset_dict["train"]))
    print("Validation size:", len(distill_code_math_dataset_dict["validation"]))

    distill_code_math_dataset_dict.save_to_disk("MNLP_M3_dpo_dataset_reproduction")

    load_dotenv(find_dotenv())
    hf_user = os.getenv("HF_USERNAME")
    distill_code_math_dataset_dict.push_to_hub(
        f"{hf_user}/MNLP_M3_dpo_dataset_reproduction",
        token=os.getenv("HF_ACCESS_TOKEN"),
        private=False,
        commit_message="Reproduction of M3 distilled preference data"
    )

    
if __name__ == "__main__":
    main()