import json
import random
from datasets import load_dataset, concatenate_datasets, Dataset
from dotenv import load_dotenv, find_dotenv
from pathlib import Path


def process_m1_questions():
    here = Path(__file__).resolve().parent
    project_root = here.parent.parent
    data_path = project_root/"data"/"m1_preference_data.json"

    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    m1_questions = []
    for ex in data:
        prompt = ex["question_body"]
        options = ex['question_options'] if isinstance(ex["question_options"], list) else []
        answer = ex['question_answer'] if isinstance(ex["question_answer"], str) else ""
        source = "m1_preference_data"
        m1_questions.append({
            "prompt": prompt,
            "options": options,
            "answer": answer,
            "source": source
        })
    return m1_questions


def process_sciq(ex):
    prompt = ex["question"]
    options = [ex["correct_answer"], ex["distractor1"], ex["distractor2"], ex["distractor3"]]
    answer = ex["correct_answer"]
    source = "sciq"
    random.shuffle(options)
    return {
        "prompt": prompt,
        "options": options,
        "answer": answer,
        "source": source
    }


def process_math(ex):
    prompt = ex["prompt"]
    options = [""]
    answer = ""
    source = "tulu3-personas-math"

    return {
        "prompt": prompt,
        "options": options,
        "answer": answer,
        "source": source
    }


def main():
    m1_questions = process_m1_questions()
    m1_ds = Dataset.from_list(m1_questions)

    sciq_raw = load_dataset("allenai/sciq", split="train")
    sciq_ds = sciq_raw.map(process_sciq, remove_columns=sciq_raw.column_names)

    math_raw = load_dataset("allenai/tulu-3-sft-personas-math", split="train")
    math_selected = math_raw.select(range(5000))
    math_ds = math_selected.map(process_math, remove_columns=math_selected.column_names)

    distill_questions = concatenate_datasets([m1_ds, sciq_ds, math_ds])
    distill_questions.save_to_disk("distill_questions")

if __name__ == "__main__":
    main()