
from lighteval.tasks.requests import Doc
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
import random
# Set to 0 for all samples. 
NSAMPLES = 0 


def prompt_func(line, task_name: str = None):
    topic = "knowledge and skills in advanced master-level STEM courses"
    prompt = f"The following are multiple choice questions (with answers) about {topic.replace('_', ' ')}.\n\n"
    prompt += line["question"]
    prompt += "\nAnswer:"
    options = line["question"].split("\n")
    options = [" " + option for option in options if option.strip().startswith(('A.', 'B.', 'C.', 'D.', 'E.', "F.", "G.", "H.", "I.", "J."))]
    correct_answer = line["answer"]
    letter = correct_answer[0]
    if letter in LETTER_INDICES:
        gold_ix = LETTER_INDICES.index(letter)
    elif letter.isdigit():
        gold_ix = int(letter)-1
    #print(options)
    #print(gold_ix)
    return Doc(
        task_name=task_name,
        query=prompt,
        choices=options,
        gold_index=gold_ix,
        instruction=f"The following are multiple choice questions (with answers) about {topic.replace('_', ' ')}.\n\n",
    )
task_names = [
    "sciq",
    "ai2_arc_easy",
    "ai2_arc_challenge",
    "aqua_rat",
    "mmlu",
    # "MMLU-Pro"
]
TASKS_TABLE = []
for task_name in task_names:
    task = LightevalTaskConfig(
            name=task_name,
            prompt_function=prompt_func,
            suite=["community"],  
            hf_subset=task_name,
            hf_repo="HAissa/MNLP_M3_mcqa_dataset",
            hf_avail_splits=["test"],
            evaluation_splits=["test"],
            metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
            generation_size=-1,
            stop_sequence=None,
            trust_dataset=True,
            limited_num_samples=NSAMPLES)
    TASKS_TABLE.append(task)



with open("evals/lighteval_test.txt", "w") as f:
    for t in TASKS_TABLE:
        for suite in t.suite:
            f.write(f"{suite}|{t.name}|0|0\n")
