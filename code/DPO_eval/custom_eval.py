from lighteval.tasks.requests import Doc
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
import random

def preference_pair(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["prompt"],
        text_chosen=line["chosen"],
        text_rejected=line["rejected"],
        instruction="",
        choices = [],
        gold_index=0,
    )

overall = LightevalTaskConfig(
    name="mnlp_dpo_evals",
    prompt_function=preference_pair,
    suite=["community"],
    hf_subset="",
    hf_repo="levinius/DPO_eval",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    metric=[Metrics.reward_model_acc],
    generation_size=-1,
    stop_sequence=None,
    trust_dataset=True,
    limited_num_samples=0
    )

helper_steer = LightevalTaskConfig(
    name="mnlp_helper_steer_evals",
    prompt_function=preference_pair,
    suite=["community"],
    hf_subset="",
    hf_repo="levinius/HelpSteer_stem_eval",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    metric=[Metrics.reward_model_acc],
    generation_size=-1,
    stop_sequence=None,
    trust_dataset=True,
    limited_num_samples=0
    )

reward_bench = LightevalTaskConfig(
    name="mnlp_reward_bench_evals",
    prompt_function=preference_pair,
    suite=["community"],
    hf_subset="",
    hf_repo="levinius/reward_bench_eval",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    metric=[Metrics.reward_model_acc],
    generation_size=-1,
    stop_sequence=None,
    trust_dataset=True,
    limited_num_samples=0
    )
    
TASKS_TABLE = [overall] + [helper_steer] + [reward_bench]
with open("rb+hs.txt", "w") as f:
    for t in TASKS_TABLE:
        for suite in t.suite:
            f.write(f"{suite}|{t.name}|0|0\n")

print("Task list written to mcqa_tasks_list")