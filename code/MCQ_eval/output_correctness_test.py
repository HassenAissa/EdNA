import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import re
import torch
from typing import Dict, List, Any
import time
import json
from tqdm import tqdm
import multiprocessing as mp
import numpy as np

mp.set_start_method('spawn', force=True)

SUBJECTS =  [
    "mmlu",
    "sciq",
    "ai2_arc_easy",
    "ai2_arc_challenge",
    "aqua_rat"]


# Example questions and answers
FEW_SHOT_EXAMPLES = {
    "no_thinking": [
        {
            "question": """Post moem lividity persists till:
A. 2-3 days
B. 18 hours
C. Merges with putrefactive changes
D. 24 hours""",
            "answer": "D. 24 hours"
        },
        {
            "question": """A man is walking at a speed of 10 km per hour. After every kilometre, he takes rest for 5 minutes. How much time will he take to cover a distance of 5 kilometres?
A. 48 min.
B. 50 min.
C. 45 min.
D. 55 min.
E. None of these""",
            "answer": "B. 50 min."
        },
        {
            "question": """In which way do sunspots occur?
A. in fours
B. in pairs
C. in threes
D. by themselves""",
            "answer": "B. in pairs"
        },
        {
            "question": """What is the chemical symbol for gold?
A. Go
B. Gd
C. Au
D. Ag""",
            "answer": "C. Au"
        },
        {
            "question": """Which planet is known as the Red Planet?
A. Venus
B. Mars
C. Jupiter
D. Saturn""",
            "answer": "B. Mars"
        }
    ],
    "thinking": [
        {
            "question": """What produces almost one-half of the earth's oxygen through photosynthesis?
A) prokaryotes
B) algae
C) protists
D) arthropods""",
            "answer": """<think>While all options perform photosynthesis, algae in the ocean are responsible for producing about 50% of Earth's oxygen. Prokaryotes include both photosynthetic and non-photosynthetic organisms. Protists are diverse but not the major oxygen producers. Arthropods don't perform photosynthesis.</think>\nAnswer: B) algae"""
        },
        {
            "question": """What do you call a type of mixture that has the same composition throughout?
A) transfusion
B) structure
C) transition
D) solution""",
            "answer": """<think>A solution is a homogeneous mixture where the composition is uniform throughout. Transfusion refers to blood transfer. Structure and transition don't describe mixture types.</think>\nAnswer: D) solution"""
        },
        {
            "question": """What will spores that eventually germinate develop into?
A) new hyphae
B) hydra
C) yeast cells
D) gametes""",
            "answer": """<think>Spores are reproductive structures of fungi that grow into new hyphae. Hydra are animals, yeast cells reproduce by budding, and gametes are reproductive cells.</think>\nAnswer: A) new hyphae"""
        },
        {
            "question": """Which of these is NOT a fundamental force of nature?
A) gravitational force
B) electromagnetic force
C) nuclear force
D) frictional force""",
            "answer": """<think>The four fundamental forces are gravitational, electromagnetic, strong nuclear, and weak nuclear. Frictional force is an emergent phenomenon from electromagnetic interactions.</think>\nAnswer: D) frictional force"""
        },
        {
            "question": """What is the derivative of ln(x)?
A) 1/x
B) x
C) e^x
D) ln(x)""",
            "answer": """<think>The derivative of the natural logarithm function ln(x) is a fundamental result in calculus. The correct derivative is 1/x. The other options represent different functions entirely.</think>\nAnswer: A) 1/x"""
        }
    ]
}
PROMPTS = {
    "no_thinking": "The following are multiple choice questions (with answers) about knowledge and skills in advanced master-level STEM courses.\n\n<|question|>\nAnswer:",
    "thinking": "The following are multiple choice questions (with answers) about knowledge and skills in advanced master-level STEM courses.\n\n<|question|>\n Thinking and Answer:"
}
    


class MMLUEvaluator:
    def __init__(self, model_name="Qwen/Qwen3-0.6B-Base", device=None, save_path=None):
        self.save_path = save_path or "mmlu_results"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="bfloat16").to(self.device)
        self.model.eval()
        os.makedirs(self.save_path, exist_ok=True)
    
    def load_dataset(self, subject: str) -> List[Dict]:
        """Load and process MMLU dataset for a specific subject"""
        dataset = load_dataset("HAissa/MNLP_M3_mcqa_dataset", subject)["test"]
        
        processed_data = []
        for example in dataset:
            question = example["question"]
            answer = example['answer']
            processed_data.append({"question": question, "result": answer})
        
        return processed_data


    
    def build_prompt(self, example: Dict, mode: str = "no_thinking", few_shot: bool = False) -> str:
        """Construct the appropriate prompt based on mode and few-shot setting"""
        # Add validation to ensure PROMPTS is a dictionary
        if not isinstance(PROMPTS, dict):
            raise TypeError("PROMPTS must be a dictionary")
            
        if mode not in PROMPTS:
            raise ValueError(f"Mode must be one of {list(PROMPTS.keys())}, got '{mode}'")
            
        prompt_template = PROMPTS[mode]
        
        if few_shot:
            examples = FEW_SHOT_EXAMPLES[mode]
            prompt_parts = []
            for ex in examples:
                if mode  == "thinking":
                    prompt_parts.append(prompt_template.replace("<|question|>", ex["question"]) + f"{ex['answer']}")
                else:
                    prompt_parts.append(prompt_template.replace("<|question|>", ex["question"]) + f"{ex['answer']}")
                    
            prompt_parts.append(prompt_template.replace("<|question|>", example["question"]))
            return "\n\n".join(prompt_parts)
        else:
            return prompt_template.replace("<|question|>", example["question"])
    
    def check_answer(self, inferred_answer: str, correct_answer: str) -> bool:
        def normalize(ans):
            ans = ans.lower().strip()
            ans = ans.split("\n")[0]
            return ans
        
        correct = correct_answer.strip().lower()
        inferred = normalize(inferred_answer)
        return inferred == correct
        
    
    def evaluate_subject(self, subject: str, output_dir: str = "results", num_runs: int = 5) -> Dict[str, Any]:
        """Evaluate model performance on a specific subject with multiple runs for mean/std"""
        results = {}
        dataset = self.load_dataset(subject)
        
        for mode in ["no_thinking"]:
            for few_shot in [False, True]:
                key = f"{mode}_{'few_shot' if few_shot else 'zero_shot'}"
                
                accuracies = []
                times = []
                
                for run in range(num_runs):
                    start_time = time.time()
                    correct = 0
                    total = 0
                    batch_size = 64
                    if few_shot and mode == "thinking":
                        batch_size //= 2

                    for i in range(0, len(dataset), batch_size):
                        batch = dataset[i:i+batch_size]
                        prompts = [self.build_prompt(x, mode, few_shot) for x in batch]
                        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
                        
                        with torch.no_grad():
                            outputs = self.model.generate(
                                **inputs,
                                max_new_tokens=32 if mode == "no_thinking" else 512,
                                pad_token_id=self.tokenizer.eos_token_id,
                                temperature=0.7,
                                do_sample=True,
                            )

                        for j in range(len(outputs)):
                            full_answer = self.tokenizer.decode(outputs[j], skip_special_tokens=True)
                            full_answer = full_answer[len(prompts[j]):]
                            
                            if mode == "thinking":
                                answer = re.sub(r'<think>.*?</think>', '', full_answer, flags=re.DOTALL)
                                answer_parts = answer.split("answer: ")
                                answer = answer_parts[-1].strip() if len(answer_parts) > 1 else answer.strip()
                            else:
                                answer = full_answer.strip()
                            
                            batch[j]["correct"] = self.check_answer(answer, batch[j]["result"])

                        correct += sum(1 for x in batch if x["correct"])
                        total += len(batch)

                    accuracy = correct / total if total > 0 else 0.0
                    accuracies.append(accuracy)
                    times.append(time.time() - start_time)

                # Calculate statistics
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)
                mean_time = np.mean(times)
                
                results[f"{subject}_{key}"] = {
                    "mean_accuracy": float(mean_acc),
                    "std_accuracy": float(std_acc),
                    "mean_time": float(mean_time),
                    "runs": accuracies
                }
                

                
                print(f"{subject} {key}: {mean_acc:.2%} ± {std_acc:.2%} ({mean_time:.2f}s avg)")

        return results

def main():
    save_path = "final_model"
    evaluator = MMLUEvaluator("HAissa/MNLP_M3_mcqa_model", save_path=save_path)
    
    # Store overall results
    all_results = {}
    
    for subject in tqdm(SUBJECTS):
        subject_results = evaluator.evaluate_subject(subject)
        all_results.update(subject_results)
        with open(os.path.join(save_path, "results.json"), "w") as f:
            json.dump(all_results, f, indent=4)
    # Calculate and print overall statistics
    print("\n=== Final Summary ===")
    for config in ["no_thinking_zero_shot", "no_thinking_few_shot", 
                  "thinking_zero_shot", "thinking_few_shot"]:
        accs = [v["mean_accuracy"] for k,v in all_results.items() if config in k]
        if accs:
            print(f"{config}: {np.mean(accs):.2%} ± {np.std(accs):.2%} (n={len(accs)})")

if __name__ == "__main__":
    main()