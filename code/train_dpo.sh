#!/bin/bash
# Create dataset
python train_dpo/prepare_distillation_questions.py
python train_dpo/distill_qwen.py
python train_dpo/distill_mistral.py
python train_dpo/distill_llama.py
python train_dpo/llm_as_a_judge.py
python train_dpo/create_distillation_dataset.py

# Train DPO model
python train_dpo/dpo_run.py