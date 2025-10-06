#!/bin/bash
python code/train_mcqa/sft_run.py
python code/train_mcqa/grpo_run.py
python code/train_mcqa/merge_and_upload.py