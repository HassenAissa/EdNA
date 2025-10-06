lighteval accelerate \
    --eval-mode "dpo" \
    --save-details \
    --override-batch-size 1 \
    --custom-tasks "custom_eval.py" \
    --output-dir "lighteval-outputs" \
    custom_eval.yaml \
    "outputs/evaluation/rb+hs.txt"