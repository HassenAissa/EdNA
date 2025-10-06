import os
import logging
import torch
import random
from tqdm import tqdm
from multiprocessing import cpu_count

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

RAFT_DATASET_ID = "Lysandrec/MNLP_M3_rag_dataset"
GENERATOR_MODEL_NAME = "HAissa/MNLP_M3_mcqa_model"
OUTPUT_DIR = "./fine_tuned_raft_generator"
HF_REPO_ID = "Lysandrec/MNLP_M3_rag_model"
MODEL_MAX_LENGTH = 1024

# Training Hyperparameters
LEARNING_RATE = 2e-5
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
NUM_EPOCHS = 1
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 20
SAVE_STEPS = 100

# DoRA Configuration
DORA_R = 32
DORA_ALPHA = 64
DORA_DROPOUT = 0.05
DORA_TARGET_MODULES = ["c_attn", "c_proj", "gate_proj", "up_proj", "down_proj"]

# RAFT-specific configuration
GOLDEN_DOC_IN_CONTEXT_RATIO = 0.8

# Preprocessing optimization settings
PREPROCESSING_BATCH_SIZE = 1000  # Increased batch size for preprocessing
NUM_PROC = min(cpu_count() - 1, 8)  # Use multiple cores for preprocessing
CACHE_DIR = "./cache"  # Cache preprocessed data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def format_raft_prompt_batch(questions, golden_docs, distractor_docs, cot_answers, include_golden_flags):
    """Optimized batch prompt formatting"""
    prompts = []
    
    for i in range(len(questions)):
        context_parts = []
        
        if include_golden_flags[i] and golden_docs[i]:
            context_parts.append(f"Relevant Document:\n{golden_docs[i]}")
        
        if distractor_docs[i]:
            # Handle distractor documents efficiently
            if isinstance(distractor_docs[i], str):
                distractors = [doc.strip() for doc in distractor_docs[i].split('|||') if doc.strip()]
            else:
                distractors = distractor_docs[i] if isinstance(distractor_docs[i], list) else [distractor_docs[i]]
                
            for j, dist_doc in enumerate(distractors):
                context_parts.append(f"Distractor Document {j+1}:\n{dist_doc}")
        
        context_str = "\n\n".join(context_parts) if context_parts else "No documents provided."
        prompt = f"Context:\n{context_str}\n\nQuestion: {questions[i]}\n\nAnswer:"
        
        if cot_answers[i]:  # Only add answer if provided (for full training text)
            prompts.append(f"{prompt} {cot_answers[i]}")
        else:  # For prompt-only (masking purposes)
            prompts.append(prompt)
    
    return prompts

def main():
    logger.info("Starting RAFT generator fine-tuning...")
    logger.info(f"Dataset: {RAFT_DATASET_ID}")
    logger.info(f"Model: {GENERATOR_MODEL_NAME}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info(f"HuggingFace Repo: {HF_REPO_ID}")
    logger.info(f"Preprocessing: {NUM_PROC} cores, batch_size={PREPROCESSING_BATCH_SIZE}")

    # 1. Load Dataset with early filtering
    logger.info("Loading and filtering RAFT dataset...")
    dataset = load_dataset(RAFT_DATASET_ID, split="train")
    logger.info(f"Original dataset size: {len(dataset)}")

    # Verify required columns
    required_cols = ["question", "golden_document", "distractor_documents", "chain_of_thought_answer"]
    missing_cols = [col for col in required_cols if col not in dataset.column_names]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}. Available: {dataset.column_names}")

    # Early filtering to remove invalid examples
    logger.info("Filtering invalid examples...")
    def filter_valid_examples(example):
        return (
            example['question'] and str(example['question']).strip() and
            example['chain_of_thought_answer'] and str(example['chain_of_thought_answer']).strip() and
            len(str(example['question']).strip()) > 10 and  # Minimum question length
            len(str(example['chain_of_thought_answer']).strip()) > 10  # Minimum answer length
        )
    
    filtered_dataset = dataset.filter(
        filter_valid_examples,
        num_proc=NUM_PROC,
        desc="Filtering valid examples"
    )
    logger.info(f"Filtered dataset size: {len(filtered_dataset)} (removed {len(dataset) - len(filtered_dataset)} invalid examples)")

    # 2. Load Tokenizer and Model
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        GENERATOR_MODEL_NAME,
        device_map="auto",
        trust_remote_code=True
    )

    # 3. Apply DoRA
    logger.info("Applying DoRA...")
    dora_config = LoraConfig(
        r=DORA_R,
        lora_alpha=DORA_ALPHA,
        lora_dropout=DORA_DROPOUT,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
        target_modules=DORA_TARGET_MODULES,
        use_dora=True
    )
    
    model = get_peft_model(model, dora_config)
    model.print_trainable_parameters()

    # 4. Optimized Dataset Preprocessing
    def preprocess_function_optimized(examples):
        batch_size = len(examples['question'])
        
        # Pre-generate include_golden flags for the entire batch
        include_golden_flags = [random.random() < GOLDEN_DOC_IN_CONTEXT_RATIO for _ in range(batch_size)]
        
        # Generate prompts in batch for full training texts
        full_texts = format_raft_prompt_batch(
            examples['question'],
            examples['golden_document'],
            examples['distractor_documents'],
            examples['chain_of_thought_answer'],
            include_golden_flags
        )
        
        # Add EOS token to all texts
        full_texts = [text + tokenizer.eos_token for text in full_texts]
        
        # Generate prompt-only texts for masking (batch operation)
        prompt_only_texts = format_raft_prompt_batch(
            examples['question'],
            examples['golden_document'],
            examples['distractor_documents'],
            [""] * batch_size,  # Empty answers for prompt-only
            include_golden_flags
        )
        
        # Tokenize all texts in batch (more efficient)
        model_inputs = tokenizer(
            full_texts,
            max_length=MODEL_MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )
        
        # Tokenize prompt parts for masking (batch operation)
        prompt_inputs = tokenizer(
            prompt_only_texts,
            max_length=MODEL_MAX_LENGTH,
            truncation=True,
            return_tensors=None,
        )
        
        # Create labels and apply masking efficiently
        labels = []
        for i in range(batch_size):
            label = list(model_inputs["input_ids"][i])
            prompt_length = len(prompt_inputs["input_ids"][i])
            
            # Mask prompt part
            for j in range(min(prompt_length, len(label))):
                label[j] = -100
                
            labels.append(label)
        
        model_inputs["labels"] = labels
        return model_inputs

    # Preprocessing with optimizations
    logger.info("Starting optimized preprocessing...")
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    tokenized_dataset = filtered_dataset.map(
        preprocess_function_optimized,
        batched=True,
        batch_size=PREPROCESSING_BATCH_SIZE,
        num_proc=NUM_PROC,
        remove_columns=filtered_dataset.column_names,
        desc="Tokenizing dataset",
        cache_file_name=f"{CACHE_DIR}/tokenized_dataset.cache"
    )
    
    logger.info(f"✅ Preprocessing completed! Final dataset size: {len(tokenized_dataset)}")

    # 5. Training
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="tensorboard",
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=min(4, NUM_PROC),  # Optimize data loading
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # 6. Train and Save
    logger.info("Starting training...")
    trainer.train()
    
    logger.info(f"Saving model locally to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # 7. Push to HuggingFace Hub
    logger.info(f"Pushing model to HuggingFace Hub: {HF_REPO_ID}...")
    try:
        # Push the model to HuggingFace Hub
        model.push_to_hub(HF_REPO_ID, use_temp_dir=False)
        tokenizer.push_to_hub(HF_REPO_ID, use_temp_dir=False)
        logger.info(f"Model successfully pushed to {HF_REPO_ID}")
    except Exception as e:
        logger.error(f"Failed to push to HuggingFace Hub: {e}")
        logger.info("Model is still saved locally and can be manually uploaded")
    
    logger.info("RAFT training completed!")

if __name__ == "__main__":
    main() 