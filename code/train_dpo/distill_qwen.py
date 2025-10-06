import torch
from datasets import load_from_disk, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def load_model(model_name: str):
    try:
        torch.cuda.empty_cache()

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        model.eval()
        return model, tokenizer

    except Exception as e:
        print(f"Failed to load model {model_name}: {e}")
        return None, None
    

def clean_thinking_tags(example, field_name = "Qwen/Qwen3-8B"):
    raw = example[field_name]
    if raw is None:
        return {field_name: None}

    parts = raw.split("</think>", 1)
    if len(parts) == 2:
        cleaned = parts[1].lstrip()
    else:
        cleaned = raw
    return {field_name: cleaned}


TEMPLATE_Q_O_A = """
You are a STEM expert. You are given the following question with options and the correct answer. Please provide a concise explanation for the correct answer.
Begin your answer with the correct answer and afterwards give your explanation. Please do not refer to this message.

Question:
{question}

Options:
{options}

Correct answer:
{answer}
""".strip()

TEMPLATE_Q_O = """
You are a STEM expert. You are given the following question with options. Please provide the correct answer and a concise explanation for it.
Begin your answer with the correct answer and afterwards give your explanation. Please do not refer to this message.

Question:
{question}

Options:
{options}
""".strip()

TEMPLATE_Q_A = """
You are a STEM expert. You are given the following question and the correct answer. Please provide a concise explanation for the correct answer.
Begin your answer with the correct answer and afterwards give your explanation. Please do not refer to this message.

Question:
{question}

Correct answer:
{answer}
""".strip()


def main():
    model_name = "Qwen/Qwen3-8B"
    batch_size = 8

    qwen_model, qwen_tokenizer = load_model(model_name)
    if qwen_model is None:
        return

    device = qwen_model.device
    print(f"Loaded model on device: {device}")

    dataset = load_from_disk("distill_questions")
    print(f"Original dataset: {dataset}")
    print(type(dataset))  
    total = len(dataset)

    all_answers = []

    for start_idx in tqdm(range(0, total, batch_size), desc="Generating"):
        indices = list(range(start_idx, min(start_idx + batch_size, total)))
        batch = dataset.select(indices)
        prompts = []
        for row in batch:
            question = row["prompt"]
            options = row["options"]
            answer = row["answer"]

            if options and options != [""] and answer:
                p = TEMPLATE_Q_O_A.format(
                    question=question,
                    options=options,
                    answer=answer,
                )
            elif options and options != [""]:
                p = TEMPLATE_Q_O.format(
                    question=question,
                    options=options
                )
            else:
                p = TEMPLATE_Q_A.format(
                    question=question,
                    answer=answer,
                )
            prompts.append(p)

        chat_inputs = []
        for p in prompts:
            messages = [{"role": "user", "content": p}]
            chat_text = qwen_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            chat_inputs.append(chat_text)

        inputs = qwen_tokenizer(
            chat_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(device)

        with torch.no_grad():
            output_ids = qwen_model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=qwen_tokenizer.eos_token_id,
                use_cache=True,
            )

        input_lens = inputs.attention_mask.sum(dim=1).tolist()
        for seq, inp_len in zip(output_ids, input_lens):
            gen_part = seq[inp_len:]
            text = qwen_tokenizer.decode(gen_part, skip_special_tokens=True)
            all_answers.append(text)

        if (start_idx // batch_size) % 50 == 0:
            torch.cuda.empty_cache()

    assert len(all_answers) == total, f"Got {len(all_answers)} answers for {total} examples"

    augmented = dataset.add_column("Qwen/Qwen3-8B", all_answers)
    print("Augmented dataset features:", augmented.features)

    augmented = augmented.map(clean_thinking_tags)

    augmented.save_to_disk("distill_questions_qwen")
    print("Saved augmented dataset to 'distill_questions_with_answers'.")


if __name__ == "__main__":
    main()