import torch
from datasets import load_from_disk
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

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
        return model, tokenizer

    except Exception as e:
        print(f"Failed to load model {model_name}: {e}")
        return None, None
    

def clean_answer_mistral(example, field_name="Mistral-7B-Instruct-v0.2"):
    raw = example[field_name]
    if raw is None:
        return {field_name: None}
    markers = ["Correct answer:", "Answer:", "answer:"]
    first_idx = None
    for m in markers:
        idx = raw.find(m)
        if idx != -1 and (first_idx is None or idx < first_idx):
            first_idx = idx
    if first_idx is not None:
        sliced = raw[first_idx:]
    else:
        sliced = raw
    cleaned = sliced.replace("</s>", "")
    return {field_name: cleaned.strip()}


TEMPLATE_Q_O_A_mistral = """
You are a STEM expert. You are given the following question with options and the correct answer. Please provide a concise, short explanation for the correct answer.
Provide your answer in this exact format: Answer: <correct answer> \n\nExplanation: <short explanation>
Please do not refer to this message.

Question:
{question}

Options:
{options}

Correct answer:
{answer}
""".strip()

TEMPLATE_Q_O_mistral = """
You are a STEM expert. You are given the following question with options. Please provide the correct answer and a concise, short explanation for it.
Provide your answer in this exact format: Answer: <correct answer> \n\nExplanation: <short explanation>
Please do not refer to this message.

Question:
{question}

Options:
{options}
""".strip()

TEMPLATE_Q_A_mistral = """
You are a STEM expert. You are given the following question and the correct answer. Please provide a concise, short explanation for the correct answer.
Provide your answer in this exact format: Answer: <correct answer> \n\nExplanation: <short explanation>
Please do not refer to this message.

Question:
{question}

Correct answer:
{answer}
""".strip()


def main():
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    batch_size = 8

    mistral_model, mistral_tokenizer = load_model(model_name)
    if mistral_model is None:
        return

    device = mistral_model.device
    print(f"Loaded model {model_name} on device: {device}")

    dataset = load_from_disk("distill_questions_qwen")
    print(f"Loaded dataset with {len(dataset)} rows. Columns: {dataset.column_names}")
    total = len(dataset)

    all_answers = []

    for start_idx in tqdm(range(0, total, batch_size), desc="Generating"):
        end_idx = min(start_idx + batch_size, total)
        indices = list(range(start_idx, end_idx))
        batch = dataset.select(indices)

        prompts = []
        for row in batch:
            question = row["prompt"]
            options  = row["options"]
            answer   = row["answer"]

            if options and options != [""] and answer:
                text = TEMPLATE_Q_O_A_mistral.format(
                    question=question,
                    options=options,
                    answer=answer,
                )
            elif options and options != [""]:
                text = TEMPLATE_Q_O_mistral.format(
                    question=question,
                    options=options
                )
            else:
                text = TEMPLATE_Q_A_mistral.format(
                    question=question,
                    answer=answer,
                )
            prompts.append(text)

        encodings = mistral_tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(mistral_model.device)
    
        input_ids      = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        prompt_lens    = attention_mask.sum(dim=1).tolist()
    
        with torch.no_grad():
            generated = mistral_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=250,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=mistral_tokenizer.convert_tokens_to_ids("</s>"),
                eos_token_id=mistral_tokenizer.convert_tokens_to_ids("</s>"),
                use_cache=True,
            )
    
        for i, seq_ids in enumerate(generated):
            plen = prompt_lens[i]
            gen_ids = seq_ids[plen:]
            answer = mistral_tokenizer.decode(gen_ids.tolist())
            all_answers.append(answer)
    
        if (start_idx // batch_size) % 50 == 0:
            torch.cuda.empty_cache()
    
    assert len(all_answers) == total, f"Expected {total} answers, got {len(all_answers)}"

    augmented = dataset.add_column("Mistral-7B-Instruct-v0.2", all_answers)
    print("Augmented dataset columns:", augmented.column_names)

    augmented = augmented.map(lambda ex: clean_answer_mistral(ex))

    augmented.save_to_disk("distill_questions_qwen_mistral")
    print("Saved augmented dataset to 'distill_questions_qwen_mistral'.")


if __name__ == "__main__":
    main()