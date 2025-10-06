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
    

def clean_answer_llama(example, field_name="Meta-Llama-3-8B-Instruct"):
    raw = example[field_name]
    if raw is None:
        return {field_name: None}

    markers = ["Correct answer:", "Answer:", "answer:"]
    last_idx = None
    for m in markers:
        idx = raw.rfind(m)
        if idx != -1 and (last_idx is None or idx > last_idx):
            last_idx = idx

    if last_idx is not None:
        sliced = raw[last_idx:]
    else:
        sliced = raw

    lower = sliced.lower()
    assist_token = "assistant"
    pos = lower.find(assist_token)
    if pos != -1:
        sliced = sliced[pos + len(assist_token):]

    return {field_name: sliced.strip()}


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
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    batch_size = 8

    llama_model, llama_tokenizer = load_model(model_name)
    if llama_model is None:
        return

    device = llama_model.device
    print(f"Loaded model on device: {device}")

    dataset = load_from_disk("distill_questions_qwen_mistral")
    print(f"Original dataset: {dataset}")
    print(type(dataset))  
    total = len(dataset)

    all_answers = []

    for start_idx in tqdm(range(0, total, batch_size), desc="Generating"):
        end_idx = min(start_idx + batch_size, total)
        batch   = dataset.select(range(start_idx, end_idx))

        # 1) Build “plain‐text” prompts for this batch
        texts = []
        for row in batch:
            question = row["prompt"]
            options  = row["options"]
            answer   = row["answer"]

            if options and options != [""] and answer:
                text = TEMPLATE_Q_O_A.format(
                    question=question,
                    options=options,
                    answer=answer
                )
            elif options and options != [""]:
                text = TEMPLATE_Q_O.format(
                    question=question,
                    options=options
                )
            else:
                text = TEMPLATE_Q_A.format(
                    question=question,
                    answer=answer
                )

            texts.append(text)

        chat_inputs = []
        for p in texts:
            messages = [{"role": "user", "content": p}]
            chat_text = llama_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors=None,
                enable_thinking=False
            )
            chat_inputs.append(chat_text)

        encodings = llama_tokenizer(
            chat_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(device)

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        prompt_lens = (attention_mask == 1).sum(dim=1).tolist()

        with torch.no_grad():
            generated = llama_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=llama_tokenizer.pad_token_id,
                eos_token_id=llama_tokenizer.eos_token_id,
                use_cache=True,
            )

        for seq, inp_len in zip(generated, prompt_lens):
            gen_part = seq[inp_len:]
            text = llama_tokenizer.decode(gen_part, skip_special_tokens=True)
            all_answers.append(text)

        if (start_idx // batch_size) % 50 == 0:
            torch.cuda.empty_cache()

    assert len(all_answers) == total, f"Expected {total} answers, got {len(all_answers)}"

    augmented = dataset.add_column("Meta-Llama-3-8B-Instruct", all_answers)
    print("Augmented dataset columns:", augmented.column_names)

    augmented = augmented.map(lambda ex: clean_answer_llama(ex))

    augmented.save_to_disk("distill_questions_qwen_mistral_llama")
    print("Saved dataset to 'distill_questions_qwen_mistral_llama'.")


if __name__ == "__main__":
    main()