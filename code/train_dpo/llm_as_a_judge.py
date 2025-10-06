import gpt_wrapper
import tiktoken
import json
from gpt_wrapper.chat import Chat
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import load_from_disk
import os


TEMPLATE_Q_O_A = """
You are an expert STEM educator and evaluator. Your job is to rank three student‐style answers to a STEM question. You must judge primarily on whether the answer is correct. If two or more are equally correct, break the tie by clarity of explanation—answer that is easier to follow gets the higher rank.

Below is the question with answer options and the correct answer, followed by three candidate answers labeled A, B, and C.  

Question:
{question_text}

Options:
{options}

Correct answer:
{correct_answer}

---  
Answer A:
{answer_A}

Answer B:
{answer_B}

Answer C:
{answer_C}

---

Step 1: For each answer (A, B, and C), judge whether it is mathematically or factually correct.  
Step 2: If more than one is correct, compare clarity and completeness of explanation.  
Step 3: At the end, output a single line in this exact format with no extra text before or after: Ranking: <best_label> > <middle_label> > <worst_label>. For example: Ranking: B > A > C. Do not output any other commentary or text.""".strip()


TEMPLATE_Q_O = """
You are an expert STEM educator and evaluator. Your job is to rank three student‐style answers to a STEM question. You must judge primarily on whether the answer is correct. If two or more are equally correct, break the tie by clarity of explanation—answer that is easier to follow gets the higher rank.

Below is the question with answer options, followed by three candidate answers labeled A, B, and C.  

Question:
{question_text}

Options:
{options}

---  
Answer A:
{answer_A}

Answer B:
{answer_B}

Answer C:
{answer_C}

---

Step 1: For each answer (A, B, and C), judge whether it is mathematically or factually correct.  
Step 2: If more than one is correct, compare clarity and completeness of explanation.  
Step 3: At the end, output a single line in this exact format with no extra text before or after: Ranking: <best_label> > <middle_label> > <worst_label>. For example: Ranking: B > A > C. Do not output any other commentary or text.""".strip()


TEMPLATE_Q_A = """
You are an expert STEM educator and evaluator. Your job is to rank three student‐style answers to a STEM question. You must judge primarily on whether the answer is correct. If two or more are equally correct, break the tie by clarity of explanation—answer that is easier to follow gets the higher rank.

Below is the question with the correct answer, followed by three candidate answers labeled A, B, and C.  

Question:
{question_text}

Correct answer:
{correct_answer}

---  
Answer A:
{answer_A}

Answer B:
{answer_B}

Answer C:
{answer_C}

---

Step 1: For each answer (A, B, and C), judge whether it is mathematically or factually correct.  
Step 2: If more than one is correct, compare clarity and completeness of explanation.  
Step 3: At the end, output a single line in this exact format with no extra text before or after: Ranking: <best_label> > <middle_label> > <worst_label>. For example: Ranking: B > A > C. Do not output any other commentary or text.""".strip()



def main():
    load_dotenv()
    api_key = os.environ.get("GPT_API_KEY")
    gpt_wrapper.api_base = "http://mnlp-backend-lb-1062233132.eu-central-1.elb.amazonaws.com"
    gpt_wrapper.api_key = api_key

    ds = load_from_disk("distill_questions_qwen_mistral_llama")
    total = len(ds)
    print(f"Loaded dataset with {total} rows. Columns: {ds.column_names}")

    total_tokens = 0
    judge_column = [None] * total

    for i in tqdm(range(total), desc="Judging examples"):
        row = ds[i]
        question = row["prompt"]
        options = row["options"]
        answer = row["answer"]
        qwen_answer = row["Qwen/Qwen3-8B"]
        mistral_answer = row["Mistral-7B-Instruct-v0.2"]
        llama_answer = row["Meta-Llama-3-8B-Instruct"]
        
        if options and options != [""] and answer:
            prompt = TEMPLATE_Q_O_A.format(
                question_text=question,
                options=options,
                correct_answer=answer,
                answer_A=qwen_answer,
                answer_B=mistral_answer,
                answer_C=llama_answer
            )
        elif options and options != [""]:
            prompt = TEMPLATE_Q_O.format(
                question_text=question,
                options=options,
                answer_A=qwen_answer,
                answer_B=mistral_answer,
                answer_C=llama_answer
            )
        else:
            prompt = TEMPLATE_Q_A.format(
                question_text=question,
                correct_answer=answer,
                answer_A=qwen_answer,
                answer_B=mistral_answer,
                answer_C=llama_answer
            )

        num_tokens = len(tiktoken.encoding_for_model("gpt-4").encode(prompt))
        total_tokens += num_tokens
        answer = Chat.create("judge").ask(content = prompt).to_dict()
        judge_answer = answer["content"]
        judge_column[i] = judge_answer

        if (i + 1) % 1000 == 0:
            backup_ds = ds.add_column("ranking", judge_column)
            backup_dir = f"distill_judge_backup_{i+1}"
            os.makedirs(backup_dir, exist_ok=True)
            backup_ds.save_to_disk(backup_dir)
            print(f"→ Backup saved after {i+1} iterations to '{backup_dir}'")
            print(f"Used tokens: {total_tokens}, budget: {Chat.budget()}")

    assert len(judge_column) == total, f"Expected {total} rankings, got {len(judge_column)}"
    final_ds = ds.add_column("ranking", judge_column)
    print("Final dataset columns:", final_ds.column_names)
    final_ds.save_to_disk("distill_questions_with_judgment")
    print(f"Saved final judged dataset to distill_questions_with_judgment")

    print("Used tokens", total_tokens)


if __name__ == "__main__":
    main()