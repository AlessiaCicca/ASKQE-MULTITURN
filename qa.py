import json
import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


def load_prompt(prompt_path: str) -> str:
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def answer_questions(tokenizer, model, prompt_template, sentence, questions):
    """
    Answer a list of questions given a sentence.
    Always returns a list of strings with the SAME length as questions.
    """
    questions_str = json.dumps(questions, ensure_ascii=False)

    prompt = (
        prompt_template
        .replace("{{sentence}}", sentence)
        .replace("{{questions}}", questions_str)
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=512,  # ⬅️ necessario per liste lunghe
                eos_token_id=tokenizer.eos_token_id
            )

        response = outputs[0][input_ids.shape[-1]:]
        text = tokenizer.decode(response, skip_special_tokens=True).strip()
        text = text.strip('"\'')

        # prova JSON
        try:
            answers = json.loads(text)
        except json.JSONDecodeError:
            print(f"[WARN] Invalid JSON output, fallback:\n{text[:200]}")
            answers = ["No Answer"] * len(questions)

        if not isinstance(answers, list):
            answers = ["No Answer"] * len(questions)

        # forza allineamento 1–1
        if len(answers) != len(questions):
            print(f"[WARN] Mismatch: {len(questions)} questions vs {len(answers)} answers")
            answers = answers[:len(questions)]
            answers += ["No Answer"] * (len(questions) - len(answers))

    except Exception as e:
        print(f"[ERROR] QA failed: {e}")
        answers = ["No Answer"] * len(questions)

    return answers


def main():
    # =========================
    # ARGS
    # =========================
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--prompt_path", type=str, required=True)
    args = parser.parse_args()

    # =========================
    # LOAD PROMPT
    # =========================
    prompt_template = load_prompt(args.prompt_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # =========================
    # MODEL
    # =========================
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map="auto"
    ).to(device)

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # =========================
    # LOAD PROCESSED IDS
    # =========================
    processed = set()
    if os.path.exists(args.output_path):
        with open(args.output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    processed.add(json.loads(line)["id"])
                except:
                    pass

    # =========================
    # PROCESS
    # =========================
    with open(args.input_path, "r", encoding="utf-8") as f_in, \
         open(args.output_path, "a", encoding="utf-8") as f_out:

        for line in f_in:
            data = json.loads(line)
            ex_id = data.get("id")

            if not ex_id or ex_id in processed:
                continue

            # ---------- questions ----------
            raw_questions = data.get("questions_bt")
            if not raw_questions:
                continue

            if isinstance(raw_questions, str):
                try:
                    questions = json.loads(raw_questions)
                except:
                    questions = [q.strip() for q in raw_questions.split("\n") if q.strip()]
            else:
                questions = list(raw_questions)

            if not questions:
                continue

            src = data.get("src")
            bt = data.get("bt")
            if not src or not bt:
                continue

            print(f"[QA | SRC + BT] {ex_id}")

            data["answers_src"] = answer_questions(
                tokenizer, model, prompt_template, src, questions
            )
            data["answers_bt"] = answer_questions(
                tokenizer, model, prompt_template, bt, questions
            )

            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

    print("✅ QA (SRC + BT) completed correctly.")


if __name__ == "__main__":
    main()
