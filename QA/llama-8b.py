import json
import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompt import qa_prompt

# =========================
# MODEL
# =========================
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--sentence_type",
        type=str,
        required=True,
        choices=["src", "bt"]
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

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
    # LOAD PROCESSED IDS (resume-safe)
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

            if ex_id in processed:
                continue

            # =========================
            # FIND QUESTIONS
            # =========================
            question_keys = [
                k for k in data.keys()
                if k.startswith("questions_bt")
            ]

            for q_key in question_keys:
                bt_suffix = q_key.replace("questions_", "")  # bt1 / bt2
                questions = data[q_key]

                # assicura lista
                if isinstance(questions, str):
                    try:
                        questions = json.loads(questions)
                    except:
                        questions = [questions]

                # =========================
                # SELECT SENTENCE
                # =========================
                if args.sentence_type == "src":
                    sentence = data.get("src")
                    answers_key = f"answers_src_{bt_suffix}"
                else:  # bt
                    sentence = data.get(bt_suffix)
                    answers_key = f"answers_{bt_suffix}"

                if not sentence:
                    continue

                # resume-safe per singolo bt
                if answers_key in data:
                    continue

                answers = []
                print(f"[QA | {args.sentence_type.upper()} ← {q_key}] {ex_id}")

                for q in questions:
                    prompt = qa_prompt \
                        .replace("{{sentence}}", sentence) \
                        .replace("{{questions}}", q)

                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ]

                    input_ids = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    ).to(model.device)

                    with torch.no_grad():
                        outputs = model.generate(
                            input_ids=input_ids,
                            max_new_tokens=128,
                            eos_token_id=tokenizer.eos_token_id
                        )

                    response = outputs[0][input_ids.shape[-1]:]
                    answer = tokenizer.decode(
                        response,
                        skip_special_tokens=True
                    ).strip().strip('"\'')

                    answers.append(answer)

                data[answers_key] = answers

            # =========================
            # SAVE
            # =========================
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

    print("✅ QA completed successfully.")


if __name__ == "__main__":
    main()
