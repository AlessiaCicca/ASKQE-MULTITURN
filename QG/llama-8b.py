from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import argparse
import re

from prompt import prompts

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
INPUT_PATH = "/content/ASKQE-Hallucination/dataNoPerturb/src_mt_bt.jsonl"
def main():
    # =========================
    # LOAD MODEL & TOKENIZER
    # =========================
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    # =========================
    # ARGS
    # =========================
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    args = parser.parse_args()

    # =========================
    # LOAD DATASET
    # =========================
    with open(INPUT_PATH, "r", encoding="utf-8") as f_in, \
         open(args.output_path, "a", encoding="utf-8") as f_out:

        for line in f_in:
            data = json.loads(line)

            # 🔍 trova tutti i campi bt*
            bt_fields = [k for k in data.keys() if k.startswith("bt")]

            if not bt_fields:
                continue

            for bt_field in bt_fields:
                sentence = data.get(bt_field)
                if not sentence:
                    continue

                print(f"[{bt_field}] {sentence}")

                prompt_template = prompts[args.prompt]

                # -------- prompt construction --------
                if args.prompt == "semantic":
                    semantic = data.get("semantic_roles")
                    if semantic:
                        prompt = prompt_template.replace(
                            "{{sentence}}", sentence
                        ).replace(
                            "{{semantic_roles}}", semantic
                        )
                    else:
                        prompt = prompt_template.replace("{{sentence}}", sentence)

                elif args.prompt == "atomic":
                    atomics = data.get("atomic_facts")
                    if atomics:
                        prompt = prompt_template.replace(
                            "{{sentence}}", sentence
                        ).replace(
                            "{{atomic_facts}}", str(atomics)
                        )
                    else:
                        prompt = prompt_template.replace("{{sentence}}", sentence)

                else:  # vanilla
                    prompt = prompt_template.replace("{{sentence}}", sentence)

                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ]

                # =========================
                # CHAT TEMPLATE → TESTO
                # =========================
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                # =========================
                # TOKENIZE
                # =========================
                inputs = tokenizer(
                    prompt_text,
                    return_tensors="pt"
                ).to(model.device)

                # =========================
                # GENERATE
                # =========================
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=32,
                        eos_token_id=tokenizer.eos_token_id
                    )

                # =========================
                # DECODE
                # =========================
                response = outputs[0][inputs["input_ids"].shape[-1]:]
                questions = tokenizer.decode(
                    response,
                    skip_special_tokens=True
                ).strip()

                out_key = f"questions_{bt_field}"
                data[out_key] = questions

                print(f"> {out_key}: {questions}")
                print("=" * 60)

            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
