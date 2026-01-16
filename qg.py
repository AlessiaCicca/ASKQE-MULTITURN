from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import argparse
import os

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

def main():
    # =========================
    # ARGS
    # =========================
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--prompt_path", type=str, required=True,
                        help="Path to prompt template file")
    args = parser.parse_args()

    # =========================
    # LOAD PROMPT
    # =========================
    if not os.path.exists(args.prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {args.prompt_path}")

    with open(args.prompt_path, "r", encoding="utf-8") as f:
        PROMPT_TEMPLATE = f.read()

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
    # PROCESS DATASET
    # =========================
    with open(args.input_path, "r", encoding="utf-8") as f_in, \
         open(args.output_path, "a", encoding="utf-8") as f_out:

        for line in f_in:
            data = json.loads(line)

            bt_field = "bt"
            sentence = data.get(bt_field)
            if not sentence:
                continue

            print(f"[{bt_field}] {sentence}")

            # =========================
            # BUILD PROMPT
            # =========================
            prompt = PROMPT_TEMPLATE.replace("{{sentence}}", sentence)

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]

            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

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
                    max_new_tokens=64,
                    eos_token_id=tokenizer.eos_token_id
                )

            # =========================
            # DECODE
            # =========================
            response = outputs[0][inputs["input_ids"].shape[-1]:]
            raw_output = tokenizer.decode(
                response,
                skip_special_tokens=True
            ).strip()

            print(f"> Raw output:\n{raw_output}")

            # =========================
            # PARSE & VALIDATE
            # =========================
            # Pulisci l'output
            cleaned = raw_output.strip().strip('`').strip()

            # Rimuovi eventuali wrapper ```python o ```json
            if cleaned.startswith('python') or cleaned.startswith('json'):
                cleaned = cleaned.split('\n', 1)[-1].strip()
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3].strip()

            # Prova a parsare come JSON
            try:
                questions = json.loads(cleaned)
                
                # Valida che sia una lista
                if not isinstance(questions, list):
                    print(f"[WARN] Output is not a list: {type(questions)}")
                    questions = [cleaned]
                
                # Pulisci le domande: rimuovi stringhe vuote
                questions = [q.strip() for q in questions if q and isinstance(q, str) and len(q.strip()) > 3]
                
                # Verifica che ci siano domande valide
                if not questions:
                    print(f"[WARN] No valid questions found")
                    continue
                    
            except json.JSONDecodeError as e:
                print(f"[ERROR] JSON parse failed: {e}")
                print(f"[ERROR] Output was: {cleaned[:200]}")
                questions = [q.strip() for q in cleaned.split('\n') if q.strip() and len(q.strip()) > 3]
                if not questions:
                    continue

            # ✅ SALVA DIRETTAMENTE LA LISTA (non fare json.dumps)
            data["questions_bt"] = questions  # ← Cambia qui!

            print(f"> Parsed {len(questions)} questions:")
            for i, q in enumerate(questions, 1):
                print(f"  {i}. {q}")
            print("=" * 60)

            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
