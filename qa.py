from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import argparse
import os
import re

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# LOAD PROMPT
def load_prompt(prompt_path: str, prompt_key: str) -> str:
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    if prompt_key not in prompts:
        raise KeyError(
            f"Prompt key '{prompt_key}' not found. "
            f"Available keys: {list(prompts.keys())}"
        )

    prompt = prompts[prompt_key]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Prompt '{prompt_key}' is empty or invalid")

    return prompt


#The following strategies are designed to enforce that the model’s output is strictly formatted as a flat list of answers.
def parse_model_output(text: str, num_questions: int, ex_id: str, which: str, debug: bool) -> list:
    # Strategy 1:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and len(parsed) == num_questions:
            if all(isinstance(x, str) or x is None for x in parsed):
                return [str(a) if a else "No Answer" for a in parsed]
    except:
        pass
    # Strategy 2:
    try:
        cleaned = text.strip()
        if '"], [' in cleaned or '], [' in cleaned or cleaned.startswith('[['):
            matches = re.findall(r'"([^"]*)"', cleaned)
            
            if len(matches) >= num_questions:
                return matches[:num_questions]
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, list):
                    flattened = []
                    for item in parsed:
                        if isinstance(item, list):
                            flattened.extend([str(x) if x else "" for x in item])
                        elif item:
                            flattened.append(str(item))
                    
                    if len(flattened) >= num_questions:
                        return flattened[:num_questions]
            except:
                pass              
    except:
        pass
    # Strategy 3:
    lines = [
        re.sub(r"^\d+[\)\.\-]\s*", "", ln).strip()
        for ln in text.split("\n")
        if ln.strip() and ln.strip() not in ['[', ']', ',', '[]']
    ]
    if len(lines) >= num_questions:
        return lines[:num_questions]
    
    # Strategy 4:
    return ["No Answer"] * num_questions



#The original prompt is extended with an additional constraint-based instruction block to enforce that the model outputs a strictly formatted JSON list of answers.
def answer_questions(
    tokenizer,
    model,
    prompt_template,
    sentence,
    questions,
    debug=False,
    ex_id=None,
    which=None,
):
    questions_json = json.dumps(questions, ensure_ascii=False)

    base_prompt = (
        prompt_template
        .replace("{{sentence}}", sentence)
        .replace("{{questions}}", questions_json)
    )
    enhanced_prompt = f"""{base_prompt}

CRITICAL: You MUST return a valid JSON array with EXACTLY {len(questions)} string answers.

CORRECT format (flat array):
["answer 1", "answer 2", "answer 3"]

WRONG formats (DO NOT USE):
[["answer 1"], ["answer 2"]]  ← NO nested arrays
["answer"], []  ← NO empty elements
{{"answers": ["answer"]}}  ← NO objects

Rules:
- Return ONLY the JSON array, no other text
- Each answer must be a string
- If unknown, use "No Answer"
- No markdown, no backticks, no explanations

Your {len(questions)} answers:"""

    

    messages = [
        {"role": "system", "content": "You are a helpful assistant that returns only valid JSON arrays."},
        {"role": "user", "content": enhanced_prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=512,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[-1]
        response = outputs[0][prompt_len:]

        text = tokenizer.decode(
            response,
            skip_special_tokens=True
        ).strip()

        

        # CLEAN OUTPUT
        text_clean = text.strip()
        if text_clean.startswith("```"):
            lines = text_clean.split("\n")
            text_clean = "\n".join(lines[1:]) if len(lines) > 1 else ""
        if text_clean.endswith("```"):
            text_clean = text_clean.rsplit("```", 1)[0]

        text_clean = text_clean.strip()
        text_clean = text_clean.replace("'", '"')
        answers = parse_model_output(text_clean, len(questions), ex_id, which, debug)

        if len(answers) != len(questions):
            answers = answers[:len(questions)]
            answers += ["No Answer"] * (len(questions) - len(answers))

        answers = [
            str(a).strip() if a and str(a).strip() else "No Answer"
            for a in answers
        ]

    except Exception as e:
        if debug:
            import traceback
            traceback.print_exc()
        answers = ["No Answer"] * len(questions)

    return answers



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--prompt_key", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max_examples", type=int, default=0)
    args = parser.parse_args()

    prompt_template = load_prompt(args.prompt_path, args.prompt_key)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Prompt key: {args.prompt_key}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map="auto",
    )

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    processed_ids = set()
    if os.path.exists(args.output_path):
        with open(args.output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)["id"])
                except:
                    pass
        print(f"[INFO] Resume mode: {len(processed_ids)} IDs already processed")

    done = 0
    skipped = 0

    with open(args.input_path, "r", encoding="utf-8") as f_in, \
         open(args.output_path, "a", encoding="utf-8") as f_out:

        for line_idx, line in enumerate(f_in, start=1):
            data = json.loads(line)
            ex_id = data.get("id")

            if not ex_id or ex_id in processed_ids:
                skipped += 1
                continue

            questions = data.get("questions_src")
            if not questions:
                skipped += 1
                continue

            if isinstance(questions, str):
                try:
                    questions = json.loads(questions)
                except:
                    questions = [
                        q.strip()
                        for q in questions.split("\n")
                        if q.strip()
                    ]

            questions = [
                q.strip()
                for q in questions
                if isinstance(q, str) and q.strip()
            ]

            if not questions:
                skipped += 1
                continue

            src = data.get("src")
            bt = data.get("bt")
            if not src or not bt:
                skipped += 1
                continue

            print(f"[{done+1}] Processing ID: {ex_id} ({len(questions)} questions)")

            data["answers_src"] = answer_questions(
                tokenizer, model, prompt_template,
                src, questions,
                debug=args.debug, ex_id=ex_id, which="SRC"
            )

            data["answers_bt"] = answer_questions(
                tokenizer, model, prompt_template,
                bt, questions,
                debug=args.debug, ex_id=ex_id, which="BT"
            )

            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
            f_out.flush()
            done += 1

            if args.max_examples and done >= args.max_examples:
                print(f"[INFO] Reached max_examples={args.max_examples}")
                break

    print(f"\nCompleted: {done} processed, {skipped} skipped")


if __name__ == "__main__":
    main()
