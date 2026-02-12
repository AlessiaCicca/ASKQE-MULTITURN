import torch
import json
import argparse
import re
from transformers import AutoTokenizer, AutoModelForCausalLM


model_id = "Qwen/Qwen2.5-7B-Instruct"


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




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--prompt_key", type=str, required=True)
    args = parser.parse_args()

    PROMPT_TEMPLATE = load_prompt(args.prompt_path, args.prompt_key) 

    if not os.path.isfile(args.input_path):
        print("[FATAL] Input file does not exist.")
        return

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"[DEBUG] Output directory ready: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("[DEBUG] pad_token was None → set to eos_token")

   
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    model.eval()


    with open(args.input_path, "r", encoding="utf-8") as f_in, \
         open(args.output_path, "a", encoding="utf-8") as f_out:

        for line_idx, line in enumerate(f_in, start=1):

            try:
                data = json.loads(line)
            except Exception as e:
                print(f"[ERROR] Line {line_idx} is not valid JSON: {e}")
                continue

          
            sent_id = data.get("id", f"line_{line_idx}")

            already_done = any(
                k.startswith("pert_mt") and args.perturbation_type in k
                for k in data.keys()
            )
            if already_done:
                print(f"[DEBUG] {sent_id} already perturbed → SKIP")
                continue

            sentence = data.get("mt")
            if not sentence:
                print(f"[WARNING] No 'mt' field found for {sent_id} → SKIP")
                continue

            mt_field = "mt"
            sentence = data.get(mt_field)

            if not sentence:
                print(f"[WARNING] Empty sentence in {mt_field} → SKIP")
                continue

            prompt = PROMPT_TEMPLATE.replace("{{sentence}}", sentence)

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]


            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                padding=True
            ).to(device)

            print(f"[DEBUG] input_ids shape: {inputs['input_ids'].shape}")
            
            # do_sample=True / temperature=0.25 /top_p=0.85 -> May make the dataset less artificial.
            with torch.no_grad():
                outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=200,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                temperature=0.25,
                top_p=0.85
                )

                prompt_len = inputs["input_ids"].shape[-1]
                response = outputs[0][prompt_len:]

                generated_text = tokenizer.decode(
                    response,
                    skip_special_tokens=True
                ).strip()

                out_field = "pert_mt"
                data[out_field] = generated_text

             
                print(f"[OK] {sent_id} | {out_field}")
                print(generated_text)
                print("-" * 80)

            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
