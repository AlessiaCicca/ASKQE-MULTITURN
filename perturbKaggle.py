import os
import torch
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from google.colab import files

# =========================
# CONFIGURAZIONE AMBIENTE
# =========================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Disabilitare i messaggi di warning di TensorFlow
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disabilitare il parallelismo dei tokenizzatori

# =========================
# CARICAMENTO DEL MODELLO
# =========================
model_id = "Qwen/Qwen2.5-7B-Instruct"

def load_prompt(prompt_path: str, prompt_key: str) -> str:
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    if prompt_key not in prompts:
        raise KeyError(f"Prompt key '{prompt_key}' not found. Available keys: {list(prompts.keys())}")
    prompt = prompts[prompt_key]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Prompt '{prompt_key}' is empty or invalid")
    return prompt

def main():
    # Impostazione degli argomenti
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--prompt_key", type=str, required=True)
    args = parser.parse_args()

    # Carica il template del prompt
    prompt_template = load_prompt(args.prompt_path, args.prompt_key)

    # Verifica se il file di input esiste
    if not os.path.isfile(args.input_path):
        print("[FATAL] Input file does not exist.")
        return

    # Crea la cartella di output se non esiste
    output_dir = os.path.dirname(args.output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Imposta il dispositivo (GPU o CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Carica il tokenizer e il modello
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

    # Processa il file riga per riga
    with open(args.input_path, "r", encoding="utf-8") as f_in, open(args.output_path, "a", encoding="utf-8") as f_out:
        line_counter = 0
        processed_data = []

        # Leggi il file di input riga per riga
        for line_idx, line in enumerate(f_in, start=1):
            try:
                data = json.loads(line)
            except Exception as e:
                print(f"[ERROR] Line {line_idx} is not valid JSON: {e}")
                continue

            sent_id = data.get("id", f"line_{line_idx}")
            sentence = data.get("mt", "")

            if not sentence:
                print(f"[WARNING] Empty or missing sentence in {sent_id}. Skipping.")
                continue

            # Costruisci il prompt
            prompt = prompt_template.replace("{{sentence}}", sentence)

            # Tokenizzazione
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]

            inputs = tokenizer(messages, return_tensors="pt", padding=True, truncation=True).to(device)

            # Aggiungi un debug per verificare la forma
            print(f"[DEBUG] input_ids shape: {inputs['input_ids'].shape}")

            # Generazione del testo
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

                # Verifica la forma dell'output
                output_ids = outputs[0]
                print(f"[DEBUG] Output shape: {output_ids.shape}")

                # Estrai la risposta correttamente
                prompt_len = inputs["input_ids"].shape[-1]
                response = output_ids[:, prompt_len:]
                generated_text = tokenizer.decode(response[0], skip_special_tokens=True).strip()

            # Aggiungi il risultato al dizionario dei dati
            data["pert_mt"] = generated_text
            processed_data.append(data)

            line_counter += 1
            if line_counter % 100 == 0:  # Salva ogni 100 righe
                print(f"[INFO] Saving results after processing {line_counter} lines...")
                for item in processed_data:
                    f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                processed_data = []  # Reset della lista per non usare troppa memoria

        # Salva i dati rimanenti
        if processed_data:
            print(f"[INFO] Saving remaining results...")
            for item in processed_data:
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Se eseguito in Google Colab, scarica il file
    if "google.colab" in str(get_ipython()):
        print(f"[INFO] Downloading the result file {args.output_path} to your local system.")
        files.download(args.output_path)

if __name__ == "__main__":
    main()
