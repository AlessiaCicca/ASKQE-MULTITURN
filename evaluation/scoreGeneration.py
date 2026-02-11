import json
import nltk
import torch
from transformers import AutoTokenizer, AutoModel
from evaluation.utils import compare_answers
import os

nltk.download("punkt")

#This script generates a new JSONL file where each question pair (answers_src vs answers_bt) 
#is assigned evaluation metrics.
    
tokenizer_sbert = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model_sbert = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # Get token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def compute_sbert_similarity(ans_src, ans_bt):
    # Tokenize the source and back-translated answers
    encoded_src = tokenizer_sbert(ans_src, padding=True, truncation=True, return_tensors='pt')
    encoded_bt = tokenizer_sbert(ans_bt, padding=True, truncation=True, return_tensors='pt')

    # Get embeddings from SBERT model
    with torch.no_grad():
        src_output = model_sbert(**encoded_src)
        bt_output = model_sbert(**encoded_bt)

    # Compute mean pooled embeddings
    src_embed = mean_pooling(src_output, encoded_src['attention_mask'])
    bt_embed = mean_pooling(bt_output, encoded_bt['attention_mask'])

    # Calculate cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(src_embed, bt_embed, dim=1).item()
    return cos_sim

languages = ["italian", "french", "spanish"]
perturbations = ["alteration", "expansion_impact", "omission"]
data_dir="/content/ASKQE-MULTITURN/results"

for language in languages:
    for perturbation in perturbations:
        data_file = f"{data_dir}/{language}/perturb_{perturbation}/src_mt_perturb_bt_qg_qa.jsonl"
        
        results_list = []
        
        try:
            with open(data_file, "r", encoding="utf-8") as file:
                for line in file:
                    try:
                        data = json.loads(line)
                        
                        answers_src = data.get("answers_src", [])
                        answers_bt = data.get("answers_bt", [])
                        
                        if not isinstance(answers_src, list) or not isinstance(answers_bt, list):
                            continue
                        
                        if not answers_src or not answers_bt or len(answers_src) != len(answers_bt):
                            continue
                        
                        row_scores = []
                        for ans_src, ans_bt in zip(answers_src, answers_bt):
                            if not isinstance(ans_src, str) or not isinstance(ans_bt, str):
                                continue
                            if ans_src.strip() == "" or ans_bt.strip() == "":
                                continue
                            
                            # Calculate F1, EM, ChrF, and BLEU metrics
                            f1, EM, chrf, bleu = compare_answers(ans_src, ans_bt)

                            # Compute SBERT cosine similarity
                            sbert_similarity = compute_sbert_similarity(ans_src, ans_bt)

                            # Store all metrics
                            row_scores.append({
                                "f1": f1,
                                "em": EM,
                                "chrf": chrf,
                                "bleu": bleu,
                                "sbert_similarity": sbert_similarity  # Adding SBERT similarity
                            })
                        
                        # Save per-row result
                        if row_scores:  # Only if there are valid scores
                            row_data = {
                                "id": data.get("id", "unknown"),
                                "src": data.get("src", ""),
                                "bt": data.get("bt", ""),
                                "scores": row_scores
                            }
                            results_list.append(row_data)
                    
                    except json.JSONDecodeError as e:
                        print(f"Skipping a corrupted line due to JSONDecodeError: {e}")
                        continue
        
        except FileNotFoundError as e:
            print(f"File not found: {e}")
            continue
        
        # Create the output directory if it doesn't exist
        output_dir = f"src-bt_{language}"
        os.makedirs(output_dir, exist_ok=True)
        
        jsonl_output_file = f"{output_dir}/{perturbation}.jsonl"
        with open(jsonl_output_file, "w", encoding="utf-8") as jsonl_file:
            for row in results_list:
                jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        
        print(f"Saved results to {jsonl_output_file} ({len(results_list)} rows)")
