import json
import nltk
import torch
from transformers import AutoTokenizer, AutoModel
from utils import compare_answers
import os

#This script computes evaluation metrics for the entire multi-turn dataset 
#contained in the input JSONL file.

nltk.download("punkt")

tokenizer_sbert = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model_sbert = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # Get token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def compute_sbert_similarity(ans_src, ans_bt):
    encoded_src = tokenizer_sbert(ans_src, padding=True, truncation=True, return_tensors='pt')
    encoded_bt = tokenizer_sbert(ans_bt, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        src_output = model_sbert(**encoded_src)
        bt_output = model_sbert(**encoded_bt)

    src_embed = mean_pooling(src_output, encoded_src['attention_mask'])
    bt_embed = mean_pooling(bt_output, encoded_bt['attention_mask'])
    
    cos_sim = torch.nn.functional.cosine_similarity(src_embed, bt_embed, dim=1).item()
    return cos_sim


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--input_file", type=str, required=True)  
args = parser.parse_args()


results_list = []

try:
    with open(args.input_file, "r", encoding="utf-8") as file:
        for line in file:
            try:
                data = json.loads(line)

                aggregated_scores = {
                    "id": data["id"],
                    "turns": [] 
                }

             
                for turn in data["multiturn"].values():
                    turn_scores = []

                    for item in turn:
                        question_id = item["question_id"]
                        question = item["question"]
                        follow_up_question = item["follow_up_question"]
                        answer_src = item["answer_src"]
                        follow_up_response = item["follow_up_response"]
                        follow_up_response_bt = item["follow_up_response_bt"]

                        row_scores = []
                        if follow_up_response.strip() == "":
                            follow_up_response = "No Answer"
                        if follow_up_response_bt.strip() == "":
                            follow_up_response_bt = "No Answer"
                       
                        if answer_src.strip() == "":
                            answer_src = "No Answer"

                       
                        f1, EM, chrf, bleu = compare_answers(question, follow_up_question)
                        sbert_similarity = compute_sbert_similarity(question, follow_up_question)
                        row_scores.append({
                            "f1": f1,
                            "em": EM,
                            "chrf": chrf,
                            "bleu": bleu,
                            "sbert_similarity": sbert_similarity,
                            "pair": "question_follow_up_question"
                        })

                        f1, EM, chrf, bleu = compare_answers(answer_src, follow_up_question)
                        sbert_similarity = compute_sbert_similarity(answer_src, follow_up_question)
                        row_scores.append({
                            "f1": f1,
                            "em": EM,
                            "chrf": chrf,
                            "bleu": bleu,
                            "sbert_similarity": sbert_similarity,
                            "pair": "answer_src_follow_up_question"
                        })

                        f1, EM, chrf, bleu = compare_answers(follow_up_response, follow_up_response_bt)
                        sbert_similarity = compute_sbert_similarity(follow_up_response, follow_up_response_bt)
                        row_scores.append({
                            "f1": f1,
                            "em": EM,
                            "chrf": chrf,
                            "bleu": bleu,
                            "sbert_similarity": sbert_similarity,
                            "pair": "follow_up_response_follow_up_response_bt"
                        })

                        # Check the number of pairs for this turn
                        if len(row_scores) < 3:
                            print(f"Warning: Less than 3 pairs for id: {data['id']}, question_id: {question_id}")
                            print(f"Generated pairs: {len(row_scores)}")

                        # Add the row to the aggregated scores if there are valid results
                        if row_scores:  # Only if there are valid scores
                            turn_scores.append(row_scores)

                    # Add the turn scores to the main 'turns' list
                    if turn_scores:
                        aggregated_scores["turns"].append(turn_scores)

                # Add the aggregated scores for the current question_id to the results
                results_list.append(aggregated_scores)

            except json.JSONDecodeError as e:
                  print(f"Errore nel leggere la riga: {line}. Errore: {e}")
                  # Create a default entry for unknown errors with "Unknown" ID
                  aggregated_scores = {
                      "id": "Unknown",  # Explicitly set ID to "Unknown"
                      "turns": [{
                          "error": "JSONDecodeError", 
                          "line": line
                      }]
                  }
                  results_list.append(aggregated_scores)  # Add default entry with "Unknown"
                  continue

except FileNotFoundError as e:
    print(f"File not found: {e}")
    exit(1)

output_dir = os.path.dirname(args.output_file) 
os.makedirs(output_dir, exist_ok=True)  
with open(args.output_file, "w", encoding="utf-8") as jsonl_file:
    for row in results_list:
        jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Saved results to {args.output_file} ({len(results_list)} rows)")
