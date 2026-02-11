import json
import nltk
import argparse
import csv
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F


#This script computes the average SBERT cosine similarity between answers_src and
#answers_bt across all language–perturbation combinations

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


nltk.download("punkt")

parser = argparse.ArgumentParser()
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--data_dir", type=str, default="../../results")
args = parser.parse_args()


languages = ["italian", "french", "spanish"]
perturbations = ["alteration", "expansion_impact", "omission"]


tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')


with open(args.output_file, mode="w", newline="", encoding="utf-8") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["language", "perturbation", "cosine_similarity", "num_comparison"])

    for language in languages:
        for perturbation in perturbations:
            print("Language: ", language)
            print("Perturbation: ", perturbation)

            data_file = f"{args.data_dir}/{language}/perturb_{perturbation}/src_mt_perturb_bt_qg_qa.jsonl"

            total_cosine_similarity = 0
            num_comparisons = 0

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
                            
                            for ans_src, ans_bt in zip(answers_src, answers_bt):
                                if not isinstance(ans_src, str) or not isinstance(ans_bt, str):
                                    continue
                                if ans_src.strip() == "" or ans_bt.strip() == "":
                                    continue

                                encoded_src = tokenizer(ans_src, padding=True, truncation=True, return_tensors='pt')
                                encoded_bt = tokenizer(ans_bt, padding=True, truncation=True, return_tensors='pt')

                                with torch.no_grad():
                                    src_output = model(**encoded_src)
                                    bt_output = model(**encoded_bt)

                                src_embed = mean_pooling(src_output, encoded_src['attention_mask'])
                                src_embeds = F.normalize(src_embed, p=2, dim=1)

                                bt_embed = mean_pooling(bt_output, encoded_bt['attention_mask'])
                                bt_embeds = F.normalize(bt_embed, p=2, dim=1)

                                cos_sim = F.cosine_similarity(src_embeds, bt_embeds, dim=1).mean().item()
                                total_cosine_similarity += cos_sim
                                num_comparisons += 1

                        except json.JSONDecodeError as e:
                            print(f"Skipping a corrupted line due to JSONDecodeError: {e}")
                            continue

            except FileNotFoundError as e:
                print(f"File not found: {e}")
                continue

            if num_comparisons > 0:
                avg_cosine_similarity = total_cosine_similarity / num_comparisons

                print("-" * 80)
                print("Average Scores:")
                print(f"Num comparisons: {num_comparisons}")
                print(f"Cosine Similarity: {avg_cosine_similarity:.3f}")
                print("=" * 80)

                csv_writer.writerow([language, perturbation, avg_cosine_similarity, num_comparisons])

            else:
                print("No valid comparisons found in the JSONL file.")

