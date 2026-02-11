import json
import nltk
import argparse
import csv
from utils import compare_answers


#This script computes the average F1, EM, BLEU and CHRF between answers_src and
#answers_bt across all language–perturbation combinations

nltk.download("punkt")

parser = argparse.ArgumentParser()
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--data_dir", type=str, default="../../results")
args = parser.parse_args()

languages = ["italian", "french", "spanish"]
perturbations = ["alteration", "expansion_impact", "omission"]

with open(args.output_file, mode="w", newline="", encoding="utf-8") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["language", "perturbation", "f1", "em", "chrf", "bleu", "num_comparison"])

    for language in languages:
        for perturbation in perturbations:
            print("Language: ", language)
            print("Perturbation: ", perturbation)

            data_file = f"{args.data_dir}/{language}/perturb_{perturbation}/src_mt_perturb_bt_qg_qa.jsonl"

            total_f1 = 0
            total_em = 0
            total_chrf = 0
            total_bleu = 0
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

                                f1, EM, chrf, bleu = compare_answers(ans_src, ans_bt)
                                total_f1 += f1
                                total_em += EM
                                total_chrf += chrf
                                total_bleu += bleu
                                num_comparisons += 1

                        except json.JSONDecodeError as e:
                            print(f"Skipping a corrupted line due to JSONDecodeError: {e}")
                            continue

            except FileNotFoundError as e:
                print(f"File not found: {e}")
                continue

            if num_comparisons > 0:
                avg_f1 = total_f1 / num_comparisons
                avg_em = total_em / num_comparisons
                avg_chrf = total_chrf / num_comparisons
                avg_bleu = total_bleu / num_comparisons

                print("-" * 80)
                print("Average Scores:")
                print(f"Num comparisons: {num_comparisons}")
                print(f"F1: {avg_f1:.3f}")
                print(f"EM: {avg_em:.3f}")
                print(f"ChrF: {avg_chrf:.3f}")
                print(f"BLEU: {avg_bleu:.3f}")
                print("=" * 80)

                csv_writer.writerow([language, perturbation, avg_f1, avg_em, avg_chrf, avg_bleu, num_comparisons])

            else:
                print("No valid comparisons found in the JSONL file.")

