import json
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import argparse

#This script performs unsupervised binary classification of translation quality using a 
#Gaussian Mixture Model (GMM) based on a chosen evaluation metric.
#First, it reads a JSONL file where each entry contains a list of "scores". For each entry, 
#it computes the average value of the selected metric and stores it as avg_<metric>. Then, it 
#fits a 2-component GMM to these averaged scores, assuming the data naturally splits into two 
#clusters (high-quality vs low-quality translations).
#The midpoint between the two Gaussian means is used as a threshold: entries below it are 
#labeled "reject" and those above it "accept".

def classify_with_gmm(input_jsonl, output_jsonl, metric):
    scores = []
    data_entries = []
  
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if "scores" in entry and len(entry["scores"]) > 0:
                metric_scores = [s[metric] for s in entry["scores"] if metric in s]
                if metric_scores:
                    avg_metric = np.mean(metric_scores)
                    entry[f"avg_{metric}"] = avg_metric
                    scores.append(avg_metric)
                else:
                    entry[f"avg_{metric}"] = 0.0
                    scores.append(0.0)
            else:
                entry[f"avg_{metric}"] = 0.0
                scores.append(0.0)
            data_entries.append(entry)
    
    scores = np.array(scores).reshape(-1, 1)
   
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(scores)
    
    probabilities = gmm.predict_proba(scores)
    
    mean_low = np.min(gmm.means_)
    mean_high = np.max(gmm.means_)
    threshold = (mean_low + mean_high) / 2
    
    print(f"\n{'='*50}")
    print(f"Metrics: {metric}")
    print(f"{'='*50}")
    print(f"Mean Low: {mean_low:.4f}")
    print(f"Mean High: {mean_high:.4f}")
    print(f"Threshold for rejection: {threshold:.4f}")
    
    for i, entry in enumerate(data_entries):
        entry["p_reject"] = probabilities[i, np.argmin(gmm.means_)]
        entry["decision"] = "reject" if entry[f"avg_{metric}"] < threshold else "accept"

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for entry in data_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    

    accept_count = sum(1 for e in data_entries if e["decision"] == "accept")
    reject_count = sum(1 for e in data_entries if e["decision"] == "reject")
    print(f"  Accept: {accept_count} ({accept_count/len(data_entries)*100:.1f}%)")
    print(f"  Reject: {reject_count} ({reject_count/len(data_entries)*100:.1f}%)")
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--metric', type=str, required=True, 
                        choices=['sbert_similarity', 'f1', 'chrf', 'bleu'])
    
    args = parser.parse_args()
    
    classify_with_gmm(args.input, args.output, args.metric)
