import json
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import argparse

def classify_with_gmm(input_jsonl, output_jsonl, metric, plot_output=None):
    """
    Classifica le entrate usando GMM su una metrica specificata.
    
    Args:
        input_jsonl: path del file di input
        output_jsonl: path del file di output
        metric: metrica da usare ('sbert_similarity', 'f1', 'chrf', 'bleu')
        plot_output: path opzionale per salvare il grafico
    """
    scores = []
    data_entries = []
    
    # Leggi i dati e calcola la media della metrica per ogni entrata
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if "scores" in entry and len(entry["scores"]) > 0:
                # Calcola la media della metrica da tutti gli scores
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
    
    # Addestra il modello GMM
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(scores)
    
    # Calcola le probabilità
    probabilities = gmm.predict_proba(scores)
    
    # Identifica quale componente è quella di bassa qualità (media più bassa)
    mean_low = np.min(gmm.means_)
    mean_high = np.max(gmm.means_)
    threshold = (mean_low + mean_high) / 2
    
    print(f"\n{'='*50}")
    print(f"Metrica: {metric}")
    print(f"{'='*50}")
    print(f"Mean Low: {mean_low:.4f}")
    print(f"Mean High: {mean_high:.4f}")
    print(f"Threshold for rejection: {threshold:.4f}")
    
    # Assegna le probabilità e decisioni
    for i, entry in enumerate(data_entries):
        # Probabilità di appartenere al cluster di bassa qualità
        entry["p_reject"] = probabilities[i, np.argmin(gmm.means_)]
        # Decisione basata sul threshold
        entry["decision"] = "reject" if entry[f"avg_{metric}"] < threshold else "accept"
    
    # Salva i risultati
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for entry in data_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    # Statistiche distribuzione decisioni
    accept_count = sum(1 for e in data_entries if e["decision"] == "accept")
    reject_count = sum(1 for e in data_entries if e["decision"] == "reject")
    print(f"\nDistribuzione decisioni:")
    print(f"  Accept: {accept_count} ({accept_count/len(data_entries)*100:.1f}%)")
    print(f"  Reject: {reject_count} ({reject_count/len(data_entries)*100:.1f}%)")
    print(f"\nProcessate {len(data_entries)} entrate")
    print(f"File salvato in: {output_jsonl}")
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classifica entrate usando GMM su metriche specifiche')
    parser.add_argument('--input', type=str, required=True, help='Path del file di input (.jsonl)')
    parser.add_argument('--output', type=str, required=True, help='Path del file di output (.jsonl)')
    parser.add_argument('--metric', type=str, required=True, 
                        choices=['sbert_similarity', 'f1', 'chrf', 'bleu'],
                        help='Metrica da utilizzare per la classificazione')
    
    args = parser.parse_args()
    
    classify_with_gmm(args.input, args.output, args.metric)
