import json
import argparse

#This script evaluates a binary classification task by comparing predicted decisions 
#(“accept” or “reject”) against human ground-truth labels stored in two JSONL files.
#It computes accuracy, builds a confusion matrix (TP, FP, TN, FN), and calculates precision, 
#recall, and F1 score to measure the model’s classification performance.

def evaluate_classifications(gold_file, pred_file):
    def load_jsonl_list(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return [json.loads(line.strip()) for line in f]

    gold_data = load_jsonl_list(gold_file)
    pred_data = load_jsonl_list(pred_file)

    if len(gold_data) != len(pred_data):
        print(f"Warning: Different number of entries in files! Gold: {len(gold_data)}, Pred: {len(pred_data)}")


    gold_dict = {entry.get("id"): entry for entry in gold_data if "id" in entry}
    pred_dict = {entry.get("id"): entry for entry in pred_data if "id" in entry}

    common_ids = set(gold_dict.keys()) & set(pred_dict.keys())

    if not common_ids:
        print("No common IDs found, comparing by position...")
        correct = 0
        total = min(len(gold_data), len(pred_data))
        
        for i in range(total):
            if gold_data[i].get("decision") == pred_data[i].get("decision"):
                correct += 1
        
        data_to_analyze = list(zip(gold_data[:total], pred_data[:total]))
    else:
        print(f"Comparing {len(common_ids)} entries by ID...")
        correct = 0
        total = len(common_ids)
        
        for entry_id in sorted(common_ids):
            if gold_dict[entry_id].get("decision") == pred_dict[entry_id].get("decision"):
                correct += 1
        
        data_to_analyze = [(gold_dict[i], pred_dict[i]) for i in common_ids]

    accuracy = (correct / total) * 100 if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Decision Accuracy: {accuracy:.2f}% ({correct}/{total} correct)")

    
    true_positives = sum(1 for g, p in data_to_analyze if g.get("decision") == "reject" and p.get("decision") == "reject")
    false_positives = sum(1 for g, p in data_to_analyze if g.get("decision") == "accept" and p.get("decision") == "reject")
    true_negatives = sum(1 for g, p in data_to_analyze if g.get("decision") == "accept" and p.get("decision") == "accept")
    false_negatives = sum(1 for g, p in data_to_analyze if g.get("decision") == "reject" and p.get("decision") == "accept")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Positives (reject→reject):  {true_positives}")
    print(f"  False Positives (accept→reject): {false_positives}")
    print(f"  True Negatives (accept→accept):  {true_negatives}")
    print(f"  False Negatives (reject→accept): {false_negatives}")
    

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nMetrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"{'='*60}\n")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {
            "tp": true_positives,
            "fp": false_positives,
            "tn": true_negatives,
            "fn": false_negatives
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold', type=str, required=True)
    parser.add_argument('--pred', type=str, required=True)
    
    args = parser.parse_args()
    
    evaluate_classifications(args.gold, args.pred)
