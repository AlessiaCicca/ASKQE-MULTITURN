import json
import nltk
import torch
from transformers import AutoTokenizer, AutoModel
import argparse

nltk.download("punkt")

# Setup SBERT model
tokenizer_sbert = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model_sbert = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def compute_sbert_similarity(ans_src, ans_bt):
    print(f"Calcolando similitudine SBERT per: {ans_src[:50]}... vs {ans_bt[:50]}...")  # Debug
    encoded_src = tokenizer_sbert(ans_src, padding=True, truncation=True, return_tensors='pt')
    encoded_bt = tokenizer_sbert(ans_bt, padding=True, truncation=True, return_tensors='pt')
    
    with torch.no_grad():
        src_output = model_sbert(**encoded_src)
        bt_output = model_sbert(**encoded_bt)
    
    src_embed = mean_pooling(src_output, encoded_src['attention_mask'])
    bt_embed = mean_pooling(bt_output, encoded_bt['attention_mask'])
    
    cos_sim = torch.nn.functional.cosine_similarity(src_embed, bt_embed, dim=1).item()
    print(f"Similitudine coseno: {cos_sim}")  # Debug
    return cos_sim

# ========== STEP 1: Weighted Average (70/30) - SOLO FILE 1 ==========

def calculate_weighted_average(question_group):
    """
    Calcola le metriche con media ponderata 70/30.
    Formula: risultato = (media × 0.3) + (pair3 × 0.7)
    """
    print(f"Calcolando media ponderata per group di {len(question_group)} pair")  # Debug
    
    pairs = {}
    for item in question_group:
        pairs[item['pair']] = item
    
    # Verifica che ci siano esattamente 3 pair
    if len(pairs) != 3:
        print(f"ATTENZIONE: Trovati {len(pairs)} pair invece di 3")  # Debug
        print(f"Pair disponibili: {list(pairs.keys())}")  # Debug
        raise ValueError(f"Numero di pair errato: {len(pairs)}")
    
    # Estrai i tre pair - usa i nomi esatti dal tuo file
    pair_names = list(pairs.keys())
    pair1 = pairs[pair_names[0]]  # question_follow_up_question
    pair2 = pairs[pair_names[1]]  # answer_src_follow_up_question
    pair3 = pairs[pair_names[2]]  # follow_up_response_follow_up_response_bt
    
    result = {}
    
    # F1
    avg_f1 = (pair1['f1'] + pair2['f1']) / 2
    result['f1'] = (avg_f1 * 0.3) + (pair3['f1'] * 0.7)
    
    # ChrF
    avg_chrf = (pair1['chrf'] + pair2['chrf']) / 2
    result['chrf'] = (avg_chrf * 0.3) + (pair3['chrf'] * 0.7)
    
    # BLEU
    avg_bleu = (pair1['bleu'] + pair2['bleu']) / 2
    result['bleu'] = (avg_bleu * 0.3) + (pair3['bleu'] * 0.7)
    
    # SBERT Similarity
    avg_sbert = (pair1['sbert_similarity'] + pair2['sbert_similarity']) / 2
    result['sbert_similarity'] = (avg_sbert * 0.3) + (pair3['sbert_similarity'] * 0.7)
    
    # EM (exact match) - usa AND logico
    result['em'] = pair1['em'] and pair2['em'] and pair3['em']
    
    print(f"Media ponderata calcolata: {result}")  # Debug
    return result

def process_weighted_average(input_file, output_file):
    """STEP 1: Calcola la media ponderata per ogni turno - SOLO FILE 1"""
    print(f"Caricamento dati da: {input_file}")  # Debug
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    results = []
    
    for item in data:
        result_item = {
            'id': item['id'],
            'turns': []
        }
        
        for turn_idx, turn in enumerate(item['turns']):
            print(f"Elaborando turno {turn_idx} per ID {item['id']}")  # Debug
            turn_results = []
            for question_idx, question_group in enumerate(turn):
                try:
                    weighted_avg = calculate_weighted_average(question_group)
                    turn_results.append(weighted_avg)
                except Exception as e:
                    print(f"\nErrore in id={item['id']}, turn={turn_idx}, question={question_idx}")
                    print(f"Question group: {question_group}")
                    raise e
            
            result_item['turns'].append(turn_results)
        
        results.append(result_item)
    
    print(f"Scrivendo il risultato su: {output_file}")  # Debug
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"✓ Step 1 completato: {output_file}")
    return output_file

# ========== STEP 2: Average Across Turns - SOLO FILE 1 ==========

def calculate_average(metrics_list):
    """Calcola la media delle metriche per una domanda nei vari turni"""
    print(f"Calcolando la media delle metriche per {len(metrics_list)} turni")  # Debug
    avg_metrics = {}
    
    for metric in metrics_list[0].keys():
        avg_metrics[metric] = sum(item[metric] for item in metrics_list) / len(metrics_list)
    
    return avg_metrics

def process_turns(turn_data):
    """Calcola la media per ogni metrica di ogni domanda nei turni"""
    print(f"Elaborando {len(turn_data)} turni per calcolare la media")  # Debug
    results = []
    num_questions = len(turn_data[0])
    
    for q_idx in range(num_questions):
        question_metrics = []
        
        for turn in turn_data:
            question_metrics.append(turn[q_idx])
        
        avg_metrics = calculate_average(question_metrics)
        results.append(avg_metrics)
    
    return results

def process_turn_average(input_file, output_file):
    """STEP 2: Calcola la media attraverso i turni - SOLO FILE 1"""
    print(f"Caricamento dati da: {input_file}")  # Debug
    combined_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                turn_data = data["turns"]
                
                combined_results = process_turns(turn_data)
                
                output_data = {
                    "id": data["id"],
                    "scores": combined_results
                }
                
                combined_data.append(output_data)
                
            except json.JSONDecodeError as e:
                print(f"Errore nel leggere la riga: {e}")
                continue
    
    print(f"Scrivendo il risultato su: {output_file}")  # Debug
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in combined_data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    
    print(f"✓ Step 2 completato: {output_file}")
    return output_file

# ========== STEP 3: Combine Two Files ==========

def calculate_file_average(metrics1, metrics2):
    """Calcola la media tra le metriche di due set di dati"""
    print(f"Calcolando la media tra i set di dati: {metrics1} vs {metrics2}")  # Debug
    averaged_metrics = {}
    for key in metrics1.keys():
        averaged_metrics[key] = (metrics1[key] + metrics2[key]) / 2
    return averaged_metrics

def combine_files(input_file1, input_file2, output_file):
    """STEP 3: Combina due file calcolando la media delle metriche"""
    print(f"Caricamento dati da FILE 1: {input_file1} e FILE 2: {input_file2}")  # Debug
    combined_data = []
    
    with open(input_file1, 'r', encoding='utf-8') as file1, open(input_file2, 'r', encoding='utf-8') as file2:
        data1 = [json.loads(line) for line in file1]
        data2 = [json.loads(line) for line in file2]
        
        for entry1, entry2 in zip(data1, data2):
            if entry1['id'] == entry2['id']:
                combined_entry = {
                    'id': entry1['id'],
                    'scores': []
                }
                
                for score1, score2 in zip(entry1['scores'], entry2['scores']):
                    combined_score = calculate_file_average(score1, score2)
                    combined_entry['scores'].append(combined_score)
                
                combined_data.append(combined_entry)
            else:
                print(f"Warning: Mismatch tra id {entry1['id']} e {entry2['id']}")  # Debug
    
    print(f"Scrivendo il risultato su: {output_file}")  # Debug
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in combined_data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    
    print(f"✓ Step 3 completato: {output_file}")
    return output_file

# ========== MAIN PIPELINE ==========

def main(input_file1, input_file2, output_file):
    """
    Pipeline completa:
    1. Applica weighted average (70/30) al file 1
    2. Calcola la media attraverso i turni SOLO per il file 1
    3. Combina il file 1 elaborato con il file 2 (non elaborato)
    """
    print("=" * 60)
    print("PIPELINE DI ELABORAZIONE")
    print("=" * 60)
    
    # Step 1: Weighted average sul file 1
    print("\n[1/3] Calcolo media ponderata (70/30) su FILE 1...")
    temp_weighted = "temp_weighted.jsonl"
    process_weighted_average(input_file1, temp_weighted)
    
    # Step 2: Average across turns SOLO sul file 1
    print("\n[2/3] Calcolo media attraverso i turni su FILE 1...")
    temp_avg1 = "temp_avg1.jsonl"
    process_turn_average(temp_weighted, temp_avg1)
    
    # Step 3: Combine file 1 elaborato con file 2 non elaborato
    print("\n[3/3] Combinazione FILE 1 (elaborato) con FILE 2 (originale)...")
    combine_files(temp_avg1, input_file2, output_file)
    
    print("\n" + "=" * 60)
    print(f"✓ COMPLETATO! Risultato finale: {output_file}")
    print("=" * 60)
    print("\nRiepilogo:")
    print(f"  - FILE 1: Media ponderata 70/30 + Media tra turni")
    print(f"  - FILE 2: Usato così com'è")
    print(f"  - Output: Media tra FILE 1 elaborato e FILE 2")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Pipeline di elaborazione metriche: applica media ponderata e media tra turni al FILE 1, poi combina con FILE 2.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''...'''
    )
    
    parser.add_argument(
        '--file1', '-f1',
        type=str,
        required=True,
        help='Primo file JSONL di input (verrà elaborato con media ponderata e media tra turni)'
    )
    
    parser.add_argument(
        '--file2', '-f2',
        type=str,
        required=True,
        help='Secondo file JSONL di input (verrà usato direttamente senza elaborazioni)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='final_output.jsonl',
        help='File JSONL di output (default: final_output.jsonl)'
    )
    
    args = parser.parse_args()
    
    main(args.file1, args.file2, args.output)
