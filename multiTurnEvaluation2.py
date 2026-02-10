import json
import nltk
import torch
from transformers import AutoTokenizer, AutoModel
import os

nltk.download("punkt")

# Setup SBERT model
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

def calculate_weighted_average(question_group):
    """
    Calcola le metriche con media ponderata 70/30.
    
    Formula per ogni metrica:
      risultato = (media × 0.3) + (pair3 × 0.7)
    
    dove:
      media = (pair1 + pair2) / 2
    """
    # Organizza i pair
    pairs = {}
    for item in question_group:
        pairs[item['pair']] = item
    
    # Estrai i tre pair
    pair1 = pairs['question_follow_up_question']
    pair2 = pairs['answer_src_follow_up_question']
    pair3 = pairs['follow_up_response_follow_up_response_bt']
    
    # Calcola per ogni metrica
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
    
    return result


def process_file(input_file: str, output_file: str):
    """
    Processa il file, calcola la media ponderata per ogni turno e salva i risultati.
    """
    # Leggi il file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    results = []
    
    for item in data:
        result_item = {
            'id': item['id'],
            'turns': []
        }
        
        for turn in item['turns']:
            turn_results = []
            for question_group in turn:
                # Calcola la media ponderata per ogni gruppo di risposte
                weighted_avg = calculate_weighted_average(question_group)
                turn_results.append(weighted_avg)
            
            result_item['turns'].append(turn_results)
        
        results.append(result_item)

    # Scrivi i risultati nel file di output
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"Risultati salvati in {output_file}")


if __name__ == "__main__":
    input_file = "/content/output_results/prova_results.jsonl"  # Il file di input
    output_file = "output.jsonl"  # Il file di output
    
    process_file(input_file, output_file)
