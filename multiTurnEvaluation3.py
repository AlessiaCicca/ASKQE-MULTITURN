import json
import sys

def calculate_average(metrics_list):
    """
    Calcola la media delle metriche per una domanda che ha valori provenienti da vari turni.
    
    Args:
    - metrics_list: Lista di dizionari con le metriche per una singola domanda nei vari turni.
    
    Returns:
    - avg_metrics: Dizionario con la media delle metriche per la domanda.
    """
    avg_metrics = {}
    
    # Calcola la media di ogni metrica
    for metric in metrics_list[0].keys():
        avg_metrics[metric] = sum(item[metric] for item in metrics_list) / len(metrics_list)
    
    return avg_metrics

def process_turns(turn_data):
    """
    Calcola la media per ogni metrica di ogni domanda nei turni.
    
    Args:
    - turn_data: Dati dei turni, ogni turno contiene una lista di domande.
    
    Returns:
    - results: Lista di dizionari con la media delle metriche per ogni domanda.
    """
    results = []
    
    # Itera su ogni domanda nei turni
    num_questions = len(turn_data[0])  # Assumiamo che ogni turno abbia lo stesso numero di domande
    
    for q_idx in range(num_questions):
        question_metrics = []
        
        # Raccoglie tutte le metriche della stessa domanda (indice q_idx) nei vari turni
        for turn in turn_data:
            question_metrics.append(turn[q_idx])  # Aggiungi la domanda corrispondente del turno
        
        # Calcola la media per questa domanda
        avg_metrics = calculate_average(question_metrics)
        
        # Aggiungi i risultati finali della domanda
        results.append(avg_metrics)
    
    return results


def process_file(input_file, output_file):
    """
    Carica il file JSONL, calcola la media per ogni domanda e salva i risultati nel file JSONL di output.
    
    Args:
    - input_file: Percorso del file JSONL di input.
    - output_file: Percorso del file JSONL di output.
    """
    combined_data = []

    # Leggi i dati dal file JSONL riga per riga
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)  # Carica ogni riga separatamente
                turn_data = data["turns"]
                
                # Calcola la media delle metriche per ogni domanda
                combined_results = process_turns(turn_data)
                
                # Prepara il formato richiesto per l'output
                output_data = {
                    "id": data["id"],
                    "scores": combined_results
                }
                
                # Aggiungi i risultati al file finale
                combined_data.append(output_data)
                
            except json.JSONDecodeError as e:
                print(f"Errore nel leggere la riga: {e}")
                continue

    # Salva i risultati nel file JSONL di output
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in combined_data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")  # Scrivi ogni oggetto JSON su una riga separata

    print(f"Risultati salvati in {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python calculate_metrics.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]  # Percorso del file JSONL di input
    output_file = sys.argv[2]  # Percorso del file JSONL di output
    
    process_file(input_file, output_file)
