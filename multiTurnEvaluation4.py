import json

def calculate_average(metrics1, metrics2):
    """
    Calcola la media tra le metriche di due set di dati.
    
    Args:
    - metrics1: Dati del primo set di metriche.
    - metrics2: Dati del secondo set di metriche.
    
    Returns:
    - dict: Dizionario con le metriche mediate.
    """
    averaged_metrics = {}
    for key in metrics1.keys():
        # Calcola la media dei valori tra i due set
        averaged_metrics[key] = (metrics1[key] + metrics2[key]) / 2
    return averaged_metrics

def combine_files(input_file1, input_file2, output_file):
    """
    Combina i dati da due file JSONL, calcolando la media delle metriche.
    
    Args:
    - input_file1: Primo file JSONL da combinare.
    - input_file2: Secondo file JSONL da combinare.
    - output_file: File JSONL di output con i dati combinati.
    """
    combined_data = []

    with open(input_file1, 'r', encoding='utf-8') as file1, open(input_file2, 'r', encoding='utf-8') as file2:
        data1 = [json.loads(line) for line in file1]
        data2 = [json.loads(line) for line in file2]
        
        # Assumiamo che i file abbiano gli stessi "id" e la stessa struttura
        for entry1, entry2 in zip(data1, data2):
            # Assicurati che gli ID corrispondano
            if entry1['id'] == entry2['id']:
                combined_entry = {
                    'id': entry1['id'],
                    'scores': []
                }
                
                # Per ogni coppia di metriche, calcola la media
                for score1, score2 in zip(entry1['scores'], entry2['scores']):
                    combined_score = calculate_average(score1, score2)
                    combined_entry['scores'].append(combined_score)
                
                combined_data.append(combined_entry)
            else:
                print(f"Warning: Mismatch between entries with id {entry1['id']} and {entry2['id']}")
    
    # Scrivi i dati combinati nel file di output
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in combined_data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    
    print(f"Risultati salvati in {output_file}")


if __name__ == "__main__":
    # Specifica i percorsi dei file di input e output
    input_file1 = '/content/output.jsonl'
    input_file2 = '/content/output_file.jsonl'
    output_file = 'combined_output.jsonl'
    
    combine_files(input_file1, input_file2, output_file)
