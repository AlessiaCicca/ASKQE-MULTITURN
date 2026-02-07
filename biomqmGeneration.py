import pandas as pd
import json
import sys

def process_translation(mtc_file, ref1_file, ref2_file, output_file):
    # Carica i file JSON (MTCarousel e due reference)
    with open(mtc_file) as f:
        mt_data = json.load(f)

    with open(ref1_file) as f:
        ref1_data = json.load(f)

    with open(ref2_file) as f:
        ref2_data = json.load(f)

    # Creare un DataFrame per ogni file per facilitare l'elaborazione
    df_mtc = pd.DataFrame(mt_data)
    df_ref1 = pd.DataFrame(ref1_data)
    df_ref2 = pd.DataFrame(ref2_data)

    # Ordina i dati per DOC_ID e SEG_ID
    df_mtc_sorted = df_mtc.sort_values(by=['DOC_ID', 'SEG_ID'])
    df_ref1_sorted = df_ref1.sort_values(by=['DOC_ID', 'SEG_ID'])
    df_ref2_sorted = df_ref2.sort_values(by=['DOC_ID', 'SEG_ID'])

    # Crea un dizionario per raccogliere i dati
    grouped_data = {}

    for _, row in df_mtc_sorted.iterrows():
        doc_id = row['DOC_ID']

        # Se il DOC_ID non esiste ancora, lo inizializzo
        if doc_id not in grouped_data:
            grouped_data[doc_id] = {
                "DOC_ID": doc_id,
                "source": [],
                "target": [],
                "SEG_ID": [],
                "filename": row['filename'],
                "MT_Engine": row['MT_Engine'],
                "source_locale": row['source_locale'],
                "target_locale": row['target_locale'],
                "source_errors": row['source_errors'],
                "target_errors": row['target_errors'],
                "Annotator_ID": row['Annotator_ID']
            }

        # Aggiungi le frasi e i SEG_ID separatamente
        # Evita duplicati nei SEG_ID
        if row['SEG_ID'] not in grouped_data[doc_id]["SEG_ID"]:
            grouped_data[doc_id]["source"].append(row['source'])
            grouped_data[doc_id]["target"].append(row['target'])
            grouped_data[doc_id]["SEG_ID"].append(row['SEG_ID'])

    # Ora raggruppiamo in blocchi da 3 frasi e assicuriamo che SEG_ID siano unici
    final_data = []

    for doc_id, doc_data in grouped_data.items():
        # Ordina i SEG_ID e le frasi associati a ogni SEG_ID
        sorted_indexes = sorted(range(len(doc_data["SEG_ID"])), key=lambda k: int(doc_data["SEG_ID"][k]))

        # Riordina le frasi e i SEG_ID in base al loro ordine numerico
        sorted_source = [doc_data["source"][i] for i in sorted_indexes]
        sorted_target = [doc_data["target"][i] for i in sorted_indexes]
        sorted_seg_id = [doc_data["SEG_ID"][i] for i in sorted_indexes]

        # Aggiungi i gruppi di frasi in blocchi da 3
        for i in range(0, len(sorted_source), 3):  # Ogni 3 frasi
            source_group = " ".join(sorted_source[i:i+3])
            target_group = " ".join(sorted_target[i:i+3])
            seg_id_group = sorted_seg_id[i:i+3]

            current_ref = []

            # Get target sentences from ref1 for the current seg_id_group
            ref1_segments = df_ref1_sorted[(df_ref1_sorted['DOC_ID'] == doc_id) & (df_ref1_sorted['SEG_ID'].isin(seg_id_group))]
            # Sort these segments by SEG_ID to ensure correct order before joining
            ref1_segments_sorted = ref1_segments.sort_values(by='SEG_ID')
            ref1_combined_target = " ".join(ref1_segments_sorted['target'].tolist())
            if ref1_combined_target:  # Only add if not empty
                current_ref.append(ref1_combined_target)

            # Get target sentences from ref2 for the current seg_id_group
            ref2_segments = df_ref2_sorted[(df_ref2_sorted['DOC_ID'] == doc_id) & (df_ref2_sorted['SEG_ID'].isin(seg_id_group))]
            # Sort these segments by SEG_ID to ensure correct order before joining
            ref2_segments_sorted = ref2_segments.sort_values(by='SEG_ID')
            ref2_combined_target = " ".join(ref2_segments_sorted['target'].tolist())
            if ref2_combined_target:  # Only add if not empty
                current_ref.append(ref2_combined_target)

            # Aggiungi il gruppo combinato con la lista di reference
            final_data.append({
                "DOC_ID": doc_data["DOC_ID"],
                "source": source_group,
                "target": target_group,
                "SEG_ID": seg_id_group,
                "ref": current_ref,  # Aggiungi le traduzioni di riferimento
                "filename": doc_data["filename"],
                "MT_Engine": doc_data["MT_Engine"],
                "source_locale": doc_data["source_locale"],
                "target_locale": doc_data["target_locale"],
                "source_errors": doc_data["source_errors"],
                "target_errors": doc_data["target_errors"],
                "Annotator_ID": doc_data["Annotator_ID"]
            })

    # Salva il risultato in un nuovo file JSON
    with open(output_file, 'w') as f:
        json.dump(final_data, f, indent=4)


if __name__ == "__main__":
    # Controlla se i percorsi sono stati passati come argomenti
    if len(sys.argv) != 5:
        print("Usage: python process_translation.py <mtcarousel_file> <reference_file1> <reference_file2> <output_file>")
        sys.exit(1)

    # Esegui la funzione con i percorsi dei file passati come argomenti
    mtcarousel_file = sys.argv[1]  # Il primo argomento è il file MTCarousel
    reference_file1 = sys.argv[2]  # Il secondo argomento è il primo file di reference
    reference_file2 = sys.argv[3]  # Il terzo argomento è il secondo file di reference
    output_file = sys.argv[4]  # Il quarto argomento è il file di output

    # Esegui il processo di traduzione
    process_translation(mtcarousel_file, reference_file1, reference_file2, output_file)
