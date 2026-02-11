import pandas as pd
import json
import sys


# This script restructures segment-level machine translation data into grouped 3-segment document-level 
# examples with aligned references, producing a JSONL file suitable for contextual MT evaluation.


def process_translation(mtc_file, ref1_file, ref2_file, output_file):
    # =========================
    # LOAD JSON FILES
    # =========================
    with open(mtc_file, encoding="utf-8") as f:
        mt_data = json.load(f)

    with open(ref1_file, encoding="utf-8") as f:
        ref1_data = json.load(f)

    with open(ref2_file, encoding="utf-8") as f:
        ref2_data = json.load(f)

    # =========================
    # CREATE DATAFRAMES
    # =========================
    df_mtc = pd.DataFrame(mt_data)
    df_ref1 = pd.DataFrame(ref1_data)
    df_ref2 = pd.DataFrame(ref2_data)

    # =========================
    # SORT BY DOC_ID & SEG_ID
    # =========================
    df_mtc_sorted = df_mtc.sort_values(by=["DOC_ID", "SEG_ID"])
    df_ref1_sorted = df_ref1.sort_values(by=["DOC_ID", "SEG_ID"])
    df_ref2_sorted = df_ref2.sort_values(by=["DOC_ID", "SEG_ID"])

    # =========================
    # GROUP BY DOC_ID
    # =========================
    grouped_data = {}

    for _, row in df_mtc_sorted.iterrows():
        doc_id = row["DOC_ID"]

        if doc_id not in grouped_data:
            grouped_data[doc_id] = {
                "DOC_ID": doc_id,
                "source": [],
                "target": [],
                "SEG_ID": [],
                "filename": row["filename"],
                "MT_Engine": row["MT_Engine"],
                "source_locale": row["source_locale"],
                "target_locale": row["target_locale"],
                "source_errors": row["source_errors"],
                "target_errors": row["target_errors"],
                "Annotator_ID": row["Annotator_ID"]
            }

        if row["SEG_ID"] not in grouped_data[doc_id]["SEG_ID"]:
            grouped_data[doc_id]["source"].append(row["source"])
            grouped_data[doc_id]["target"].append(row["target"])
            grouped_data[doc_id]["SEG_ID"].append(row["SEG_ID"])

    # =========================
    # BUILD FINAL DATA (GROUPS OF 3)
    # =========================
    final_data = []
    global_id = 1  # ⬅ ID incrementale globale

    for doc_id, doc_data in grouped_data.items():
        sorted_indexes = sorted(
            range(len(doc_data["SEG_ID"])),
            key=lambda k: int(doc_data["SEG_ID"][k])
        )

        sorted_source = [doc_data["source"][i] for i in sorted_indexes]
        sorted_target = [doc_data["target"][i] for i in sorted_indexes]
        sorted_seg_id = [doc_data["SEG_ID"][i] for i in sorted_indexes]

        for i in range(0, len(sorted_source), 3):
            source_group = " ".join(sorted_source[i:i + 3])
            target_group = " ".join(sorted_target[i:i + 3])
            seg_id_group = sorted_seg_id[i:i + 3]

            current_ref = []

            # ---- Reference 1
            ref1_segments = df_ref1_sorted[
                (df_ref1_sorted["DOC_ID"] == doc_id) &
                (df_ref1_sorted["SEG_ID"].isin(seg_id_group))
            ].sort_values(by="SEG_ID")

            ref1_combined = " ".join(ref1_segments["target"].tolist())
            if ref1_combined:
                current_ref.append(ref1_combined)

            # ---- Reference 2
            ref2_segments = df_ref2_sorted[
                (df_ref2_sorted["DOC_ID"] == doc_id) &
                (df_ref2_sorted["SEG_ID"].isin(seg_id_group))
            ].sort_values(by="SEG_ID")

            ref2_combined = " ".join(ref2_segments["target"].tolist())
            if ref2_combined:
                current_ref.append(ref2_combined)

            final_data.append({
                "id": global_id,                 # ⬅ ID aggiunto
                "DOC_ID": doc_data["DOC_ID"],
                "src": source_group,
                "pert_mt": target_group,
                "SEG_ID": seg_id_group,
                "ref": current_ref,
                "filename": doc_data["filename"],
                "MT_Engine": doc_data["MT_Engine"],
                "source_locale": doc_data["source_locale"],
                "target_locale": doc_data["target_locale"],
                "source_errors": doc_data["source_errors"],
                "target_errors": doc_data["target_errors"],
                "Annotator_ID": doc_data["Annotator_ID"]
            })

            global_id += 1  # ⬅ incremento

    # =========================
    # SAVE OUTPUT
    # =========================
    with open(output_file, "w", encoding="utf-8") as f:
        for record in final_data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(
            "Usage: python process_translation.py "
            "<mtcarousel_file> <reference_file1> <reference_file2> <output_file>"
        )
        sys.exit(1)

    mtcarousel_file = sys.argv[1]
    reference_file1 = sys.argv[2]
    reference_file2 = sys.argv[3]
    output_file = sys.argv[4]

    process_translation(
        mtcarousel_file,
        reference_file1,
        reference_file2,
        output_file
    )

