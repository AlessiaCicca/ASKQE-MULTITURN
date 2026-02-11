from datasets import load_dataset
import spacy
import json
import random
from tqdm import tqdm
import re
import argparse


#Downloads scientific abstracts from PubMed
#Cleans them (removes section headers like “Methods:” or “Results:”, fixes
#formatting, removes very short/noisy sentences) since by construction PubMed's abstracts
#have this structure.
#Splits them into sentences
#Groups every 3 consecutive sentences into a short paragraph
#Randomly selects 500 of these paragraphs
#Saves them into a JSONL file with an incremental ID


HEADER_PATTERN = re.compile(
    r"(?:(?<=^)|(?<=[\.\n]))\s*[A-Za-z][A-Za-z\s\-]{0,20}\s*:\s*",
    flags=re.IGNORECASE
)

ATTACHED_HEADER_PATTERN = re.compile(
    r"\.(results|methods|conclusions?|objectives?)\b",
    flags=re.IGNORECASE
)

def cleaning(text: str, nlp):
    text = re.sub(r"\s+", " ", text).strip()
    text = ATTACHED_HEADER_PATTERN.sub(r". \1", text)
    text = HEADER_PATTERN.sub("", text)
    doc = nlp(text)
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
    cleaned = []
    for s in sentences:
        t = s.lower().strip(" .:")
        if len(t.split()) <= 2 and t.isalpha():
            continue
        cleaned.append(s)

    return cleaned



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="pubmed_mt_askqe_2000.jsonl")
    args = parser.parse_args()


    DATASET_NAME = "ccdv/pubmed-summarization"
    SPLIT = "train"
    MAX_ABSTRACTS = 2000
    MAX_CONTEXTS = 500
    K = 3
    SEED = 42

    random.seed(SEED)


    nlp = spacy.load(
        "en_core_sci_sm",
        disable=["ner", "parser", "lemmatizer"]
    )
    nlp.add_pipe("sentencizer")
    
    dataset = load_dataset(DATASET_NAME, split=SPLIT)


    records = []
    for doc_id in tqdm(range(MAX_ABSTRACTS), desc="Processing abstracts"):
        abstract = dataset[doc_id].get("abstract", "")
        if not abstract:
            continue

        sentences = cleaning(abstract, nlp)

        if len(sentences) < K:
            continue

        for i in range(0, len(sentences), K):
            chunk = sentences[i:i + K]
            if len(chunk) < K:
                continue

            records.append({
                "context": " ".join(chunk)
            })

    if len(records) < MAX_CONTEXTS:
        raise ValueError(
            f"Only {len(records)} contexts available, less than {MAX_CONTEXTS}"
        )

    # We apply random sampling to improve the consistency and reliability of the selected contexts.
    records = random.sample(records, MAX_CONTEXTS)

    final_records = []
    for i, r in enumerate(records, start=1):
        final_records.append({
            "id": i,
            "src": r["context"]
        })


    with open(args.output, "w", encoding="utf-8") as fout:
        for r in final_records:
            fout.write(json.dumps(r) + "\n")

    print(f"Saved {len(final_records)} contexts to {args.output}")


if __name__ == "__main__":
    main()

