import json
import argparse


def compute_metrics(input_path, no_answer="No Answer"):
    total_questions = 0
    yes_count = 0
    no_count = 0
    no_answer_count = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)

            questions = data.get("contrastive_questions", [])
            answers = data.get("contrastive_answers_src", [])

            if not questions or not answers:
                continue

            # sicurezza: allinea lunghezze
            n = min(len(questions), len(answers))
            questions = questions[:n]
            answers = answers[:n]

            for a in answers:
                total_questions += 1
                a_norm = a.strip().lower()

                if a_norm == "yes":
                    yes_count += 1
                elif a_norm == "no":
                    no_count += 1
                else:
                    no_answer_count += 1

    # ===== METRICHE =====
    chr_rate = no_count / total_questions if total_questions > 0 else 0.0

    verified_den = yes_count + no_count
    chr_verified = no_count / verified_den if verified_den > 0 else 0.0

    chr_strict = (no_count + no_answer_count) / total_questions if total_questions > 0 else 0.0

    return {
        "total_questions": total_questions,
        "yes": yes_count,
        "no": no_count,
        "no_answer": no_answer_count,
        "CHR": chr_rate,
        "CHR_verified": chr_verified,
        "CHR_strict": chr_strict
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to JSONL with contrastive answers")
    args = parser.parse_args()

    metrics = compute_metrics(args.input_path)

    print("\n====== CONTRASTIVE HALLUCINATION METRICS ======")
    print(f"Total contrastive questions : {metrics['total_questions']}")
    print(f"Yes                        : {metrics['yes']}")
    print(f"No                         : {metrics['no']}")
    print(f"No Answer                  : {metrics['no_answer']}")
    print("---------------------------------------------")
    print(f"CHR (No / All)             : {metrics['CHR']:.4f}")
    print(f"CHR_verified (No / Y+N)    : {metrics['CHR_verified']:.4f}")
    print(f"CHR_strict ((N+NA) / All)  : {metrics['CHR_strict']:.4f}")
    print("=============================================\n")


if __name__ == "__main__":
    main()
