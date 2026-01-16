import json
import argparse

def ucr_hallucination(answers_src, answers_bt, no_answer="No Answer"):
    assert len(answers_src) == len(answers_bt), \
        f"Length mismatch: src={len(answers_src)}, bt={len(answers_bt)}"

    hallucinated = 0
    for a_src, a_bt in zip(answers_src, answers_bt):
        if a_src.strip() == no_answer and a_bt.strip() != no_answer:
            # debug opzionale
            print("[HALLUCINATION]")
            print("SRC:", a_src)
            print("BT :", a_bt)
            print("-" * 40)
            hallucinated += 1

    return hallucinated / max(1, len(answers_src))


def load_data(path):
    print(f"[STEP] Loading dataset: {path}")
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            data = json.load(f)
        else:
            data = [json.loads(line) for line in f if line.strip()]

    print(f"[OK] Loaded {len(data)} samples")
    return data


def main():
    # =========================
    # ARGS
    # =========================
    parser = argparse.ArgumentParser(description="Compute UCR Hallucination score")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to JSON or JSONL file containing answers_src and answers_bt"
    )
    args = parser.parse_args()

    # =========================
    # LOAD DATA
    # =========================
    data = load_data(args.input_path)

    # =========================
    # AGGREGATE ANSWERS
    # =========================
    all_answers_src = []
    all_answers_bt = []

    for item in data:
        answers_src = item.get("answers_src")
        answers_bt = item.get("answers_bt")

        if not answers_src or not answers_bt:
            continue

        assert len(answers_src) == len(answers_bt), \
            f"Mismatch in item {item.get('id')}"

        all_answers_src.extend(answers_src)
        all_answers_bt.extend(answers_bt)

    # =========================
    # COMPUTE UCR
    # =========================
    ucr = ucr_hallucination(all_answers_src, all_answers_bt)
    print(f"\nUCR Hallucination Score: {ucr:.4f}")


if __name__ == "__main__":
    main()
