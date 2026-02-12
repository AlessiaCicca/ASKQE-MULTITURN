import json
import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model used for QA and follow-up generation
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"



# LOAD PROMPTS
def load_prompt(path, key):
    with open(path, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    if key not in prompts:
        raise KeyError(f"Prompt key '{key}' not found")
    return prompts[key]

# Cleans model output to ensure it is returned as a plain string.
# Handles cases where the model returns JSON arrays or quoted strings.
def clean_response(response):
    response_str = str(response).strip()
    
    # Remove surrounding brackets if the model outputs a JSON list
    if response_str.startswith('[') and response_str.endswith(']'):
        response_str = response_str[1:-1].strip()
        if response_str.startswith('"') and response_str.endswith('"'):
            response_str = response_str[1:-1]
    try:
        parsed = json.loads(response_str)
        if isinstance(parsed, list) and len(parsed) > 0:
            return str(parsed[0])
        return str(parsed)
    except:
        return response_str



# GENERATE QA RESPONSE -> Generates an answer to a given question based on the provided text.
def generate_response(
    tokenizer,
    model,
    qa_prompt,
    text,
    question
):

    prompt = (
        qa_prompt
        .replace("{{sentence}}", text)
        .replace("{{questions}}", json.dumps([question], ensure_ascii=False))
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    # Deterministic generation (greedy decoding)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=128,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip()

    return response



# GENERATE FOLLOW-UP QUESTION ->    
#Generates a follow-up question based on:
#    - original text (src)
#    - previous question
#    - previous answer
def generate_followup_question(
    tokenizer,
    model,
    followup_prompt,
    text,
    prev_question,
    prev_answer
):
    prompt = (
        followup_prompt
        .replace("{{text}}", text)
        .replace("{{prev_question}}", prev_question)
        .replace("{{prev_answer}}", str(prev_answer))
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=64,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )
    q = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip()

    return q


# MAIN PIPELINE
  """
    Multi-turn ASKQE pipeline:

    For each input example:
    - Takes original questions and answers
    - Generates follow-up questions providing also the original text
    - Generates answers for SRC and BT
    - Iterates for multiple turns
    - Stores structured multi-turn QA results
    """

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--prompt_path", required=True)
    parser.add_argument("--max_turns", type=int, default=3)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Load prompt templates
    qa_prompt = load_prompt(args.prompt_path, "qa_prompt")
    followup_prompt = load_prompt(args.prompt_path, "followup_prompt")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    with open(args.input_path, "r", encoding="utf-8") as f_in, \
         open(args.output_path, "w", encoding="utf-8") as f_out:

        for line in f_in:
            data = json.loads(line)

            src = data.get("src")
            bt = data.get("bt")
            questions = data.get("questions_src")
            answers_src = data.get("answers_src")
            entry_id = data.get("id")

            # Skip incomplete entries
            if not src or not bt or not questions or not answers_src or not entry_id:
                continue

            multiturn = {}

            current_questions = questions
            current_answers = answers_src
            turn_counter = 1
            question_counter = 1

            # Iterative multi-turn QA loop
            while turn_counter <= args.max_turns:
                next_questions = []
                next_answers = []
                multiturn[f"turn_{turn_counter}"] = []

                for q, a in zip(current_questions, current_answers):

                    # Handle empty or "No Answer" by providing the all original text
                    if a == "No Answer" or not a or str(a).strip() == "":
                        a = src
                    else:
                        a = str(a)

                    question_id = f"{entry_id}.{question_counter}"

                    # Generate follow-up question
                    next_q = generate_followup_question(
                        tokenizer,
                        model,
                        followup_prompt,
                        text=src,
                        prev_question=q,
                        prev_answer=a
                    )

                    # Generate answer from SRC
                    followup_response = generate_response(
                        tokenizer,
                        model,
                        qa_prompt,
                        text=src,
                        question=next_q
                    )
                    # 
                    followup_response = clean_response(followup_response)

                    # Generate answer from BT
                    followup_response_bt = generate_response(
                        tokenizer,
                        model,
                        qa_prompt,
                        text=bt,
                        question=next_q
                    followup_response_bt = clean_response(followup_response_bt)

                    # Store multi-turn interaction
                    multiturn[f"turn_{turn_counter}"].append({
                        "question_id": question_id,
                        "question": q,
                        "answer_src": str(a) if a != src else a,
                        "follow_up_question": next_q,
                        "follow_up_response": followup_response,
                        "follow_up_response_bt": followup_response_bt
                    })

                    next_questions.append(next_q)
                    next_answers.append(followup_response)
                    question_counter += 1

                current_questions = next_questions
                current_answers = next_answers
                turn_counter += 1

            data["multiturn"] = multiturn
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

    print("MULTI-TURN ASKQE COMPLETED")

if __name__ == "__main__":
    main()
