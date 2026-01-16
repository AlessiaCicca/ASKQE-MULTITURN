qg_prompt = """Task: Generate relevant questions based on the given English sentence.

CRITICAL RULES:
- Output MUST be a valid JSON array: ["Question 1?", "Question 2?", "Question 3?"]
- Each question MUST end with a question mark (?)
- Generate between 3-5 questions
- NO explanations, NO code blocks (```), NO extra text
- Output ONLY the JSON array, nothing else

*** Example Starts ***
Sentence: It is not yet known whether the severity or level of control of underlying health conditions affects the risk for severe disease associated with COVID-19.
Questions: ["What is not yet known?", "What might affect the risk for severe disease associated with COVID-19?", "What is associated with severe disease?", "What conditions are mentioned in the sentence?", "What is the disease mentioned in the sentence?"]

Sentence: The number of accessory proteins and their function is unique depending on the specific coronavirus.
Questions: ["What is unique depending on the specific coronavirus?", "What is unique about the function of accessory proteins?"]
*** Example Ends ***

Sentence: {{sentence}}
Questions:"""