"""Version 5.0 — small, manually labeled RAG evaluation set.

Source: Vaswani et al., 'Attention Is All You Need'.
Run: python eval_questions.py
"""

EVAL_QUESTIONS = [
    {
        "id": "Q1",
        "question": "What optimizer was used to train the Transformer?",
        "expected_answer": "Adam optimizer (β1 = 0.9, β2 = 0.98, ε = 10^-9).",
        "evidence_section": "§5.3 Optimizer",
        "evidence_text": "We used the Adam optimizer",
    },
    {
        "id": "Q2",
        "question": "How many attention heads did the Transformer use?",
        "expected_answer": "8 attention heads.",
        "evidence_section": "§3.2.2 Multi-Head Attention",
        "evidence_text": "we employ h = 8 parallel attention layers",
    },
    {
        "id": "Q3",
        "question": "How many layers are in the Transformer encoder stack?",
        "expected_answer": "6 identical encoder layers.",
        "evidence_section": "§3.1 Encoder and Decoder Stacks",
        "evidence_text": "a stack of N = 6 identical layers",
    },
    {
        "id": "Q4",
        "question": "Which dataset was used for English-to-German translation training, and approximately how many sentence pairs did it contain?",
        "expected_answer": "WMT 2014 English-German; approximately 4.5 million sentence pairs.",
        "evidence_section": "§5.1 Training Data and Batching",
        "evidence_text": "WMT 2014 English-German dataset",
    },
    {
        "id": "Q5",
        "question": "What BLEU score did the Transformer (big) achieve on the WMT 2014 English-to-German translation task?",
        "expected_answer": "28.4 BLEU.",
        "evidence_section": "§6.1 Machine Translation (also Table 2)",
        "evidence_text": "a new state-of-the-art BLEU score of 28.4",
    },
]


if __name__ == "__main__":
    print(f"Evaluation questions: {len(EVAL_QUESTIONS)}\n")
    for item in EVAL_QUESTIONS:
        print(f"{item['id']}: {item['question']}")
        print(f"  Expected answer: {item['expected_answer']}")
        print(f"  Evidence: {item['evidence_section']}")
        print(f"  Evidence text: {item['evidence_text']}\n")
