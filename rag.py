import retrieve


retrieved_chunks = []

for rank, index in enumerate(retrieve.top_indices, start=1):
    chunk = retrieve.chunks[index]

    retrieved_chunks.append(
        f"[Retrieved Chunk {rank} | Original Index {index}]\n{chunk}"
    )


context = "\n\n".join(retrieved_chunks)

query = retrieve.query

prompt = f"""You are answering a question about a research paper.

Use only the context below to answer the question.
If the context does not contain enough information,
say that the information is not available in the provided context.

Context:
{context}

Question:
{query}

Answer:
"""

print("===== FINAL PROMPT =====")
print(prompt)

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError("请先设置 DEEPSEEK_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": prompt}
    ],
    stream=False
)

answer = response.choices[0].message.content

print("\n===== RAG ANSWER =====")
print(answer)

print("\n===== TOKEN USAGE =====")
print(response.usage)