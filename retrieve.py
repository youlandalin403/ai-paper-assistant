import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from chunk_text import chunk_text


# 1. 读取全文
with open("output.txt", "r", encoding="utf-8") as f:
    full_text = f.read()


# 2. Chunking
chunks = chunk_text(
    full_text,
    chunk_size=1000,
    overlap=200
)

print("number of chunks:", len(chunks))


# 3. 加载 embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# 4. 所有 chunk → embeddings
embeddings = model.encode(chunks)

print("chunk embeddings shape:", embeddings.shape)


# 5. 用户问题
query = "What optimizer was used to train the Transformer?"


# 6. query → embedding
query_embedding = model.encode([query])

print("query embedding shape:", query_embedding.shape)


# 7. 计算 query 和所有 chunks 的 cosine similarity
cos_sim = cosine_similarity(
    embeddings,
    query_embedding
)

print("cos_sim shape:", cos_sim.shape)


# (54, 1) → (54,)
scores = cos_sim.flatten()

print("scores shape:", scores.shape)


# 8. 从大到小找 Top-3
top_indices = np.argsort(scores)[-3:][::-1]


# 9. 打印检索结果
print("\nQuery:")
print(query)

for rank, index in enumerate(top_indices, start=1):
    print(f"\n===== TOP {rank} =====")
    print("index:", index)
    print("score:", scores[index])
    print("text:")
    print(chunks[index])

