from sentence_transformers import SentenceTransformer
import numpy as np

# 1. 加载预训练模型
model = SentenceTransformer("all-MiniLM-L6-v2")
 
# 2. 待编码的句子列表
sentences = [
    "The cat is sitting on the sofa.",
    "A kitten is resting on the couch.",
    "The Transformer uses self-attention."
]
 
# 3. 计算嵌入向量
embeddings = model.encode(sentences)

print(f"embeddings type: {type(embeddings)}")
print(f"embeddings shape: {embeddings.shape}")
print(f"embeddings[0] type: {type(embeddings[0])}")
print(f"embeddings[0] shape: {embeddings[0].shape}")
print(f"embedding[0] 前 10 个数字: {embeddings[0][:10]}")

#similarity
def cosine_similarity(a, b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

print(f"sentence 1 vs 2 = {cosine_similarity(embeddings[0],embeddings[1])}")
print(f"sentence 1 vs 3 = {cosine_similarity(embeddings[0],embeddings[2])}")