import faiss
import pickle
import numpy as np
from rag.embedder import embed_text

index = faiss.read_index('vector.index')

with open('chunks.pkl', 'rb') as f:
    chunks = pickle.load(f)

def retrieve(query, k=3):
    query_vector = embed_text(query).astype('float32')
    query_vector = np.array([query_vector])
    distances, indices = index.search(query_vector, k)
    results = []
    for i in indices[0]:
        results.append(chunks[i])
    
    return results