from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text):
    print('đang xử lí text-embedding')

    vector = model.encode(text)
    print('xong text-embedding')

    return np.array(vector).astype('float32')
