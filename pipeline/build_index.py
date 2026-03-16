from rag.loader import load_pdf
from rag.chunker import chunk_text
from rag.embedder import embed_text
import os

import faiss
import numpy as np
import pickle 

text = load_pdf('data/data_sample.pdf')
chunks = chunk_text(text)
embeddings = []

for chunk in chunks:
    embeddings.append(embed_text(chunk))

dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype('float32'))
faiss.write_index(index, 'vector.index')

with open('chunks.pkl', 'wb') as f:
    pickle.dump(chunks, f)

print('Vector database built successfully')
