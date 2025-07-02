import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def main():
    # Load your FAQ data
    with open('faq.json', 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)

    # Extract texts to embed: typically we embed question + answer
    texts = [item['question'] + ' ' + item['answer'] for item in qa_pairs]

    # Load the embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Encode all texts to get embeddings
    print("🔄 Encoding texts...")
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product = cosine similarity with normalized embeddings
    index.add(embeddings)

    # Save the index
    faiss.write_index(index, 'faiss_index.index')
    print("✅ FAISS index saved to faiss_index.index")

    # Save the QA metadata
    with open('qa_metadata.pkl', 'wb') as f:
        pickle.dump(qa_pairs, f)
    print("✅ QA metadata saved to qa_metadata.pkl")

if __name__ == '__main__':
    main()
