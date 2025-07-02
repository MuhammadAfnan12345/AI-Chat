# retrieval.py
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import streamlit as st

@st.cache_resource
def get_models():
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    reranker = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2')
    return embed_model, reranker

@st.cache_resource
def load_faiss():
    index = faiss.read_index('faiss_index.index')
    with open('qa_metadata.pkl', 'rb') as f:
        qa_data = pickle.load(f)
    return index, qa_data

def retrieve_top_k(query: str, faiss_k=10, rerank_k=3):
    embed_model, reranker = get_models()
    index, qa_data = load_faiss()

    # Encode query
    q_emb = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

    # FAISS search
    _, indices = index.search(q_emb, faiss_k)
    candidates = [qa_data[idx] for idx in indices[0]]

    # Rerank
    inputs = [[query, c['question'] + ' ' + c['answer']] for c in candidates]
    scores = reranker.predict(inputs)
    scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

    return [c for _, c in scored[:rerank_k]]
