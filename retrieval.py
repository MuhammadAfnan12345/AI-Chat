# retrieval.py
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import streamlit as st

# --- CHANGE 1: DEFINE A CUSTOM EXCEPTION CLASS ---
class RetrievalError(Exception):
    """Custom exception for errors during the retrieval process."""
    pass

@st.cache_resource
def get_models():
    """Loads the AI models. Raises RetrievalError if models can't be loaded."""
    try:
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        reranker = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2')
        return embed_model, reranker
    except Exception as e:
        # Catch any error during model loading (e.g., network issues)
        raise RetrievalError(f"Failed to load AI models. Please check your network connection or model cache. Error: {e}")

@st.cache_resource
def load_faiss():
    """Loads the FAISS index and metadata. Raises RetrievalError if files are missing."""
    # --- CHANGE 2: ADD EXCEPTION HANDLING FOR FILE LOADING ---
    try:
        index = faiss.read_index('faiss_index.index')
        with open('qa_metadata.pkl', 'rb') as f:
            qa_data = pickle.load(f)
        return index, qa_data
    except FileNotFoundError as e:
        # This is a very common and important error to handle gracefully.
        raise RetrievalError(f"A required data file was not found: {e.filename}. Please ensure 'faiss_index.index' and 'qa_metadata.pkl' are in the project's root directory.")
    except Exception as e:
        # Catch other potential errors, like a corrupted file.
        raise RetrievalError(f"Failed to load FAISS index or metadata. The file might be corrupted. Error: {e}")

def retrieve_top_k(query: str, faiss_k=10, rerank_k=3):
    # The functions below will now raise RetrievalError if they fail
    embed_model, reranker = get_models()
    index, qa_data = load_faiss()

    # Encode query
    q_emb = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

    # FAISS search
    _, indices = index.search(q_emb, faiss_k)
    
    # Handle case where FAISS returns no valid indices
    valid_indices = [idx for idx in indices[0] if idx != -1]
    if not valid_indices:
        return []
        
    candidates = [qa_data[idx] for idx in valid_indices]

    # Rerank
    inputs = [[query, c['question'] + ' ' + c['answer']] for c in candidates]
    scores = reranker.predict(inputs)
    
    # Zip scores with candidates and sort
    scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

    # Return the (score, candidate) pairs.
    return scored[:rerank_k]