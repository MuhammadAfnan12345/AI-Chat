# app.py
import streamlit as st
import re
from retrieval import retrieve_top_k

st.set_page_config(page_title="FAST Chatbot", page_icon="🤖", layout="centered")
st.title("🧠 FAST-NUCES & KDD Lab Chatbot")

# Session state for chat history
if 'history' not in st.session_state:
    st.session_state.history = []

st.markdown("---")
st.subheader("Ask a Question")

# Sidebar
st.sidebar.header("Settings")
if st.sidebar.button("🧹 Clear History"):
    st.session_state.history = []

# History handler
def handle_history_request(query):
    q_lower = query.lower()
    if "history" in q_lower or "last" in q_lower:
        n = min(len(st.session_state.history), 3)
        items = st.session_state.history[-n:]
        return "\n\n".join([f"{i+1}. Q: {q}\n   A: {a}" for i, (q, a) in enumerate(items)])
    return None

# Process query
def process_query(query):
    history_ans = handle_history_request(query)
    if history_ans:
        return history_ans

    results = retrieve_top_k(query, faiss_k=10, rerank_k=3)
    if not results:
        return "Sorry, I couldn't find an answer."

    best = results[0]
    st.session_state.history.append((query, best['answer']))

    display = f"**Answer:** {best['answer']}\n\n_Source Q: {best['question']}_"
    if len(results) > 1:
        display += "\n\n---\n**Other Suggestions:**\n"
        for r in results[1:]:
            display += f"- **Q:** {r['question']}\n  **A:** {r['answer']}\n"
    return display

# Chat form
with st.form("query_form", clear_on_submit=True):
    user_query = st.text_input("Enter your question:")
    submit = st.form_submit_button("Ask")

if submit and user_query:
    with st.spinner("Searching..."):
        response = process_query(user_query)
        st.markdown(response)

# Show full chat history
if st.session_state.history:
    st.markdown("### 📜 Conversation History")
    for q, a in reversed(st.session_state.history):
        with st.expander(f"Q: {q[:50]}{'...' if len(q) > 50 else ''}"):
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Bot:** {a}")
