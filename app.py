# app.py
import streamlit as st
import re
from retrieval import retrieve_top_k, RetrievalError

GREETINGS_RESPONSES = {
    "hello": "Hello! How can I help you with questions about FAST-NUCES or the KDD Research Lab?",
    "hi": "Hi there! What can I do for you?",
    "hey": "Hey! How can I assist you?",
    "thanks": "You're welcome!",
    "thank you": "You're welcome! Is there anything else I can help with?",
    "bye": "Goodbye! Have a great day."
}

SIMILARITY_THRESHOLD = 0.5 

st.set_page_config(page_title="FAST Chatbot", page_icon="🤖", layout="centered")
st.title("🧠 FAST-NUCES & KDD Lab Chatbot")

if 'history' not in st.session_state:
    st.session_state.history = []

st.markdown("---")
st.subheader("Ask a Question")

# Sidebar
st.sidebar.header("Settings")
if st.sidebar.button("🧹 Clear History"):
    st.session_state.history = []


def handle_greetings(query):
    """
    Checks if any word in a short query is a greeting and returns a canned response.
    """
    clean_query = query.strip().lower()
    words = clean_query.split()

    if len(words) <= 3:
        for word in words:
            # Check if any individual word is a key in our greetings dictionary
            if word in GREETINGS_RESPONSES:
                # If we find a greeting, return its corresponding response and stop.
                return GREETINGS_RESPONSES[word]
    return None

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

    # This now returns a list of (score, candidate) tuples
    results_with_scores = retrieve_top_k(query, faiss_k=10, rerank_k=3)
    
    if not results_with_scores:
        return "Sorry, I couldn't find an answer to your question."

    # Get the score and data for the best result
    top_score, best_result = results_with_scores[0]
    
    # --- THRESHOLD LOGIC ---
    if top_score < SIMILARITY_THRESHOLD:
        return "I can only answer questions about the FAST-NUCES university and the KDD Research Lab. Please ask a relevant question."

    # If the score is high enough, proceed as normal
    st.session_state.history.append((query, best_result['answer']))
    
    display = f"**Answer:** {best_result['answer']}\n\n"
    # You can optionally show the score for debugging/transparency
    #display += f"_(Source Q: {best_result['question']} | Relevance Score: {top_score:.2f})_"

    # Handle other suggestions
    if len(results_with_scores) > 1:
        display += "\n\n---\n**Other Suggestions:**\n"
        for score, r in results_with_scores[1:]:
             # Optionally, only show suggestions that are also above a certain (lower) threshold
             if score > 0.1: 
                display += f"- **Q:** {r['question']}\n  **A:** {r['answer']}\n"
    return display


# Chat form
with st.form("query_form", clear_on_submit=False):

    user_query = st.text_area(
        "Enter your question:", 
        height=100, # Optional: makes the input box a bit taller
        placeholder="Ask anything about the KDD Lab or FAST-NUCES..." # Optional: better prompt
    )
    submit = st.form_submit_button("Ask")

if submit and user_query:
    clean_query = user_query.strip()

    # 1. First, check for simple greetings
    greeting_response = handle_greetings(clean_query)
    if greeting_response:
        st.markdown(greeting_response)
        st.session_state.history.append((clean_query, greeting_response))
        # No st.stop() needed here because it's a successful path.
    
    # 2. If not a greeting, proceed to validation
    else:
        # --- MODIFIED AND CORRECTED VALIDATION BLOCK ---
        if len(clean_query.split()) > 200:
            st.warning("⚠️ Your question is too long. Please limit it to 200 words.")
            st.stop() 

        elif len(clean_query) < 3: 
            st.warning("⚠️ Please ask a more specific question.")
            st.stop() # <-- THE FIX: Stop execution here

        elif re.fullmatch(r'[\d\s.,-]+', clean_query):
            st.warning("⚠️ Your question appears to contain only numbers. Please ask a full question.")
            st.stop() # <-- THE FIX: Stop execution here
        
        elif not re.search(r'[a-zA-Z]', clean_query):
            st.warning("⚠️ Your question must contain letters. Please ask a valid question.")
            st.stop() # <-- THE FIX: Stop execution here
        
        # If ALL validation passes, run the main query processor
        with st.spinner("Searching..."):
            try:
                response = process_query(clean_query)
                st.markdown(response)
            except RetrievalError as e:
                st.error(f"🚨 Application Error: {e}. Please contact the administrator.")
            except Exception as e:
                st.error("🚨 An unexpected error occurred. Please try again later.")
                print(f"Unexpected error: {e}")

# Show full chat history
if st.session_state.history:
    st.markdown("### 📜 Conversation History")
    for q, a in reversed(st.session_state.history):
        with st.expander(f"Q: {q[:50]}{'...' if len(q) > 50 else ''}"):
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Bot:** {a}")