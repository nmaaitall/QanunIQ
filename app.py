import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os
import sys
import re

sys.path.append(".")

from src.embeddings.embedder import load_embedder
from src.retrieval.vector_store import load_vector_store

load_dotenv(".env")

st.set_page_config(
    page_title="QanunIQ",
    page_icon="⚖️",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stTitle { color: #c9a84c; font-size: 3rem; }
    .source-box {
        background-color: #1e2130;
        border-left: 3px solid #c9a84c;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        font-size: 0.85rem;
    }
    .law-badge {
        background-color: #c9a84c;
        color: black;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.75rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    embedder = load_embedder()
    vector_store = load_vector_store(embedder)
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return embedder, vector_store, client


embedder, vector_store, client = load_models()

with st.sidebar:
    st.markdown("# ⚖️")
    st.title("QanunIQ")
    st.markdown("**Your intelligent guide through Jordanian law**")
    st.divider()
    st.markdown("### Available Laws")
    st.markdown("🟡 **Jordan Constitution 1952**")
    st.markdown("🔜 Labor Law *(coming soon)*")
    st.markdown("🔜 Civil Law *(coming soon)*")
    st.markdown("🔜 Commercial Law *(coming soon)*")
    st.divider()
    st.markdown("Built with RAG + LLM")

st.markdown("# ⚖️ QanunIQ")
st.markdown("### Your intelligent guide through Jordanian law")
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            st.markdown("**Sources:**")
            for src in msg["sources"]:
                match = re.search(r'Article\s+\d+', src)
                citation = match.group(0) if match else "Jordan Constitution"
                st.markdown(f"""
                <div class="source-box">
                    <span class="law-badge">Source: {citation} - Jordan Constitution 1952</span><br>
                    {src[:200]}...
                </div>
                """, unsafe_allow_html=True)

query = st.chat_input("Ask a legal question...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            results = vector_store.similarity_search(query, k=3)
            context = "\n\n".join([doc.page_content for doc in results])
            sources = [doc.page_content for doc in results]

            prompt = f"""You are a legal assistant for Jordanian law.
Use the context below to answer the question.
Be concise and clear.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {query}
Answer:"""

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )

            answer = response.choices[0].message.content
            st.write(answer)

            st.markdown("**Sources:**")
            for src in sources:
                match = re.search(r'Article\s+\d+', src)
                citation = match.group(0) if match else "Jordan Constitution"
                st.markdown(f"""
                <div class="source-box">
                    <span class="law-badge">Source: {citation} - Jordan Constitution 1952</span><br>
                    {src[:200]}...
                </div>
                """, unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
