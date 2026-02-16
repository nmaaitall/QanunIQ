import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os
import sys

sys.path.append(".")

from src.embeddings.embedder import load_embedder
from src.retrieval.vector_store import load_vector_store

load_dotenv(".env")

st.set_page_config(page_title="QanunIQ", page_icon="⚖️")
st.title("QanunIQ")
st.subheader("Your intelligent guide through Jordanian law")


@st.cache_resource
def load_models():
    embedder = load_embedder()
    vector_store = load_vector_store(embedder)
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return embedder, vector_store, client


embedder, vector_store, client = load_models()

query = st.text_input("Ask a legal question:")

if st.button("Ask"):
    if query:
        with st.spinner("Searching..."):
            results = vector_store.similarity_search(query, k=3)
            context = "\n\n".join([doc.page_content for doc in results])

            prompt = f"""You are a legal assistant for Jordanian law.
Use the context below to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {query}
Answer:"""

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )

            st.markdown("### Answer")
            st.write(response.choices[0].message.content)
    else:
        st.warning("Please enter a question")