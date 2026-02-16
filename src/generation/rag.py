from groq import Groq
from dotenv import load_dotenv
import os
import sys

sys.path.append("../..")
sys.path.append("../../src")

from src.embeddings.embedder import load_embedder
from src.retrieval.vector_store import load_vector_store

load_dotenv("../../.env")


def ask(query, vector_store, client):
    # retrieve relevant chunks
    results = vector_store.similarity_search(query, k=3)

    # build context from results
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

    return response.choices[0].message.content


if __name__ == "__main__":
    embedder = load_embedder()
    vector_store = load_vector_store(embedder)

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    questions = [
        "What are the rights of Jordanian citizens?",
        "Who is the head of state in Jordan?",
        "What is the official religion of Jordan?"
    ]

    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {ask(q, vector_store, client)}")
        print("-" * 50)