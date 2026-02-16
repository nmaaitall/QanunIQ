from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os
import sys

sys.path.append("..")
sys.path.append("../src")

from src.embeddings.embedder import load_embedder
from src.retrieval.vector_store import load_vector_store

load_dotenv(".env")

app = FastAPI()

# load models once on startup
embedder = load_embedder()
vector_store = load_vector_store(embedder)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class Question(BaseModel):
    query: str


@app.get("/")
def root():
    return {"message": "QanunIQ API is running"}


@app.post("/ask")
def ask(question: Question):
    # retrieve relevant chunks
    results = vector_store.similarity_search(question.query, k=3)

    # build context
    context = "\n\n".join([doc.page_content for doc in results])

    prompt = f"""You are a legal assistant for Jordanian law.
Use the context below to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question.query}
Answer:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "question": question.query,
        "answer": response.choices[0].message.content
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)