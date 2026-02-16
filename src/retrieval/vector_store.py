from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def build_vector_store(chunks, embedder):
    # build FAISS index from chunks
    vector_store = FAISS.from_documents(chunks, embedder)
    print(f"Vector store built with {len(chunks)} chunks")
    return vector_store


def save_vector_store(vector_store, path="data/vector_store"):
    # save to disk
    vector_store.save_local(path)
    print(f"Saved to {path}")


def load_vector_store(embedder, path="data/vector_store"):
    # load from disk
    vector_store = FAISS.load_local(path, embedder, allow_dangerous_deserialization=True)
    print(f"Loaded from {path}")
    return vector_store


def search(vector_store, query, k=3):
    # search for similar chunks
    results = vector_store.similarity_search(query, k=k)
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(doc.page_content[:300])
    return results


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from ingestion.pdf_loader import load_pdf
    from ingestion.text_chunker import chunk_documents
    from embeddings.embedder import load_embedder

    pages = load_pdf("../../data/raw/Jordan_2016-en.pdf")
    chunks = chunk_documents(pages)
    embedder = load_embedder()

    vector_store = build_vector_store(chunks, embedder)
    save_vector_store(vector_store)

    print("\n--- Search Test ---")
    search(vector_store, "what are the rights of Jordanian citizens")