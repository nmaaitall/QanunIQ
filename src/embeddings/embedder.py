from langchain_community.embeddings import HuggingFaceEmbeddings


def load_embedder():
    # load embedding model
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("Embedder loaded")
    return embedder


def embed_chunks(chunks, embedder):
    # extract text from chunks
    texts = [chunk.page_content for chunk in chunks]

    # convert texts to vectors
    vectors = embedder.embed_documents(texts)

    print(f"Embedded: {len(vectors)} chunks")
    print(f"Vector size: {len(vectors[0])}")

    return vectors


if __name__ == "__main__":
    import sys

    sys.path.append("..")
    from ingestion.pdf_loader import load_pdf
    from ingestion.text_chunker import chunk_documents

    pages = load_pdf("../../data/raw/Jordan_2016-en.pdf")
    chunks = chunk_documents(pages)

    embedder = load_embedder()
    vectors = embed_chunks(chunks, embedder)