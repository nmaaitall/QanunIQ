from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_documents(pages, chunk_size=500, chunk_overlap=50):
    # split pages into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(pages)

    print(f"Pages: {len(pages)}")
    print(f"Chunks: {len(chunks)}")

    return chunks


def preview_chunks(chunks, num_chunks=3):
    # print first N chunks
    for i, chunk in enumerate(chunks[:num_chunks]):
        print(f"\n--- Chunk {i + 1} ---")
        print(chunk.page_content[:200])


if __name__ == "__main__":
    from pdf_loader import load_pdf

    pages = load_pdf("../../data/raw/Jordan_2016-en.pdf")
    chunks = chunk_documents(pages)
    preview_chunks(chunks)