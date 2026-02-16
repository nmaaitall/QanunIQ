from langchain_community.document_loaders import PyPDFLoader
import os


def load_pdf(file_path: str):
    # check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    loader = PyPDFLoader(file_path)
    pages = loader.load()

    print(f"Loaded: {len(pages)} pages")

    return pages


def preview_pages(pages, num_pages: int = 2):
    # print first N pages content
    for i, page in enumerate(pages[:num_pages]):
        print(f"\n--- Page {i + 1} ---")
        print(page.page_content[:300])


if __name__ == "__main__":
    pdf_path = "../../data/raw/Jordan_2016-en.pdf"

    pages = load_pdf(pdf_path)
    preview_pages(pages)