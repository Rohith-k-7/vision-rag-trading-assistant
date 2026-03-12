import os
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import pickle

PDF_DIR = r"C:\Users\Pc\Downloads\rohith.freeee7\trading chart rag\data\trading_pdfs"
DB_DIR = r"C:\Users\Pc\Downloads\rohith.freeee7\trading chart rag\vector_db"

os.makedirs(DB_DIR, exist_ok=True)

def load_pdfs():
    texts = []
    for file in os.listdir(PDF_DIR):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(PDF_DIR, file))
            for page in reader.pages:
                texts.append(page.extract_text())
    return texts

def build_vector_db():
    texts = load_pdfs()
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(texts)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, f"{DB_DIR}/index.faiss")
    with open(f"{DB_DIR}/texts.pkl", "wb") as f:
        pickle.dump(texts, f)

    print("✅ Trading PDFs ingested into vector DB")

if __name__ == "__main__":
    build_vector_db()