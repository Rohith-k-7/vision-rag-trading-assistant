import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
from google import genai

# 🔑 API KEY
client = genai.Client(api_key="YOUR API KEY")

DB_DIR = r"C:\Users\Pc\Downloads\rohith.freeee7\trading chart rag\vector_db"

def retrieve_context(query, k=3):
    index = faiss.read_index(f"{DB_DIR}/index.faiss")
    with open(f"{DB_DIR}/texts.pkl", "rb") as f:
        texts = pickle.load(f)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query]).astype("float32")

    _, indices = index.search(query_embedding, k)
    return "\n".join([texts[i] for i in indices[0]])

def explain_chart_with_rag(image_path, question):
    image = Image.open(image_path)

    rag_context = retrieve_context(question)

    prompt = f"""
You are an educational trading assistant.

Trading Theory Context:
{rag_context}

User Question:
{question}

Explain the chart clearly for a beginner.
Do NOT give financial advice.
"""

    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=[prompt, image]
    )

    return response.text

if __name__ == "__main__":
    answer = explain_chart_with_rag(
        r"C:\Users\Pc\Downloads\rohith.freeee7\trading chart rag\data\chart.jfif",
        "Explain this trading chart"
    )
    print("\n===== AI Explanation =====\n")
    print(answer)