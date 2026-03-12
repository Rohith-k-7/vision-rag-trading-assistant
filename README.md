# Trading Chart AI Assistant (Vision + RAG)

An AI-powered system that explains trading charts in a beginner-friendly way by combining **Computer Vision** and **Retrieval-Augmented Generation (RAG)**.

Users can upload a trading chart image and ask questions. The system retrieves relevant trading knowledge from PDFs and generates a clear explanation.

---

## Features

- Upload trading chart images
- Ask questions about chart patterns
- Retrieve trading knowledge using RAG
- Generate easy-to-understand explanations
- Simple Flask web interface

---

## Technologies Used

- Python
- Flask
- FAISS (Vector Database)
- Sentence Transformers
- Gemini Vision API
- PyPDF
- PIL

---

## System Architecture

User → Flask Web Interface → RAG Retrieval → Vector Database (FAISS) → Gemini Vision Model → AI Explanation

---
