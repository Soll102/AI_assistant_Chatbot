# Chat Bot - PDF RAG

A PDF RAG chatbot with document upload, PDF preview, and chat history.

## Features

- Upload and preview PDF files
- Extract PDF text with PyMuPDF
- Split text into chunks with page metadata
- Create local embeddings with Sentence Transformers
- Store vectors in ChromaDB
- Retrieve relevant chunks and generate answers with Gemini
- Save chat history with SQLite
- React UI with chat, PDF preview, and resizable panels

## Tech Stack

- Frontend: React, Vite
- Backend: FastAPI
- LLM: Gemini API
- Embeddings: Sentence Transformers
- Vector DB: ChromaDB
- PDF processing: PyMuPDF
- Database: SQLite

## Project Structure

```text
backend/    FastAPI API, RAG pipeline, PDF processing, vector store
frontend/   React UI
```

## Setup

### Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `backend/.env`:

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.1-flash-lite
```

Run the backend:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

## Notes

- The first backend run may download the local embedding model.
- Uploaded PDFs, ChromaDB data, and chat history are stored locally under `backend/storage/`.
