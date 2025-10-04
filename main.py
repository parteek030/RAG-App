from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List
import os
import shutil
from langchain_groq import ChatGroq
from app.query_processing import Preprocessing, EmbeddingManager, VectorStore, RAGRetriever, RAG

app = FastAPI()

# --- Absolute paths for safety ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
INDEX_FILE = os.path.join(BASE_DIR, "index.html")
STATIC_FOLDER = os.path.join(BASE_DIR, "static")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Serve frontend ---
if os.path.exists(STATIC_FOLDER):
    app.mount("/static", StaticFiles(directory=STATIC_FOLDER), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    if not os.path.exists(INDEX_FILE):
        return HTMLResponse("<h1>Frontend file not found!</h1>", status_code=500)
    try:
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return HTMLResponse(f"<h1>Error loading frontend: {str(e)}</h1>", status_code=500)

# --- Global objects ---
embedding_manager: EmbeddingManager | None = None
vectorstore: VectorStore | None = None
processed_documents: list = []

# --- Step 1: Upload PDFs ---
@app.post("/upload-multiple")
async def upload_multiple(files: List[UploadFile] = File(...)):
    global embedding_manager, vectorstore, processed_documents

    if not files:
        return JSONResponse({"error": "No files uploaded."}, status_code=400)

    try:
        # --- Clear existing uploads safely ---
        if os.path.exists(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        processed_documents = []  # reset
        uploaded_files = []

        # Save uploaded PDFs
        for file in files:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            uploaded_files.append({"filename": file.filename, "path": file_path})

        print(f"Uploaded {len(uploaded_files)} files.")

        # Process all PDFs
        documents = Preprocessing.process_all_pdfs(UPLOAD_FOLDER)
        chunks = Preprocessing.split_documents(documents)

        # Initialize embedding manager and vector store
        embedding_manager = EmbeddingManager()
        vectorstore = VectorStore()

        texts = [doc.page_content for doc in chunks]
        embeddings = embedding_manager.generate_embeddings(texts)
        vectorstore.add_documents(chunks, embeddings)

        processed_documents = documents

        return {
            "message": "Files uploaded and processed successfully",
            "files": [f["filename"] for f in uploaded_files]
        }

    except Exception as e:
        return JSONResponse({"error": f"Failed to process files: {str(e)}"}, status_code=500)

# --- Step 2: Query endpoint ---
class Query(BaseModel):
    text: str

@app.post("/query")
def query_rag(query: Query):
    global embedding_manager, vectorstore

    if not embedding_manager or not vectorstore:
        return JSONResponse({"error": "No PDFs uploaded yet."}, status_code=400)

    try:
        rag_retriever = RAGRetriever(vectorstore, embedding_manager)

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise HTTPException(status_code=500, detail="GROQ_API_KEY not set in environment variables")

        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="gemma2-9b-it",
            temperature=0.1,
            max_tokens=1024
        )

        result = RAG.rag_advanced(
            query.text,
            rag_retriever,
            llm,
            top_k=1,
            min_score=0.1,
            return_context=True
        )
        print(result.get("sources"))

        return {
            "Answer": result.get("answer"),
            "Sources": result.get("sources"),
            "Confidence": result.get("confidence")
        }

    except Exception as e:
        return JSONResponse({"error": f"Query failed: {str(e)}"}, status_code=500)
