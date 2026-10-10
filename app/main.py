import os
import shutil
import uuid
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.rag import RAGService

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="DocuMind AI", version="1.0.0",
              description="Document question answering with retrieval-augmented generation.")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
rag = RAGService()

class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1500)
    top_k: int = Field(default=4, ge=1, le=8)

@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")

@app.get("/health")
def health():
    return {"status": "ok", "documents_indexed": rag.document_count}

@app.post("/api/documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    accepted = []
    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        safe_name = Path(file.filename).name
        destination = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_name}"
        try:
            with destination.open("wb") as out:
                shutil.copyfileobj(file.file, out)
            count = rag.add_pdf(destination, display_name=safe_name)
            if count == 0:
                destination.unlink(missing_ok=True)
                raise HTTPException(status_code=400, detail=f"No readable text found in {safe_name}. Scanned PDFs need OCR.")
            accepted.append({"filename": safe_name, "chunks_added": count})
        except HTTPException:
            raise
        except Exception as exc:
            destination.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail=f"Could not process {safe_name}: {exc}") from exc
        finally:
            await file.close()
    return {"message": "Documents indexed successfully.", "documents": accepted,
            "total_chunks": rag.document_count}

@app.post("/api/ask")
def ask(request: AskRequest):
    if rag.document_count == 0:
        raise HTTPException(status_code=400, detail="Upload at least one text-based PDF first.")
    try:
        return rag.answer(request.question, top_k=request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Question answering failed: {exc}") from exc

@app.delete("/api/documents")
def clear_documents():
    rag.clear()
    for path in UPLOAD_DIR.glob("*"):
        if path.is_file():
            path.unlink(missing_ok=True)
    return {"message": "All indexed documents cleared."}
