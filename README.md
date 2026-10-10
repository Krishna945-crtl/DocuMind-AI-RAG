# DocuMind AI — Document Q&A with RAG

A portfolio-ready starter project that answers questions about uploaded PDFs using Retrieval-Augmented Generation (RAG).

## Features

- FastAPI REST backend and a simple browser UI
- PDF text extraction and overlapping text chunks
- Semantic embeddings with Sentence Transformers
- Cosine-similarity retrieval
- Gemini-powered grounded answers (optional API key)
- Source passages with filename, chunk number, similarity score, and excerpt
- Fallback when no API key is configured: retrieval still works and displays the passages
- Basic automated chunking tests
- Prompt-injection-aware instruction to treat document content as untrusted

## Architecture

1. **Ingest:** Upload PDF → extract text with `pypdf`.
2. **Chunk:** Split text into overlapping chunks.
3. **Embed:** Convert chunks to vectors with `all-MiniLM-L6-v2`.
4. **Retrieve:** Embed the question and rank chunks by cosine similarity.
5. **Generate:** Send the question plus retrieved passages to Gemini.
6. **Inspect:** Return answer and source excerpts to the UI.

This starter stores document chunks and vectors in memory, so uploaded documents must be re-indexed after a server restart. It is intended for a local portfolio demo, not production use.

## Requirements

- Python 3.11 or 3.12 recommended
- Internet access on first run to install packages and download the embedding model
- A Gemini API key for generated answers; retrieval-only mode works without one

## Setup (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set `GEMINI_API_KEY` if you want generated answers. Never commit `.env` or publish your API key.

The app automatically loads values from `.env`. Set `GEMINI_API_KEY` there, or set the environment variable in PowerShell:

```powershell
$env:GEMINI_API_KEY="your_key_here"
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000. API docs: http://127.0.0.1:8000/docs

## Setup (macOS/Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
export GEMINI_API_KEY="your_key_here"
uvicorn app.main:app --reload
```

## Run tests

```bash
pytest -q
```

The embedding model may take a little time to download on first app startup.

## API endpoints

- `GET /health` — status and indexed chunk count
- `POST /api/documents` — multipart form upload; field name `files`; accepts PDFs
- `POST /api/ask` — JSON body such as `{"question":"What is the main conclusion?","top_k":4}`
- `DELETE /api/documents` — clear the in-memory index and uploaded PDFs

## Evaluation plan (recommended before putting on your resume)

Create a small set of 20–30 questions with expected source passages. Track:
- Retrieval Recall@k: did a relevant passage appear in the top k?
- Answer faithfulness: are factual claims supported by retrieved context?
- Answer correctness: does the answer match the document?
- Abstention quality: does it admit when the document does not contain the answer?
- Latency and failure rate

Do not claim a specific accuracy score until you have run a repeatable evaluation and documented the results.

## Known limitations

- Text-based PDFs only; scanned PDFs require OCR.
- In-memory vector store; no persistence across restarts.
- No user accounts, rate limiting, or multi-tenant isolation.
- Similarity scores are not calibrated probabilities.
- LLM answers can be wrong; verify them against source excerpts.
- Prompt instructions help but do not guarantee complete protection against prompt injection.
- For production, add persistent storage, authentication, file size limits, deletion policies, observability, and stronger isolation.

## Suggested portfolio extensions

1. Add OCR support for scanned PDFs.
2. Persist vectors in FAISS or Chroma.
3. Add a repeatable evaluation script and report.
4. Add Docker and CI tests.
5. Add a cloud deployment and document the trade-offs.
6. Add hybrid search or reranking only after establishing a baseline.
