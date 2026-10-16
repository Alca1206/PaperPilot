# PaperPilot

A local-first AI research paper assistant. Upload an academic PDF, get a structured
summary, ask questions answered **only** from the paper with page-level citations,
and export references in APA, MLA, IEEE or BibTeX.

Everything runs on your own machine: no paid APIs, no data leaves your computer.

**Stack:** Python · Streamlit · Ollama (`qwen3:8b`, `embeddinggemma`) · PyMuPDF · ChromaDB · Pydantic · pytest

> **Status:** Phase 1 (foundation) complete. The app starts and checks that Ollama and
> the required models are available. PDF upload, Q&A, summaries and references arrive
> in later phases; see [Roadmap](#roadmap).

---

## Requirements

- Windows 10/11 (macOS/Linux work too; adapt the commands)
- Python 3.11 or 3.12 → check with `py --list`
- [Ollama](https://ollama.com/download)
- ~6 GB free disk space for the models
- 8 GB+ RAM recommended

## Setup (Windows, PowerShell)

### 1. Get the code

```powershell
git clone <your-repo-url> PaperPilot
cd PaperPilot
```

### 2. Create and activate a virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Your prompt should now start with `(.venv)`.

> **"running scripts is disabled on this system"?** Run this once, then activate again:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Install Ollama and the models

```powershell
winget install --id Ollama.Ollama -e    # or use the installer from ollama.com
```

Open a **new** PowerShell window so `ollama` is on your PATH, then:

```powershell
ollama pull qwen3:8b
ollama pull embeddinggemma
ollama list
```

### 5. Run the app

```powershell
streamlit run app.py
```

Opens at <http://localhost:8501>. The sidebar shows whether Ollama is reachable and
whether both models are installed. The **System status** tab shows details and the
exact `ollama pull` command for anything missing.

## Running tests

```powershell
python -m pytest                       # all tests
python -m pytest -v                    # verbose
python -m pytest -m "not integration"  # skip tests that need a running Ollama
```

Unit tests use fake clients and pass without Ollama. The integration test talks to a
real Ollama server and is **skipped** (not failed) if none is running.

## Configuration

Defaults live in `paperpilot/config.py`. Override any of them with environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `PAPERPILOT_OLLAMA_HOST` | `http://localhost:11434` | Ollama server address |
| `PAPERPILOT_CHAT_MODEL` | `qwen3:8b` | Model that writes answers and summaries |
| `PAPERPILOT_EMBEDDING_MODEL` | `embeddinggemma` | Model that embeds text for search |
| `PAPERPILOT_REQUEST_TIMEOUT_SECONDS` | `10` | Timeout for Ollama requests |
| `PAPERPILOT_DATA_DIR` | `./data` | Uploads, vector store and exports |

Example (current PowerShell session only):

```powershell
$env:PAPERPILOT_CHAT_MODEL = "llama3.2:3b"
streamlit run app.py
```

No secrets are needed. `data/` is git-ignored.

## Project structure


## Concept notebook

`notebooks/paperpilot_concept.ipynb` prototypes the whole pipeline, one idea per part:
models → PDF pages → chunks → retrieval → ChromaDB → grounded Q&A → summaries →
references. It needs a few extra packages that the app only adds in later phases:

```powershell
python -m pip install ipykernel pymupdf chromadb
```

## Roadmap

| Phase | Feature | Status |
|---|---|---|
| 1 | Foundation: config, health check, Streamlit shell, tests | ✅ |
| 2 | PDF upload, page-level extraction, chunking | ⏳ |
| 3 | Embeddings + persistent ChromaDB index | ⏳ |
| 4 | Grounded Q&A with page citations | ⏳ |
| 5 | Structured summaries + Markdown download | ⏳ |
| 6 | Bibliography (APA, MLA, IEEE, BibTeX) | ⏳ |
| 7 | Polish, multi-paper management, integration tests | ⏳ |

Later: multi-paper comparison, hybrid retrieval, deployment.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ollama : The term 'ollama' is not recognized` | Ollama isn't installed, or PowerShell was open before installing. Install it and open a new window. |
| Sidebar: "Ollama is not reachable" | Start the Ollama app (llama icon in the system tray), then click **Re-check**. |
| Sidebar: "Missing model(s)" | Run the `ollama pull ...` command shown in the System status tab. |
| `ModuleNotFoundError: No module named 'paperpilot'` | Run commands from the project root, with the venv active. |
| First answer is very slow | Normal: Ollama loads the model into memory on first use. |

## License

See [LICENSE](LICENSE).


| `PAPERPILOT_MODEL_TIMEOUT_SECONDS` | `300` | Timeout for embedding/chat calls (CPU can be slow) |
| `PAPERPILOT_EMBED_BATCH_SIZE` | `32` | Chunks sent to Ollama per embedding request |
| `PAPERPILOT_EMBED_USE_PREFIXES` | `true` | EmbeddingGemma task prefixes on queries/documents |