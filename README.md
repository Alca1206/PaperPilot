# PaperPilot

**A local-first AI assistant for research papers.** Upload a PDF, ask questions and get answers with page citations, generate a structured summary, and format references in APA, MLA, IEEE or BibTeX.

Everything runs on your own computer through [Ollama](https://ollama.com). There are no cloud services, no paid APIs, and your papers never leave your machine.

---

## Features

| Tab | What it does |
|---|---|
| 📄 **Paper** | Upload a PDF. Text is extracted page by page and split into chunks, and you can browse pages and chunks. **Build search index** embeds the chunks into a local ChromaDB vector store. |
| 💬 **Ask** | Ask questions about the paper. Answers come **only** from retrieved passages and cite them as `[1]`, `[2]` with page numbers. If nothing relevant is found, it says *"Not found in the paper."* |
| 📝 **Summary** | Structured summary: title, one-sentence summary, research problem, methodology, key findings (with pages), datasets, limitations, future work. Download as Markdown. |
| 📚 **References** | Extracts metadata from page 1, warns about anything not found on the page, and lets you check and confirm it in a form. Then formats it as APA 7, MLA 9, IEEE or BibTeX. Download `.bib` or `.md`. |
| 🩺 **System status** | Checks that Ollama is reachable and the required models are installed, and shows the `ollama pull` commands for any that are missing. |

### How PaperPilot avoids making things up

- **Answers are grounded.** The model sees only the retrieved passages and is told to answer from them alone.
- **Gate 1, relevance:** if the best passage scores below `QA_MIN_SCORE`, the model is **not called** and the answer is *"Not found in the paper."*
- **Gate 2, citations:** every `[n]` the model writes is checked. Citations to passages it was not given are flagged.
- **Page numbers come from metadata**, never from the model's text.
- **Structured output** (summary and metadata) is validated with Pydantic and retried with the error message on failure.
- **References are deterministic.** The AI only drafts the metadata. You confirm it, and plain Python code formats the citation styles.

> These safeguards *reduce* hallucinations; they do not eliminate them. Always check important claims against the paper.

---

## How it works

```
PDF ──► extract pages (PyMuPDF) ──► sentence chunks (≤800 chars, page-aware)
                                          │
                                          ▼
                              embeddings (embeddinggemma)
                                          │
                                          ▼
                              ChromaDB (cosine, on disk)
                                          │
question ──► embed ──► top-k chunks ──► Gate 1 (score) ──► qwen3:8b ──► Gate 2 (citations) ──► answer + pages
```

- **Summaries**: short papers are sent in one call. Long papers (> ~9000 tokens) use **map-reduce**: notes per group of pages first, then one summary from the notes.
- **Index reuse**: each paper's collection is named by its file hash plus the indexing settings, so the same PDF with the same settings is never embedded twice.

For details see [`docs/flow.md`](docs/flow.md) (how the code runs) and [`docs/decision.md`](docs/decision.md) (every design decision and why).

---

## Tech stack

Python 3.12+ · Streamlit · Ollama (`qwen3:8b`, `embeddinggemma`) · PyMuPDF · ChromaDB · Pydantic · pytest · Docker

---

## Quick start (local)

### 1. Install Ollama and the models

Install Ollama from <https://ollama.com/download>, then:

```bash
ollama pull qwen3:8b
ollama pull embeddinggemma
```

### 2. Install PaperPilot

```bash
git clone <your-repo-url>
cd PaperPilot
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Run

```bash
streamlit run app.py
```

Open <http://localhost:8501>. The sidebar should show **✅ Ollama connected. All models ready.**

---

## Run with Docker

The app container talks to the Ollama that is **already running on your computer** (it uses your GPU and the models you already downloaded):

```bash
docker compose up --build
```

Open <http://localhost:8501>. Uploads, the vector store and exports are saved in `./data`.

**Fully containerized** (Ollama in a container too):

```powershell
$env:PAPERPILOT_OLLAMA_HOST = "http://ollama:11434"     # PowerShell
docker compose --profile ollama up --build
docker compose exec ollama ollama pull qwen3:8b
docker compose exec ollama ollama pull embeddinggemma
```

Run the tests inside the container:

```bash
docker compose exec app pytest -q
```

---

## Configuration

All settings have sensible defaults. Override any of them with environment variables (prefix `PAPERPILOT_`):

| Variable | Default | Meaning |
|---|---|---|
| `PAPERPILOT_OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `PAPERPILOT_CHAT_MODEL` | `qwen3:8b` | Model for answers, summaries, metadata |
| `PAPERPILOT_EMBEDDING_MODEL` | `embeddinggemma` | Model for embeddings |
| `PAPERPILOT_DATA_DIR` | `./data` | Uploads, ChromaDB and exports |
| `PAPERPILOT_CHUNK_MAX_CHARS` | `800` | Maximum chunk size |
| `PAPERPILOT_CHUNK_OVERLAP_SENTENCES` | `1` | Sentences repeated between chunks |
| `PAPERPILOT_EMBED_BATCH_SIZE` | `32` | Chunks per embedding request |
| `PAPERPILOT_EMBED_USE_PREFIXES` | `true` | EmbeddingGemma task prefixes |
| `PAPERPILOT_QA_TOP_K` | `4` | Passages retrieved per question |
| `PAPERPILOT_QA_MIN_SCORE` | `0.35` | Relevance threshold (Gate 1) |
| `PAPERPILOT_CHAT_TEMPERATURE` | `0.1` | Low = more factual |
| `PAPERPILOT_SUMMARY_NUM_CTX` | `16384` | Context window for summaries |
| `PAPERPILOT_SUMMARY_DIRECT_LIMIT_TOKENS` | `9000` | Above this, use map-reduce |
| `PAPERPILOT_SUMMARY_PAGES_PER_GROUP` | `4` | Pages per map-reduce group |
| `PAPERPILOT_SUMMARY_MAX_ATTEMPTS` | `3` | Retries for invalid JSON |
| `PAPERPILOT_REQUEST_TIMEOUT_SECONDS` | `10` | Health-check timeout |
| `PAPERPILOT_MODEL_TIMEOUT_SECONDS` | `300` | Timeout for model calls |

Changing the chunk or embedding settings creates a new index automatically (the old one is not reused).

---

## Tests

```bash
pytest -q
```

**224 tests.** They use fake Ollama clients and PDFs generated in code, so they run in seconds **without Ollama**. A few integration tests (marked `integration`) talk to the real models and are skipped automatically when Ollama isn't running.

```bash
pytest -m integration    # only the real-model tests
pytest -m "not integration"
```

---

## Project structure

```
PaperPilot/
├── app.py                    # Streamlit entry point: sidebar + 5 tabs
├── paperpilot/
│   ├── config.py             # Settings (defaults, env vars, validation)
│   ├── health.py             # Ollama + model checks
│   ├── schemas.py            # Pydantic models (pages, chunks, summary, metadata)
│   ├── pdf_processing.py     # PDF → cleaned page text
│   ├── chunking.py           # sentence chunks, never crossing pages
│   ├── embeddings.py         # Ollama embeddings in batches
│   ├── vector_store.py       # ChromaDB collection per paper
│   ├── indexing.py           # extract → embed → store (complete or nothing)
│   ├── retrieval.py          # question → top-k chunks
│   ├── llm.py                # Ollama chat wrapper
│   ├── qa.py                 # grounded Q&A with the two gates
│   ├── summarizer.py         # structured summary, map-reduce for long papers
│   ├── bibliography.py       # metadata extraction + APA/MLA/IEEE/BibTeX
│   └── ui/                   # one module per tab
├── tests/                    # pytest suite + fakes
├── docs/
│   ├── decision.md           # decision log
│   └── flow.md               # code execution flow
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Limitations

- **Scanned PDFs** (images without a text layer) are not supported; there is no OCR. Password-protected PDFs are rejected.
- **One paper at a time.** Comparing several papers is planned.
- **Tables, figures and equations** are read only as extracted text, so their content may be incomplete.
- **The relevance threshold (0.35) is a starting value**, not yet calibrated on real questions.
- **Speed depends on your hardware.** Summaries of long papers can take several minutes on a laptop.
- **Summaries and references live in the browser session.** They disappear when you close the tab (the search index is saved on disk).
- Answer quality is limited by the local 8B model. Verify important claims.

---

## Roadmap

- [x] Phase 1: project setup, configuration, Ollama health check
- [x] Phase 2: PDF extraction and page-aware chunking
- [x] Phase 3: embeddings and ChromaDB index
- [x] Phase 4: grounded Q&A with page citations
- [x] Phase 5: structured summaries (map-reduce for long papers)
- [x] Phase 6: metadata extraction and citation formatting
- [x] Docker support
- [ ] Save summaries and references to disk
- [ ] Multi-paper library and comparison (similarities, limitations, cross-paper Q&A)
- [ ] Calibrate the relevance threshold on real questions

---

## License

See [LICENSE](LICENSE).
