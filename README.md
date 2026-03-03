# Construction Arbitration — Multi-Agent Document Intelligence Pipeline

An LLM-driven, multi-phase pipeline that analyzes construction arbitration documents end-to-end: from raw OCR text to a structured case-level analytical report. Built with **Gemini** (via Vertex AI), **Pydantic v2** schemas, and a resumable, parallelizable orchestrator.

---

## Project Workflow

The pipeline processes ~50 segmented arbitration documents through four sequential phases. Phases 2 and 3 run **in parallel** after Phase 1 completes; Phase 4 depends on all prior outputs.

```
segments/                        (raw OCR text per document)
   ↓
Phase 1 — Index & Structured Extraction
   ↓
outputs/index.json               (structured metadata per segment)
   ├── Phase 2 — Document Type Taxonomy & Classification
   │     Step 1: Corpus-level taxonomy generation   → type_taxonomy.json
   │     Step 2: Per-segment classification          → type_classifications.json
   │
   └── Phase 3 — Claim Head Discovery & Mapping
         Step 1: Corpus-level claim head discovery   → claim_heads.json
         Step 2: Per-segment claim mapping            → claim_classifications.json
            ↓
Phase 4 — Case-Level Reasoning & Report              → case_analysis.md
```

### Phase Summaries

| Phase | Purpose | Model | Output |
|-------|---------|-------|--------|
| **1** | Extract parties, dates, amounts, dispute signals from OCR text | `gemini-2.0-flash` | `index.json` |
| **2** | Build a document-type taxonomy, then classify each segment | `gemini-2.0-flash` | `type_taxonomy.json`, `type_classifications.json` |
| **3** | Discover financial claim heads from the corpus, then map segments to claims | `gemini-2.0-flash` | `claim_heads.json`, `claim_classifications.json` |
| **4** | Generate a grounded, section-by-section case analysis report | `gemini-2.5-pro` | `case_analysis.md` |

All LLM calls use **schema-enforced JSON mode** with automatic retry on malformed outputs. Per-segment processing is fully parallelizable via `ThreadPoolExecutor`.

---

## Project Structure

```
├── orchestrator.py          # CLI entry point — runs the full pipeline
├── pipeline/
│   ├── phase1_index.py      # Phase 1: structured extraction
│   ├── phase2_types.py      # Phase 2: taxonomy + classification
│   ├── phase3_claims.py     # Phase 3: claim discovery + mapping
│   └── phase4_analysis.py   # Phase 4: case-level report
├── models/
│   └── schemas.py           # Pydantic v2 data contracts
├── utils/
│   ├── llm.py               # GeminiClient wrapper (Vertex AI)
│   └── file_io.py           # Segment discovery, I/O helpers
├── segments/                # Input: one folder per document segment
├── outputs/                 # Output: JSON artifacts + final report
├── pyproject.toml           # Project metadata & dependencies
├── PLAN.md                  # Detailed design document
└── reflection.md            # Scaling & design reflections
```

---

## Setup & Installation

### Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** (recommended) or pip
- A **Google Cloud** project with the Vertex AI API enabled
- A **service account** JSON key with Vertex AI permissions

### 1. Clone the repository

```bash
git clone <repo-url>
cd agentic_arbitration
```

### 2. Install dependencies

Using **uv** (recommended):

```bash
uv sync
```

Or with pip:

```bash
pip install .
```

For development tools (ruff, pytest):

```bash
uv sync --extra dev
```

### 3. Configure credentials

Place your Vertex AI service account key at the project root:

```
vertex-ai-credentials.json
```

Or set environment variables (via `.env` or shell):

```bash
GOOGLE_APPLICATION_CREDENTIALS=vertex-ai-credentials.json
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=us-central1
```

### 4. Add document segments

Place OCR text files under `segments/`, one folder per document. Each folder should contain a text file with the raw OCR content.

---

## Usage

Run the full pipeline:

```bash
uv run python orchestrator.py
```

### CLI Options

| Flag | Description |
|------|-------------|
| `--force` | Re-run phases even if outputs already exist |
| `--no-parallel` | Run Phase 2 and Phase 3 sequentially |
| `--phase 1 2 3 4` | Run only specific phase(s), e.g. `--phase 1 4` |

### Examples

```bash
# Force re-run everything
uv run python orchestrator.py --force

# Re-run only Phase 4 (report generation)
uv run python orchestrator.py --phase 4 --force

# Run sequentially (useful for debugging)
uv run python orchestrator.py --no-parallel
```

The pipeline is **resumable** — it skips phases whose output files already exist unless `--force` is passed.

---

## Outputs

All artifacts are written to `outputs/`:

| File | Description |
|------|-------------|
| `index.json` | Structured metadata for each document segment |
| `type_taxonomy.json` | Corpus-derived document type taxonomy |
| `type_classifications.json` | Per-segment type classifications |
| `claim_heads.json` | Discovered financial claim heads with sub-heads |
| `claim_classifications.json` | Per-segment claim mappings with relevance scores |
| `case_analysis.md` | Final case-level analytical report |
