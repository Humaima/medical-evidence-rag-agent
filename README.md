# Medical Evidence RAG Agent

A lightweight, safety-conscious Retrieval-Augmented Generation (RAG) demo that
answers medical questions **only** from a trusted evidence corpus (e.g. PubMed
abstracts), grounds every claim in citations, and **refuses to answer** when
the evidence is insufficient — instead of hallucinating.

Built to demonstrate applied skills in domain-specific RAG, structured LLM
output, and AI-safety-style confidence handling for a clinical-decision-support
context.

> ⚠️ **Not medical advice.** This is a research/portfolio demo, not a clinical
> tool. It is not validated for real patient care.

---

## Architecture

```
 User question
      │
      ▼
 FAISS retriever (top-k similarity search over embedded evidence chunks)
      │
      ├──► retrieval similarity score (numeric, model-independent)
      │
      ▼
 LLM (Groq, structured output via Pydantic)
      │  - answer text
      │  - citations (doc_id, pmid, title, snippet)
      │  - self_confidence (0-1)
      │  - is_answerable_from_evidence (bool)
      ▼
 Confidence gate  (retrieval_score AND llm_self_confidence must both
                    clear their thresholds)
      │
      ├── PASS ──► Answer + citations shown to user
      └── FAIL ──► "I don't have sufficient evidence to answer this."
```

The key design decision: **two independent confidence signals, combined with
an AND gate.** The retriever's similarity score is computed in plain Python
from vector distances — it doesn't trust the LLM at all. The LLM's own
self-reported confidence is a second, separate signal. Either one being weak
is enough to trigger a refusal. This biases the system toward caution, which
is the right default for anything touching medical information.

---

## Project structure

```
medical-rag-agent/
├── app.py                     # interactive CLI demo
├── streamlit_app.py           # Streamlit UI (journal-style design) for the same agent
├── requirements.txt
├── .env.example
├── data/
│   ├── sample_medical_corpus.json   # 10 original sample docs (see note below)
│   ├── raw_pubmed/             # raw PubMed exports (.txt/.csv/.xml), one per topic
│   └── converted/               # per-file JSON output of convert_pubmed.py
├── index/                     # FAISS index gets saved here (gitignored)
├── src/
│   ├── schemas.py              # Pydantic models: Citation, RAGAnswer, FinalResponse
│   ├── ingest.py                # builds the FAISS index from a corpus JSON
│   ├── confidence.py            # retrieval + LLM confidence gating / refusal logic
│   ├── rag_chain.py             # retrieval + structured LLM call, ties it together
│   ├── convert_pubmed.py        # converts a real PubMed text/CSV/XML export → corpus JSON
│   └── merge_corpus.py          # merges data/converted/*.json into one corpus file, deduped by PMID
└── tests/
    └── eval_refusal.py          # mini eval: does refusal fire on the right questions?
```

### A note on the sample data

`data/sample_medical_corpus.json` contains **10 short, original summaries**
I wrote myself (covering metformin, statins, aspirin, SGLT2 inhibitors, ACE
inhibitors, antibiotic stewardship, GLP-1 agonists, colorectal cancer
screening, anticoagulation, and vaccination) — not verbatim PubMed abstracts.
It exists purely so the pipeline runs end-to-end immediately. **Once you
upload your real PubMed dataset, swap it in** using the converter described
in Step 3 below — the rest of the pipeline (chunking, embedding, retrieval,
citation, confidence gating) works unchanged on real data.

---

## Step-by-step implementation guide (~1–2 hours)

### Step 0 — Prerequisites (5 min)
- Python 3.10+
- A Groq API key (embeddings run locally via sentence-transformers, so no OpenAI key is needed)
- `pip install -r requirements.txt`
- Copy `.env.example` → `.env` and paste in your key:
  ```bash
  cp .env.example .env
  # edit .env and set GROQ_API_KEY=gsk_...
  ```

### Step 1 — Understand the schemas first (10 min)
Open `src/schemas.py`. This is the contract the rest of the system is built
around:
- `Citation` — one evidence source backing a claim (doc id, PMID, title, short snippet)
- `RAGAnswer` — the raw structured output from the LLM, including its own
  `self_confidence` and `is_answerable_from_evidence`
- `FinalResponse` — what the user actually sees, after Python-side confidence
  gating is applied on top of `RAGAnswer`

Forcing the LLM into a typed schema (via LangChain's
`with_structured_output`) is what makes citation grounding checkable in code,
rather than something you just hope the prompt achieved.

### Step 2 — Build the FAISS index from the sample corpus (10 min)
```bash
python src/ingest.py --input data/sample_medical_corpus.json --out index/faiss_medical
```
Read `src/ingest.py` while it runs. Note:
- `RecursiveCharacterTextSplitter` chunks each document (800 chars, 120
  overlap) so retrieval works at the passage level, not whole-document level.
- Every chunk keeps `doc_id`, `pmid`, `title`, `url` in its metadata — this is
  what later lets us cite sources precisely instead of just "the corpus said so."

### Step 3 — (Once you have it) swap in your real PubMed dataset
`convert_pubmed.py` supports three export formats:

**Text** (PubMed website → Search → Save → Format: "Abstract (text)" → Create
File — the easiest route, and the only one of the three that reliably
includes abstract text):
```bash
python src/convert_pubmed.py --input "data/raw_pubmed/your_topic.txt" --format text --out data/converted/your_topic.json
```
Quote the `--input` path if the filename contains spaces (PowerShell/bash
otherwise split it into multiple arguments).

**CSV** (columns: `PMID, Title, Abstract`, optionally `Journal`, `URL`):
```bash
python src/convert_pubmed.py --input your_export.csv --format csv --out data/converted/your_topic.json
```

**Native PubMed XML** (`PubmedArticleSet`, from PubMed's "Save > XML" or
E-utilities `efetch`):
```bash
python src/convert_pubmed.py --input your_export.xml --format xml --out data/converted/your_topic.json
```

If you have multiple exports (e.g. one `.txt` file per topic), convert each
into `data/converted/`, then merge them into a single corpus, deduped by PMID:
```bash
python src/merge_corpus.py --input-dir data/converted --out data/my_corpus.json
```

Then rebuild the index against your real data:
```bash
python src/ingest.py --input data/my_corpus.json --out index/faiss_medical
```
Everything downstream (retrieval, citations, confidence gating) works
identically — nothing else needs to change.

> The text-format parser is heuristic (PubMed doesn't publish a formal spec
> for this export, and has changed the layout before — e.g. dropping the
> literal "Abstract" header line that used to precede the abstract body).
> Spot-check a few converted entries in `data/converted/*.json` after running
> it; if PubMed tweaks the layout again, only `convert_text()` in
> `src/convert_pubmed.py` needs updating.

### Step 4 — Understand the confidence/refusal logic (15 min)
Open `src/confidence.py`. This is the "impressive feature":
- `normalize_faiss_distance` converts FAISS's raw L2 distance into a 0–1
  similarity-like score.
- `decide()` applies the AND-gate: refuse if retrieval similarity is too low,
  **or** the LLM's self-confidence is too low, **or** the LLM itself flagged
  `is_answerable_from_evidence = False`.
- Thresholds (`RETRIEVAL_SCORE_THRESHOLD = 0.55`, `LLM_CONFIDENCE_THRESHOLD =
  0.60`) are demo defaults — in a real system you'd tune these against a
  labeled set of answerable/unanswerable questions (which is exactly what
  `tests/eval_refusal.py` is a small-scale version of).

### Step 5 — Wire it together and read the chain (15 min)
Open `src/rag_chain.py`. `MedicalRAGAgent.answer()`:
1. Retrieves top-k chunks + raw distances from FAISS
2. Formats them into a context block, each tagged with `doc_id`
3. Calls the LLM with `.with_structured_output(RAGAnswer)`
4. **Filters out any cited `doc_id` that wasn't actually retrieved** — a cheap
   guard against citation hallucination, independent of the confidence score
5. Passes the result to `confidence.decide()` for the final refusal decision

### Step 6 — Run the interactive demo (10 min)
```bash
python app.py
```
Try an in-corpus question:
```
Why is metformin used as first-line therapy for type 2 diabetes?
```
You should see a grounded answer with a citation to `doc001` and confidence
scores above threshold.

Now try an out-of-corpus question:
```
What is the recommended pediatric dosage of amoxicillin for otitis media?
```
You should see the refusal message, because nothing in the sample corpus
covers pediatric antibiotic dosing — this is the confidence-based refusal
mechanism working as intended, not an error.

### Step 7 — Run the Streamlit UI (5 min)
```bash
streamlit run streamlit_app.py
```
This wraps the exact same `MedicalRAGAgent` used by `app.py` in a browser UI —
no changes to the RAG logic itself, just a presentation layer on top. Notes
on `streamlit_app.py`:
- The design direction is deliberately "clinical journal / lab report," not a
  generic healthcare-app gradient: serif headlines echo journal mastheads,
  monospace numerals echo lab-data readouts, and a single ECG-trace SVG is
  the one signature graphic (used once, not as repeated decoration).
- The sidebar shows the active retrieval/confidence thresholds
  (`RETRIEVAL_SCORE_THRESHOLD`, `LLM_CONFIDENCE_THRESHOLD` from
  `src/confidence.py`) and a few clickable example questions.
- Answers render in a card that switches to a red "EVIDENCE INSUFFICIENT —
  REFUSED" style when `response.refused` is true, plus a confidence strip
  showing retrieval score, model self-confidence, and the combined score —
  the same three numbers the CLI prints, just laid out visually.
- `load_agent()` is wrapped in `@st.cache_resource` so the FAISS index and
  embedding model load once per session, not on every question.

### Step 8 — Run the mini evaluation (10 min)
```bash
python tests/eval_refusal.py
```
This runs 6 hand-labeled questions (3 that should be answered, 3 that should
be refused) and reports accuracy — a quick, screenshot-able way to
demonstrate the refusal mechanism is calibrated correctly. This is also the
natural place to extend if you want a more rigorous eval later (e.g. a larger
labeled set, or comparing against ground-truth answers).

### Step 9 — Polish for GitHub (15–20 min)
- Add a couple of screenshots (CLI and/or Streamlit UI, one answered case and
  one refused case) to this README.
- Consider adding a `LICENSE` file.
- Mention explicitly in the README (as this one does) that this is a demo,
  not a clinical tool — that framing itself is part of what the project is
  meant to demonstrate (safety-aware design, not just RAG mechanics).

---

## What this project demonstrates

- **Domain-specific RAG**: FAISS + LangChain retrieval over a medical evidence
  corpus, with chunking tuned for passage-level retrieval.
- **Citation grounding**: structured Pydantic output ties every claim to a
  specific `doc_id`/PMID, and cited IDs are validated against what was
  actually retrieved (not just trusted from the LLM's output).
- **Confidence handling**: two independent signals (retrieval similarity +
  LLM self-confidence), combined with an AND gate rather than either alone.
- **Safety-aware refusal**: the system explicitly prefers "I don't know" over
  a fabricated answer when evidence is insufficient — directly relevant to
  hallucination mitigation in clinical decision-support contexts.
- **Two interfaces, one RAG core**: a CLI (`app.py`) and a styled Streamlit UI
  (`streamlit_app.py`) both call the same `MedicalRAGAgent`, showing the
  retrieval/generation logic is decoupled from presentation.

## Extending this further
- Swap the Groq model (`model_name` in `MedicalRAGAgent`, e.g.
  `llama-3.3-70b-versatile` vs `llama-3.1-8b-instant`) and compare refusal
  calibration. Groq's lineup changes often — list what's currently active
  with `python -c "from groq import Groq; [print(m.id) for m in Groq().models.list().data if m.active]"`.
- Add a re-ranker (e.g. cross-encoder) between FAISS retrieval and the LLM
  call to improve retrieval precision before the confidence gate.
- Log every query/response/confidence tuple to build a labeled dataset for
  threshold tuning.
- Add source-recency or study-type weighting (e.g. prefer RCTs/meta-analyses
  over case reports) if your real PubMed corpus includes that metadata.
