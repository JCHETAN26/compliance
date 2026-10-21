# Build Handoff: Multi-Agent Compliance & Document Intelligence Workspace

> **To Claude:** You are building this project end to end. Read this whole document before writing code. Work in the phases in Section 14, and do not start a phase until the previous phase's acceptance criteria pass. If something in this spec is impossible or clearly wrong, stop and explain before deviating. **Never fabricate results.** Every metric in the README must come from a script you actually ran, and the exact command that produced it must be recorded.

---

## 1. What we are building

A **full-stack, multi-agent compliance review workspace**.

1. A user uploads messy business documents: contracts (PDF/DOCX), scanned PDFs, and spreadsheets such as a vendor register (XLSX/CSV).
2. They pick a **rule pack**, a versioned set of compliance rules.
3. A team of specialized agents extracts the relevant facts, checks every rule against every relevant document, reasons across documents, and verifies each claim.
4. The output is a **compliance report**. Every finding links to the exact source passage, highlighted in the original document.

The **ML core** is a **rule-checking model we train ourselves**. Given a rule and a document, it predicts *satisfied / violated / not mentioned* and returns the evidence sentences. It runs in a **cascade** with an LLM: the small model handles confident cases cheaply, and uncertain cases escalate to the LLM.

The app also ships its **own MCP server**, so any MCP client (Claude Desktop, Claude Code, Cursor) can search documents, run checks and fetch reports.

### Real-world problem
Legal, compliance and operations teams in regulated industries spend much of their time manually reading contracts, policy updates and audit evidence to verify compliance. That is slow, expensive and error-prone. This product automates first-pass review with verifiable citations, and keeps a human in the loop for final decisions.

### MVP domain (deliberately narrow)
**NDA and vendor-contract review.** This matches the public training data (ContractNLI, CUAD; see Section 7), so the model's results are credible. The rule system is generic, so other domains can be added later.

### Flagship demo scenario (must work end to end)
1. Upload `vendor_register.xlsx`, listing about 10 vendors with columns: vendor name, data access level, contract file name, renewal date.
2. Upload the matching NDA and contract files, including at least one scanned PDF.
3. Select the rule pack "Vendor Confidentiality Review."
4. The agents:
   - check each NDA against the rules;
   - cross-reference the spreadsheet, e.g. "every vendor with *high* data access must have an NDA that restricts sharing with third parties and requires return or destruction of confidential information";
   - flag vendors whose contract is missing or non-compliant.
5. The report shows per-vendor and per-rule status, with clickable citations that open the document with the evidence highlighted.

---

## 2. Non-negotiable requirements

1. **Full stack:** a real web UI, a backend API, a database, background workers, and auth. No placeholder screens.
2. **Multi-agent:** at least 6 specialized agents orchestrated as a graph (LangGraph), with typed shared state.
3. **Own MCP server:** a real server built with the official MCP Python SDK, exposing the tools and resources in Section 8. It must work from Claude Desktop.
4. **Own trained model:** a fine-tuned rule-checking model with a reproducible training pipeline, honest evaluation against baselines, and a model service that serves it.
5. **Verifiable citations:** every finding links to a document, page and character span (plus a bounding box for PDFs), and the UI highlights it.
6. **Evaluation:** offline evals for the model, retrieval and end-to-end reports, wired into CI with thresholds.
7. **Runs locally with one command:** `docker compose up`. Seed data and a demo workspace load automatically.
8. **LLM-provider agnostic:** supports Anthropic and OpenAI through env vars, a local model via Ollama, and a **mock LLM** for tests and offline development.
9. **Honest README:** results tables come from real runs, with a limitations section.

---

## 3. Tech stack

| Layer | Choice |
|---|---|
| Frontend | Next.js (App Router) + TypeScript, Tailwind, shadcn/ui, TanStack Query, `react-pdf` (pdf.js) for the document viewer with highlight overlays |
| Backend API | Python 3.11+, FastAPI (async), Pydantic v2, SQLAlchemy 2.0 (async) + Alembic |
| Agents | LangGraph (typed state, checkpointing, human-in-the-loop interrupts) |
| Jobs | Redis + `arq` (or Celery) worker for ingestion and review runs |
| Streaming | Server-Sent Events (SSE) for live agent progress |
| Database | PostgreSQL 16 + `pgvector` (HNSW index) + Postgres full-text search (BM25-style) |
| File storage | Local volume through a storage interface (S3/MinIO-compatible adapter optional) |
| Ingestion | PyMuPDF (text + layout + bboxes), Tesseract or docTR OCR fallback for scanned pages, python-docx, openpyxl/pandas |
| Embeddings | Sentence-transformers model (e.g. `BAAI/bge-small-en-v1.5`) served locally |
| Reranker | Cross-encoder (e.g. `BAAI/bge-reranker-base`), optional, behind a flag |
| ML training | PyTorch, Hugging Face Transformers + Datasets, scikit-learn baselines; runs on CPU for smoke tests and on a single T4 (Colab) for real training |
| Model serving | A separate FastAPI `model-service` with dynamic batching; ONNX Runtime export + int8 quantization for CPU |
| MCP | Official `mcp` Python SDK (FastMCP), stdio + streamable HTTP transports |
| Observability | Structured JSON logs, OpenTelemetry traces for agent steps, per-run token, cost and latency accounting |
| Testing | pytest (unit + integration), Playwright (E2E), eval suite (Section 10) |
| CI | GitHub Actions: lint, typecheck, tests, small eval subset with thresholds, Docker build |
| Quality | ruff, mypy (strict on core packages), eslint, `tsc --noEmit`, pre-commit |

---

## 4. Repository structure

```
compliance-workspace/
├── apps/
│   ├── web/                    # Next.js frontend
│   ├── api/                    # FastAPI backend (REST + SSE)
│   ├── worker/                 # Background jobs (ingestion, review runs)
│   ├── model-service/          # Serves the trained rule-check model
│   └── mcp-server/             # Our MCP server
├── packages/
│   ├── core/                   # Shared domain models, DB, storage, config
│   ├── ingestion/              # Parsers, OCR, layout-aware chunking
│   ├── retrieval/              # Hybrid search, RRF fusion, reranking
│   ├── agents/                 # LangGraph graph, agents, tools, prompts
│   └── llm/                    # Provider abstraction (anthropic/openai/ollama/mock)
├── ml/
│   ├── data/                   # Download + preprocessing scripts (no raw data in git)
│   ├── train/                  # Training scripts + configs
│   ├── eval/                   # Model evaluation, calibration, cascade analysis
│   ├── export/                 # ONNX export, quantization, latency benchmark
│   └── notebooks/              # Colab training notebook (thin wrapper over scripts)
├── evals/                      # Retrieval + end-to-end report evals, golden sets
├── rule_packs/                 # YAML rule packs (versioned)
├── seed/                       # Demo workspace generator (vendor register, sample docs)
├── docker-compose.yml
├── Makefile
└── README.md
```

---

## 5. Data model (Postgres)

- `users` (id, email, password_hash, created_at)
- `workspaces` (id, name, owner_id, created_at)
- `workspace_members` (workspace_id, user_id, role: owner | reviewer | viewer)
- `documents` (id, workspace_id, filename, mime_type, storage_key, sha256, status: uploaded | processing | ready | failed, page_count, doc_type: contract | spreadsheet | policy | other, error, created_at)
- `pages` (id, document_id, page_number, width, height, is_ocr)
- `chunks` (id, document_id, page_id, text, char_start, char_end, bbox_json, section_path, token_count, embedding vector(384), tsv tsvector). Indexes: HNSW on `embedding`, GIN on `tsv`.
- `table_rows` (id, document_id, sheet_name, row_index, data_json). Spreadsheets are stored as structured rows, not only as text.
- `extracted_entities` (id, document_id, type such as party, effective_date, governing_law, term, clause:<type>, value_json, chunk_id, confidence)
- `rule_packs` (id, workspace_id nullable for global packs, name, version, description, created_at)
- `rules` (id, rule_pack_id, key, title, description, hypothesis_text, scope: per_document | cross_document, applies_to_doc_types, severity: low | medium | high, model_supported: bool)
- `review_runs` (id, workspace_id, rule_pack_id, status, started_at, finished_at, llm_provider, totals_json covering tokens, cost, latency and cascade stats)
- `findings` (id, review_run_id, rule_id, document_id nullable, subject such as a vendor name, status: satisfied | violated | not_mentioned | needs_review, confidence, decided_by: model | llm | human, rationale, verified: bool, created_at)
- `citations` (id, finding_id, chunk_id, quote, char_start, char_end, page_number, bbox_json, support_score)
- `agent_events` (id, review_run_id, agent, event_type, payload_json, ts). Persisted and also streamed over SSE.
- `human_overrides` (id, finding_id, user_id, new_status, comment, created_at)

---

## 6. Ingestion and retrieval

**Ingestion pipeline** (worker job, idempotent, keyed by file sha256):
1. Detect type. For PDFs, extract text with PyMuPDF, keeping per-span bounding boxes. If a page has almost no text layer, rasterize it and OCR it, then map OCR words back to bboxes.
2. For DOCX, extract paragraphs with style info and compute character offsets.
3. For XLSX/CSV, store rows in `table_rows`, auto-detect the header row, and create one text chunk per row ("Vendor: Acme; Data access: High; Contract: acme_nda.pdf …") so rows are searchable too.
4. Do **layout-aware chunking**: split on headings, numbered clauses and paragraphs, targeting about 200–400 tokens with small overlap. Never split mid-sentence. Keep `char_start/char_end` and bboxes so citations can be highlighted exactly.
5. Embed chunks in batches and store embeddings in pgvector (HNSW). Populate the full-text `tsvector`.
6. Emit progress events, and set status to `ready` or `failed` with an error message.

**Hybrid retrieval** (`packages/retrieval`):
- Run vector search (pgvector HNSW) and keyword search (Postgres full-text, `ts_rank_cd`) in parallel, then combine them with **reciprocal rank fusion**. The cross-encoder reranker is optional.
- Make `ef_search`, top-k and fusion weights configurable. Include a script that sweeps HNSW `m`, `ef_construction` and `ef_search` and reports recall@k against latency, written to `evals/results/`.

---

## 7. The trained model: rule checker (the ML core)

### 7.1 Task
Input: a **rule hypothesis** (e.g. "The receiving party may share confidential information with its employees") and a **contract**.
Output:
- a document-level label: `satisfied` (entailment), `violated` (contradiction) or `not_mentioned`;
- **evidence spans**: the sentences or clauses that support the label;
- a calibrated confidence score.

### 7.2 Data
- **Primary: ContractNLI** (Koreeda & Manning, 2021). 607 NDAs, 17 hypotheses, document-level labels (entailment / contradiction / not mentioned), and annotated evidence spans. Download it with a script in `ml/data/`, and keep the official train/dev/test splits.
- **Secondary: CUAD** (Hendrycks et al., 2021). 510 commercial contracts with expert labels for 41 clause types. Use it for the clause-extraction evaluation and optional pretraining.
- **Before using either dataset, check its license and record it in `ml/data/LICENSES.md`.** Do not commit raw data to git.
- Map the 17 ContractNLI hypotheses to rules in `rule_packs/nda_standard.yaml` (`model_supported: true`).

### 7.3 Model design (span-based, handles long documents)
1. Split each contract into sentence or clause spans, using the same segmentation as ingestion.
2. **Evidence head:** a cross-encoder over (hypothesis, span) predicts whether the span is evidence.
3. **Label head:** take the top-k evidence spans (k=3–5) with the hypothesis and predict the 3-way label.
4. **Backbone:** `microsoft/deberta-v3-base` (try `answerdotai/ModernBERT-base` as an ablation). A single model with two heads, or two fine-tuned models, is fine; choose based on dev results and document why.
5. Handle class imbalance (`not_mentioned` dominates), using weighted loss or sampling, and report per-class results.

### 7.4 Baselines (all required, all evaluated on the same test split)
1. Majority class.
2. TF-IDF + logistic regression over (hypothesis, retrieved spans).
3. Zero-shot LLM: the same retrieval, with an LLM judging the label and evidence. Record cost and latency.
4. Our fine-tuned model.

### 7.5 Metrics
- Label: accuracy and **macro-F1**, plus per-class precision, recall and F1, and a confusion matrix.
- Evidence: precision, recall and F1 at the span level, plus mean average precision over spans.
- **Calibration:** expected calibration error (ECE) before and after temperature scaling, with a reliability diagram.
- Efficiency: CPU latency per (rule, document) pair at p50 and p95 for PyTorch FP32, ONNX FP32 and ONNX int8, plus model size, and the accuracy change from quantization.
- **Generalization experiment:** hold out 3–4 hypotheses entirely during training and report performance on those unseen rules. This tells us how well the model handles new custom rules.

### 7.6 Cascade (model + LLM)
- If the model's calibrated confidence is at least threshold τ, accept its decision (`decided_by=model`). Otherwise, escalate to the LLM agent (`decided_by=llm`).
- Rules with `model_supported: false` (custom rules unlike the training hypotheses) always go to the LLM.
- Choose τ on the **dev** split, never test. Report a **coverage vs. accuracy curve** (fraction handled by the model vs. overall accuracy), and the final cascade's accuracy, cost per document and latency against LLM-only.
- Save everything to `ml/eval/results/*.json`, plus plots as PNGs; the UI evals page reads the JSON.

### 7.7 Reproducibility
- `make train` (configurable; a small CPU smoke config runs in CI in about 2 minutes), `make eval-model`, `make export-model`.
- Fixed seeds, configs in YAML, and every run's results saved with config hash, git commit and timestamp.
- A Colab notebook that clones the repo and runs the same scripts on a T4.

### 7.8 Model service
- `apps/model-service`: `POST /v1/check` takes `{hypothesis, spans[]}` and returns `{label, confidence, evidence:[{span_id, score}]}`. It also exposes `GET /health` and `GET /metadata` (model version and metrics).
- Dynamic micro-batching (collect requests for up to N ms or a maximum batch size). The model loads the ONNX int8 build by default, with an FP32 fallback flag.
- If no trained weights exist, it starts in a clearly labeled "untrained fallback" mode so the app still runs, and the UI shows a banner.

---

## 8. Multi-agent system (LangGraph)

**Shared typed state:** `run_id`, `workspace_id`, `rule_pack`, `documents`, `extracted_entities`, `tasks` (rule × subject), `findings`, `citations`, `budget` (tokens/cost), `errors`.

### Agents
1. **Planner / Orchestrator.** Reads the rule pack and the inventory of available documents, then expands it into concrete check tasks:
   - per-document rules become (rule × document);
   - cross-document rules become (rule × subject), e.g. each vendor row in the register.

   It decides the order, runs tasks in parallel where it's safe, and enforces the budget.
2. **Extraction agent.** Uses structured LLM outputs (Pydantic schemas) to extract key entities and clause types (parties, dates, term, governing law, CUAD-style clauses), each with a chunk citation. It links spreadsheet rows to documents by filename or party-name matching, using fuzzy matching with a confidence score.
3. **Retrieval agent.** For each task, runs hybrid retrieval scoped to the relevant documents and returns candidate spans.
4. **Rule-check agent.** Calls the **model service** for model-supported rules and applies the cascade. When it escalates, it calls the LLM with the hypothesis and retrieved spans and asks for a structured verdict plus evidence span IDs.
5. **Cross-document reasoning agent.** Handles multi-step rules. Example: "for each vendor with high data access → find its contract → required NDA clauses must be satisfied → contract must not have expired relative to the renewal date." It combines findings into subject-level results and explains each reasoning step.
6. **Citation verifier agent.** Checks every finding's citations:
   - the quoted text must exist verbatim at the stored offsets;
   - the span must actually support the claim, scored by an NLI/LLM support check.

   Unsupported findings are downgraded to `needs_review` and never silently kept.
7. **Report writer agent.** Produces an executive summary, a per-rule table, a per-subject table, risk ranking by severity, and recommended actions. It may only state facts that are backed by verified findings.

### Orchestration requirements
- The graph supports parallel fan-out over tasks, retries with backoff on tool and LLM errors, and checkpointing so a run can be resumed.
- **Human-in-the-loop:** the graph can pause on `needs_review` findings when the setting is enabled. A reviewer overrides the result in the UI, and the run continues.
- Every agent step emits `agent_events` (start, tool call, result, error, tokens, cost), streamed to the UI over SSE and persisted.
- **Prompt-injection safety:** document text is always treated as untrusted data. Use delimiting, never execute instructions found in documents, and add a test with a contract containing "ignore previous instructions…" to confirm the findings are unaffected.

---

## 9. MCP server (`apps/mcp-server`)

Build with the official `mcp` Python SDK (FastMCP), supporting the **stdio** and **streamable HTTP** transports. Authenticate with a per-user API token that is created in the web app's Settings page.

### Tools
- `list_workspaces()`
- `list_documents(workspace_id)`
- `upload_document(workspace_id, filename, content_base64)`: stdio only, with a size limit
- `search_documents(workspace_id, query, top_k=8)` returns spans with citations
- `check_rule(workspace_id, document_id, hypothesis)` returns a cascade verdict with evidence
- `run_review(workspace_id, rule_pack_id)` returns a run_id (asynchronous)
- `get_run_status(run_id)`
- `get_report(run_id, format="markdown"|"json")`
- `get_citation(citation_id)` returns the quote, document, page and surrounding context

### Resources
- `rulepack://{id}`: rule pack YAML
- `report://{run_id}`: the latest report as markdown

### Requirements
- The MCP tools call the **same service layer** as the REST API; no duplicated logic.
- Include `apps/mcp-server/README.md` with a ready-to-paste Claude Desktop config, plus an integration test that runs the server over stdio and calls each tool.

---

## 10. Evaluation suite (`evals/`)

1. **Model evals:** Section 7.5. Results are written to `ml/eval/results/`.
2. **Retrieval eval:** use ContractNLI evidence spans as gold. For each (hypothesis, document), measure recall@k and MRR for vector-only, keyword-only, hybrid (RRF) and hybrid with reranking. Include an HNSW parameter sweep reporting recall@k against latency.
3. **End-to-end report eval:** a golden set of about 30–50 review tasks built from held-out ContractNLI test documents, plus the seeded vendor scenario with hand-written expected outcomes. Measure:
   - finding accuracy against expected status;
   - **citation precision**: the share of citations whose span is gold evidence or judged supporting;
   - the unsupported-claim rate in the final report;
   - faithfulness and context precision with Ragas, or an equivalent LLM-judge implementation if Ragas is impractical; document which one was used;
   - cost, tokens and latency per run.
4. **CI gate:** a small, fast subset (mock LLM plus a tiny model, or cached model outputs) runs on every PR, with thresholds checked in `evals/thresholds.yaml`. The full eval runs with `make eval-all`.
5. **Evals page in the UI:** show the latest model metrics, the cascade coverage/accuracy curve, retrieval comparison and end-to-end results, read from the results JSON files.

---

## 11. Backend API (FastAPI)

Base path `/api/v1`. All endpoints are authenticated (JWT access token in an httpOnly cookie) and scoped to workspace membership.

- Auth: `POST /auth/register`, `POST /auth/login`, `POST /auth/logout`, `GET /auth/me`
- Workspaces: `GET/POST /workspaces`, `GET/PATCH/DELETE /workspaces/{id}`, `POST /workspaces/{id}/members`
- Documents: `POST /workspaces/{id}/documents` (multipart, multiple files), `GET /workspaces/{id}/documents`, `GET /documents/{id}`, `GET /documents/{id}/file` (stream original), `GET /documents/{id}/chunks`, `DELETE /documents/{id}`
- Rule packs: `GET/POST /rule-packs`, `GET/PUT /rule-packs/{id}`, `POST /rule-packs/import` (YAML), `GET /rule-packs/{id}/export`
- Reviews: `POST /workspaces/{id}/reviews` (start run), `GET /reviews/{run_id}`, `GET /reviews/{run_id}/events` (**SSE**), `GET /reviews/{run_id}/findings`, `POST /findings/{id}/override`, `POST /reviews/{run_id}/resume`, `GET /reviews/{run_id}/report?format=md|pdf|json`
- Search: `POST /workspaces/{id}/search`
- Citations: `GET /citations/{id}`
- Evals: `GET /evals/latest`
- Settings: `GET/POST /settings/api-tokens`, `GET/PUT /settings/llm`

Use an OpenAPI schema with generated TypeScript types for the frontend (`openapi-typescript`), and consistent error responses.

---

## 12. Frontend (Next.js) — pages and UX

1. **Auth:** sign up and log in.
2. **Workspaces list:** create a workspace; cards show document count and last review status.
3. **Workspace → Documents:** drag-and-drop multi-file upload with per-file progress and processing status (live updates). A table shows type, pages, OCR flag, status and errors. Clicking a document opens the viewer.
4. **Document viewer:** PDF rendering with a text-layer search, a sidebar of extracted entities and clauses, and click-to-highlight. Spreadsheets render as a table.
5. **Rule packs:** a list and editor for rules (title, hypothesis text, scope, severity, document types), YAML import/export, and a badge showing whether the trained model supports the rule.
6. **New review → Live run view:** an agent timeline streamed over SSE, showing each agent's steps, tool calls and errors. A progress grid of rules × subjects fills in live, with running cost and token counters and a pause/resume control.
7. **Report view:**
   - **Content:** a summary, a filterable findings table (status chips, severity, confidence, `decided_by` badge for model/LLM/human, verified ✓), and a per-subject view, e.g. per vendor.
   - **Citations:** clicking a citation opens the viewer, scrolled to and highlighting the exact span.
   - **Review actions:** override with a comment.
   - **Export:** Markdown, PDF or JSON.
8. **Evals dashboard:** the charts from Section 10.
9. **Settings:** LLM provider and model, cascade threshold τ (default from evaluation), API tokens for MCP, and a budget cap per run.

UX bar: clean, responsive, keyboard-accessible, real empty, loading and error states, and no lorem ipsum.

---

## 13. Ops, security and DevEx

- `docker compose up` starts: `web`, `api`, `worker`, `model-service`, `mcp-server` (HTTP), `postgres` (with pgvector), and `redis`. Healthchecks on every service.
- `make seed` creates a demo user (`demo@example.com` / `demo1234`), a demo workspace, the vendor register, sample NDAs (from ContractNLI test documents, with at least one rendered as a scanned image PDF to exercise OCR), and both rule packs (`nda_standard`, `vendor_confidentiality_review`).
- `.env.example` covers every variable. Without an LLM key, the app runs with the mock LLM and shows a banner.
- Enforce file type and size limits, sanitize filenames, rate-limit uploads and review starts, and never log document contents at info level.
- Trace each review run with OpenTelemetry, and attach the run_id to every log line.
- The README includes:
  - a screenshot or GIF of the live run and report;
  - an architecture diagram (Mermaid);
  - quickstart instructions;
  - an MCP setup guide;
  - a **Results** section with real numbers from the eval scripts (model vs. baselines, cascade, retrieval, end-to-end);
  - a **Limitations** section.

---

## 14. Build phases with acceptance criteria

Work through these in order. At the end of each phase, run the tests and summarize what works and what doesn't.

**Phase 0: Skeleton**
Monorepo, docker compose, DB migrations, auth, workspaces CRUD, CI pipeline, lint and typecheck.
✅ `docker compose up`; sign up, create a workspace; CI green.

**Phase 1: Ingestion + viewer**
Upload, the parsers (PDF with layout, OCR fallback, DOCX, XLSX/CSV), chunking with offsets and bboxes, embeddings, the documents page and the viewer with highlighting.
✅ Seed documents ingest. A scanned PDF gets OCR text. Clicking a chunk highlights the right region. Tests cover each parser.

**Phase 2: Retrieval**
Hybrid search, the search UI and the retrieval eval script.
✅ `make eval-retrieval` produces recall@k for vector, keyword and hybrid. Hybrid beats at least one single method; report honestly if it doesn't.

**Phase 3: ML model**
Data scripts, the baselines, training, evaluation, calibration, ONNX export and int8 quantization, the model service, and the Colab notebook.
✅ Model results beat the TF-IDF baseline on macro-F1 on the ContractNLI test split. The results JSON and plots are generated, and the model service answers `/v1/check` with batching. If the target isn't met, report the actual numbers and analysis; do not tune on test.

**Phase 4: Agents**
The LangGraph graph with all 7 agents, the cascade, cross-document reasoning, the verifier, the report writer, SSE events, the live run UI, the report UI and overrides.
✅ The flagship vendor demo runs end to end with the mock LLM and with a real LLM. Every finding has a verified citation or is marked `needs_review`. The prompt-injection test passes.

**Phase 5: MCP server**
All tools and resources, auth tokens, the stdio integration test and the Claude Desktop guide.
✅ From Claude Desktop: search documents, run a review, and fetch the report.

**Phase 6: Evals + polish**
The end-to-end eval suite, the CI gate, the evals dashboard, report export, the README with real results, screenshots and limitations, and Playwright E2E for upload → review → report → citation click.
✅ `make eval-all` passes thresholds. Playwright E2E is green in CI. The README results match the results files.

---

## 15. Coding standards

- Typed everywhere (Pydantic models, mypy, strict TypeScript). Services are separated from routes. Configuration is set only through env vars.
- Every agent prompt lives in a versioned file under `packages/agents/prompts/`, and its version is recorded on each finding.
- Every external call (LLM, model service) has timeouts, retries and cost accounting.
- Tests are written alongside the features: unit tests for parsers, retrieval, cascade logic and verifier logic; integration tests for the API and the graph using the mock LLM.
- Commits are small, with clear messages.

## 16. Out of scope (for now)
SSO, billing, multi-tenant cloud deployment, real-time co-editing, and languages other than English. Mention these as future work in the README.

## 17. Definition of done
- [ ] `docker compose up && make seed` gives a working app at `localhost:3000` with the demo workspace.
- [ ] Flagship vendor review runs end to end, with live agent streaming and a report whose citations highlight in the viewer.
- [ ] Trained rule-check model with baselines, calibration, cascade analysis and int8 CPU serving; results reproducible with `make` targets.
- [ ] MCP server works from Claude Desktop, with all tools tested.
- [ ] Eval suite with a CI gate; evals dashboard in the UI.
- [ ] README with architecture, real results, screenshots, MCP guide and limitations.