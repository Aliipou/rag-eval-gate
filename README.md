# rag-eval-gate

**Suomeksi lyhyesti:** Tämä on hakupohjainen kysymys-vastauspalvelu, jonka
pääasia ei ole chatbotti vaan yhdyskäytävä: jos vastausten laatu (perusteltavuus,
lähdeviittausten tarkkuus, oikea kieltäytyminen) heikkenee, CI epäonnistuu eikä
muutosta voi yhdistää. Jokaisen vastauksen mukana tulee läpinäkyvyystietue
(malli, lähteet, todennus). Kaikki data on julkista Finlex- ja
työministeriöaineistoa; ei ole EU:n tekoälyasetuksen mukaisuusväite, vain
saman ajatuksen mukaan mallinnettu läpinäkyvyys.

A retrieval-augmented answering service whose CI build fails when answer
quality regresses. The product here is the evaluation gate, not the chatbot —
almost every public RAG demo has no evaluation at all; this one measures
groundedness, citation precision, and refusal correctness against a golden
set on every push, with committed thresholds that a pull request cannot merge
past.

## Live demo

- **API**: https://rag-eval-gate.vercel.app (`GET /api/health`, `POST /api/answer`)
- **Frontend**: not yet deployed — see "What is not built yet" below.

Try it directly:

```bash
curl -X POST https://rag-eval-gate.vercel.app/api/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "What does Section 1 of the Constitution say?"}'
```

## The evaluation gate

`eval/golden.yaml` holds 48 independently-authored cases — 34 where the
corpus genuinely contains the answer, 14 where it deliberately does not and
the correct behaviour is to refuse. Each case's expected chunk ids and
expected behaviour were determined by reading the source PDFs directly, not
by running the pipeline and recording what it happened to do.

`eval/run_eval.py` measures three metrics against that golden set and exits
non-zero if any falls below the threshold committed in `eval/thresholds.yaml`:

| Metric | Threshold | Baseline (48 cases, 129 claims) |
|---|---|---|
| Groundedness — every claim maps to a retrieved chunk | 0.95 | 1.0000 |
| Citation precision — cited chunks actually contain the claim | 0.90 | 1.0000 |
| Refusal correctness — refuses exactly when it should | 0.85 | 0.9167 (44/48) |

CI (`.github/workflows/ci.yml`) runs this against **recorded LLM responses**
(`eval/fixtures/recorded_responses.json`), so the gate is deterministic and
costs nothing to run on every push. A separate nightly workflow
(`.github/workflows/nightly.yml`) re-runs it live.

The 4 known-missing refusal cases all trace to TF-IDF retrieval recall, not
the verification or generation logic — documented in the eval fixtures
rather than quietly excluded.

## How answering works

1. **Retrieval** (`rag/retrieval.py`) — TF-IDF cosine similarity over a
   corpus of real public-sector text (below), not a demo/placeholder corpus.
2. **Generation** (`rag/generation.py`) — every claim must be paired with the
   chunk id the model says supports it.
3. **Verification** (`rag/verification.py`) — a pure, deterministic function
   is the sole judge of whether a citation actually holds up. An unsupported
   claim is dropped, never rewritten. If nothing survives, the response says
   so explicitly rather than improvising an answer.
4. **Transparency record** — every response carries model id, prompt hash,
   retrieved chunk ids with similarity scores, and per-claim verification
   outcome, exportable as JSON (`GET /api/transparency/{prompt_hash}`).

Also exposed as an MCP server (`mcp_server/server.py`) with `search_corpus`
and `get_chunk` tools, so the corpus is callable from any MCP client.

## Corpus

Two real Finnish public-sector documents, 158 chunks total — not synthetic
filler:

1. **Constitution of Finland (731/1999)**, official Finlex translation, 128
   sections.
2. **Guide to the Employment Contracts Act**, published by the Finnish
   Ministry of Economic Affairs and Employment — a plain-language guide
   *about* the Act, not the statute text itself, and the golden set has
   cases that specifically probe that distinction.

Full sourcing, licensing notes, and ingestion details in
[`corpus/SOURCES.md`](corpus/SOURCES.md).

## What is implemented / what is not

- **Retrieval is TF-IDF, not pgvector.** The project brief calls for
  PostgreSQL + pgvector; there's no live Postgres in this build environment,
  so `rag/retrieval.py` defines the `VectorIndex` interface both a real
  pgvector-backed index and this TF-IDF fallback implement, and the fallback
  is what actually runs — in the app, the tests, and CI. `PgVectorIndex` is a
  documented stub showing the exact swap-in shape; it is not implemented or
  tested.
- **Claim verification is lexical, not a language-model entailment check.**
  There's no NLI model available here, so a claim is "supported" when enough
  of its significant words appear in the cited chunk's text — intentionally
  conservative, occasionally too strict (a true paraphrase can fail it).
- **The transparency-record cache is in-memory**, not persistent. Restarting
  the process or exceeding `MAX_CACHE` drops old entries; every `/api/answer`
  response still returns its full transparency record inline regardless.
- **This is modelled on the EU AI Act's transparency obligations, not a
  compliance claim.** It is not EU AI Act compliant software — no such
  claim is made anywhere in this repository.
- **Live generation needs `ANTHROPIC_API_KEY`.** Without one, the API uses a
  documented deterministic extractive fallback (`extractive-lexical-v1`,
  visible in the `model_id` field of every transparency record) — which is
  what the live demo above actually runs on right now.

## What is not built yet

The Next.js frontend (`frontend/`) is real and passes its own build/lint, but
is not yet deployed as a separate Vercel project. No Playwright end-to-end
test exists against the live deployment.

## Running it locally

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt -r requirements-dev.txt
python -m pytest                      # 72 tests
python -m eval.run_eval               # the CI gate, against recorded fixtures
python -m eval.run_eval --record      # refresh fixtures against a live model (needs ANTHROPIC_API_KEY)
uvicorn api.main:app --reload
```

```bash
cd frontend && npm install && npm run dev   # http://localhost:3000
```
