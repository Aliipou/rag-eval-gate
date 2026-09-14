# rag-eval-gate frontend

Next.js (App Router) + TypeScript UI for the answering service. Talks to
the FastAPI backend in `../api` over `NEXT_PUBLIC_API_BASE_URL`
(defaults to `http://localhost:8000`, see `.env.example`).

No hosted deployment exists for this project (no Vercel deploy was
attempted -- out of scope for this build; see the top-level README's
"What is implemented / what is not" section and suggested next steps).

## Run locally

From the repository root, in one terminal:

```bash
pip install -r requirements-dev.txt
uvicorn api.main:app --reload
```

In another terminal:

```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

Then open http://localhost:3000.

## What's here

- `src/app/page.tsx` -- the question box, example questions, and answer/refusal display.
- `src/components/CitedAnswer.tsx` -- parses the backend's inline `[chunk_id]`
  citation markers and renders them as numbered, expandable source chips.
- `src/components/TransparencyPanel.tsx` -- renders the full transparency
  record (model id, prompt hash, retrieved chunks + scores, per-claim
  verification outcome, UTC timestamp) and an "Export JSON" button that
  downloads it client-side.
- `src/lib/api.ts`, `src/lib/types.ts` -- typed fetch wrapper for `POST /api/answer`.

`npm run build` and `npm run lint` both pass (verified locally: production
build compiles, TypeScript type-checks clean, zero ESLint warnings).
