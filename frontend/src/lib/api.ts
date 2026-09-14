import type { AnswerResponse } from "./types";

// Defaults to the local FastAPI dev server (see api/main.py, run with
// `uvicorn api.main:app --reload`). No live deployment of either side
// exists for this project -- see README "What is implemented / what is
// not" and DECISIONS.md.
const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export class ApiRequestError extends Error {
  constructor(message: string, public status: number) {
    super(message);
    this.name = "ApiRequestError";
  }
}

export async function askQuestion(question: string): Promise<AnswerResponse> {
  const res = await fetch(`${API_BASE_URL}/api/answer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      // response body wasn't JSON; fall back to statusText
    }
    throw new ApiRequestError(detail, res.status);
  }

  return res.json();
}
