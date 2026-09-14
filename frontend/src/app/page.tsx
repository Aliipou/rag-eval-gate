"use client";

import { useState } from "react";
import { ApiRequestError, askQuestion } from "@/lib/api";
import type { AnswerResponse } from "@/lib/types";
import CitedAnswer from "@/components/CitedAnswer";
import TransparencyPanel from "@/components/TransparencyPanel";

const EXAMPLE_QUESTIONS = [
  "Is everyone equal before the law in Finland?",
  "Who exercises legislative power in Finland?",
  "What is the purpose of a trial period in a Finnish employment contract?",
  "What is the statutory minimum wage in Finland?", // deliberately unanswerable by this corpus
];

export default function Home() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnswerResponse | null>(null);

  async function submit(q: string) {
    const trimmed = q.trim();
    if (!trimmed || loading) return;
    setLoading(true);
    setError(null);
    try {
      const res = await askQuestion(trimmed);
      setResult(res);
    } catch (e) {
      if (e instanceof ApiRequestError) {
        setError(
          e.status === 0 || e.message.includes("fetch")
            ? "Could not reach the API. Is `uvicorn api.main:app --reload` running on localhost:8000? See README for setup."
            : e.message
        );
      } else {
        setError("Something went wrong.");
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-6 px-4 py-10">
      <header>
        <h1 className="text-2xl font-semibold">rag-eval-gate</h1>
        <p className="mt-1 text-sm text-neutral-500">
          Retrieval-augmented answering over the Constitution of Finland and a
          Ministry of Economic Affairs and Employment guide (see{" "}
          <code className="font-mono">corpus/SOURCES.md</code>). Every claim is
          cited to a source chunk and independently verified before it&apos;s
          shown -- unsupported claims are dropped, not rewritten.
        </p>
      </header>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          submit(question);
        }}
        className="flex gap-2"
      >
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question about Finnish constitutional or employment law..."
          className="flex-1 rounded-md border border-neutral-300 bg-white px-3 py-2 text-sm outline-none focus:border-blue-500 dark:border-neutral-700 dark:bg-neutral-950"
        />
        <button
          type="submit"
          disabled={loading}
          className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
        >
          {loading ? "Asking..." : "Ask"}
        </button>
      </form>

      <div className="flex flex-wrap gap-2">
        {EXAMPLE_QUESTIONS.map((q) => (
          <button
            key={q}
            onClick={() => {
              setQuestion(q);
              submit(q);
            }}
            className="rounded-full border border-neutral-300 px-3 py-1 text-xs text-neutral-600 hover:bg-neutral-100 dark:border-neutral-700 dark:text-neutral-300 dark:hover:bg-neutral-900"
          >
            {q}
          </button>
        ))}
      </div>

      {error && (
        <div className="rounded-md border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-800 dark:border-red-900 dark:bg-red-950 dark:text-red-200">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-4">
          <div className="rounded-lg border border-neutral-200 p-4 dark:border-neutral-800">
            {result.answered ? (
              <CitedAnswer answerText={result.answer_text} citations={result.citations} />
            ) : (
              <div>
                <p className="text-neutral-700 dark:text-neutral-300">{result.answer_text}</p>
                {result.refusal_reason && (
                  <p className="mt-1 text-xs text-neutral-400">
                    reason: {result.refusal_reason}
                  </p>
                )}
              </div>
            )}
          </div>

          <TransparencyPanel record={result.transparency} />
        </div>
      )}
    </main>
  );
}
