"use client";

import type { TransparencyRecord } from "@/lib/types";

interface Props {
  record: TransparencyRecord;
}

function downloadJson(record: TransparencyRecord) {
  const blob = new Blob([JSON.stringify(record, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `transparency-${record.prompt_hash}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export default function TransparencyPanel({ record }: Props) {
  return (
    <div className="space-y-4 rounded-lg border border-neutral-200 bg-neutral-50 p-4 text-sm dark:border-neutral-800 dark:bg-neutral-900">
      <div className="flex items-start justify-between gap-2">
        <div>
          <p className="text-xs font-medium uppercase tracking-wide text-neutral-500">
            Transparency record
          </p>
          <p className="mt-1 text-xs text-neutral-400">
            Modelled loosely on EU AI Act-style transparency expectations
            (traceability, not a compliance claim -- see README).
          </p>
        </div>
        <button
          onClick={() => downloadJson(record)}
          className="shrink-0 rounded border border-neutral-300 px-2 py-1 text-xs font-medium hover:bg-neutral-100 dark:border-neutral-700 dark:hover:bg-neutral-800"
        >
          Export JSON
        </button>
      </div>

      <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
        <dt className="text-neutral-500">Model</dt>
        <dd className="font-mono">{record.model_id}</dd>
        <dt className="text-neutral-500">Prompt hash</dt>
        <dd className="font-mono">{record.prompt_hash}</dd>
        <dt className="text-neutral-500">Timestamp (UTC)</dt>
        <dd className="font-mono">{record.timestamp_utc}</dd>
        <dt className="text-neutral-500">Answered</dt>
        <dd>
          {record.answered ? "yes" : `no${record.refusal_reason ? ` (${record.refusal_reason})` : ""}`}
        </dd>
      </dl>

      <div>
        <p className="mb-1 text-xs font-medium uppercase tracking-wide text-neutral-500">
          Retrieved chunks
        </p>
        <ul className="space-y-1">
          {record.retrieved_chunks.map((c) => (
            <li key={c.chunk_id} className="flex justify-between font-mono text-xs">
              <span>{c.chunk_id}</span>
              <span className="text-neutral-400">{c.similarity_score.toFixed(4)}</span>
            </li>
          ))}
          {record.retrieved_chunks.length === 0 && (
            <li className="text-xs text-neutral-400">none</li>
          )}
        </ul>
      </div>

      <div>
        <p className="mb-1 text-xs font-medium uppercase tracking-wide text-neutral-500">
          Verification per claim
        </p>
        <ul className="space-y-1.5">
          {record.verification.map((v, i) => (
            <li key={i} className="rounded border border-neutral-200 p-2 dark:border-neutral-800">
              <div className="flex items-center gap-2">
                <span
                  className={`inline-flex h-4 min-w-4 items-center justify-center rounded-full px-1 text-[10px] font-bold text-white ${
                    v.supported ? "bg-green-600" : "bg-red-500"
                  }`}
                >
                  {v.supported ? "✓" : "✗"}
                </span>
                <span className="font-mono text-xs text-neutral-500">{v.chunk_id}</span>
                <span className="ml-auto text-[10px] text-neutral-400">{v.reason}</span>
              </div>
              <p className="mt-1 text-xs text-neutral-600 dark:text-neutral-300">{v.claim_text}</p>
            </li>
          ))}
          {record.verification.length === 0 && (
            <li className="text-xs text-neutral-400">no claims were generated</li>
          )}
        </ul>
      </div>
    </div>
  );
}
