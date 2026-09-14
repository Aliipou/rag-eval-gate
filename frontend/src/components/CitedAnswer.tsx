"use client";

import { useMemo, useState } from "react";
import type { Citation } from "@/lib/types";

interface Props {
  answerText: string;
  citations: Citation[];
}

interface Segment {
  text: string;
  citationIndex: number | null; // index into the de-duplicated citation list, or null for plain text
}

// The backend embeds citations inline as "<claim text>. [chunk_id]"
// (see rag/pipeline.py's finalize node). This splits that text into
// plain-text segments and citation markers, and maps each marker to a
// stable, de-duplicated citation number (1, 2, 3, ...) in order of
// first appearance -- the same chunk cited twice gets the same number.
function parseSegments(answerText: string, citations: Citation[]): { segments: Segment[]; orderedCitations: Citation[] } {
  const byId = new Map(citations.map((c) => [c.chunk_id, c]));
  const orderedCitations: Citation[] = [];
  const indexById = new Map<string, number>();

  const markerRe = /\[([a-zA-Z0-9._-]+)\]/g;
  const segments: Segment[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = markerRe.exec(answerText)) !== null) {
    const chunkId = match[1];
    if (!byId.has(chunkId)) continue; // not a real citation marker, leave as plain text

    const plain = answerText.slice(lastIndex, match.index);
    if (plain) segments.push({ text: plain, citationIndex: null });

    if (!indexById.has(chunkId)) {
      indexById.set(chunkId, orderedCitations.length);
      orderedCitations.push(byId.get(chunkId)!);
    }
    segments.push({ text: match[0], citationIndex: indexById.get(chunkId)! });
    lastIndex = markerRe.lastIndex;
  }

  const rest = answerText.slice(lastIndex);
  if (rest) segments.push({ text: rest, citationIndex: null });

  return { segments, orderedCitations };
}

export default function CitedAnswer({ answerText, citations }: Props) {
  const { segments, orderedCitations } = useMemo(
    () => parseSegments(answerText, citations),
    [answerText, citations]
  );
  const [openIndex, setOpenIndex] = useState<number | null>(null);

  return (
    <div className="space-y-3">
      <p className="leading-relaxed text-neutral-900 dark:text-neutral-100">
        {segments.map((seg, i) =>
          seg.citationIndex === null ? (
            <span key={i}>{seg.text}</span>
          ) : (
            <button
              key={i}
              onClick={() =>
                setOpenIndex(openIndex === seg.citationIndex ? null : seg.citationIndex)
              }
              className="mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded-full bg-blue-100 px-1 text-xs font-medium text-blue-800 align-super hover:bg-blue-200 dark:bg-blue-900 dark:text-blue-200 dark:hover:bg-blue-800"
              aria-expanded={openIndex === seg.citationIndex}
            >
              {seg.citationIndex + 1}
            </button>
          )
        )}
      </p>

      {orderedCitations.length > 0 && (
        <div className="space-y-2 border-t border-neutral-200 pt-3 dark:border-neutral-800">
          <p className="text-xs font-medium uppercase tracking-wide text-neutral-500">
            Sources
          </p>
          {orderedCitations.map((c, i) => (
            <div key={c.chunk_id} className="rounded-md border border-neutral-200 dark:border-neutral-800">
              <button
                onClick={() => setOpenIndex(openIndex === i ? null : i)}
                className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm"
              >
                <span className="inline-flex h-5 min-w-5 items-center justify-center rounded-full bg-blue-100 px-1 text-xs font-medium text-blue-800 dark:bg-blue-900 dark:text-blue-200">
                  {i + 1}
                </span>
                <span className="font-mono text-xs text-neutral-500">{c.chunk_id}</span>
                <span className="ml-auto text-xs text-neutral-400">
                  {openIndex === i ? "hide" : "expand"}
                </span>
              </button>
              {openIndex === i && (
                <div className="border-t border-neutral-200 px-3 py-2 text-sm text-neutral-600 dark:border-neutral-800 dark:text-neutral-300">
                  {c.text}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
