"""Build corpus/data/chunks.jsonl from the registered sources.

Usage:
    python -m corpus.ingest.build_corpus            # use local cached PDFs
    python -m corpus.ingest.build_corpus --live      # re-fetch from Finlex/gov sites

`--live` re-downloads the source PDFs into corpus/raw/ before parsing, so
the corpus can be refreshed against the live sources without touching any
other part of the pipeline. Everything downstream (retrieval, eval,
golden set) only ever reads corpus/data/chunks.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from corpus.ingest.finlex_statute import FinlexStatuteSource, ingest as ingest_finlex_statute
from corpus.ingest.gov_guide_pdf import GovGuideSource, ingest as ingest_gov_guide

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "corpus" / "raw"
DATA_DIR = ROOT / "corpus" / "data"

CONSTITUTION = FinlexStatuteSource(
    document_id="const-731-1999",
    document_title="Constitution of Finland (731/1999, unofficial translation)",
    finlex_page_url="https://www.finlex.fi/en/legislation/translations/1999/eng/731",
    licence=(
        "Unofficial translation published by the Ministry of Justice, Finland via "
        "Finlex (finlex.fi). Legally binding only in Finnish and Swedish. No "
        "separate open-data licence is stated on the source page; reproduced here "
        "for a non-commercial portfolio demo with full source attribution."
    ),
    media_id="1070874",
    local_pdf_path=str(RAW_DIR / "constitution_731_1999.pdf"),
)

EMPLOYMENT_GUIDE = GovGuideSource(
    document_id="emp-guide-tem",
    document_title=(
        "The position of employers and employees under the Employment Contracts Act "
        "(Ministry of Economic Affairs and Employment of Finland, guide)"
    ),
    source_url=(
        "https://tem.fi/documents/1410877/2918935/Employment+Contracts+Act/"
        "b0fca473-f224-46b9-974c-acd153587680"
    ),
    licence=(
        "Published by the Ministry of Economic Affairs and Employment of Finland "
        "(tem.fi) as a public guidance document. No separate licence stated; "
        "reproduced here for a non-commercial portfolio demo with full source "
        "attribution. Note: this is an explanatory guide, not the statute text "
        "itself -- treat citations to it as 'ministry guidance says', not 'the "
        "law says'."
    ),
    local_pdf_path=str(RAW_DIR / "employment_contracts_act.pdf"),
    min_chars=150,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live",
        action="store_true",
        help="Re-fetch PDFs from Finlex/tem.fi instead of using corpus/raw/ cache",
    )
    args = parser.parse_args()

    const_source = CONSTITUTION
    guide_source = EMPLOYMENT_GUIDE
    if args.live:
        const_source = FinlexStatuteSource(
            **{**CONSTITUTION.__dict__, "local_pdf_path": None}
        )
        guide_source = GovGuideSource(
            **{**EMPLOYMENT_GUIDE.__dict__, "local_pdf_path": None}
        )

    chunks = []
    chunks += ingest_finlex_statute(const_source)
    chunks += ingest_gov_guide(guide_source)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "chunks.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_json(), ensure_ascii=False) + "\n")

    print(f"Wrote {len(chunks)} chunks to {out_path}")
    by_doc: dict[str, int] = {}
    for c in chunks:
        by_doc[c.document_id] = by_doc.get(c.document_id, 0) + 1
    for doc, n in by_doc.items():
        print(f"  {doc}: {n} chunks")


if __name__ == "__main__":
    main()
