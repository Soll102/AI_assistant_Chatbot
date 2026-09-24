"""Diagnostic: when the pipeline refuses, WHO refused?

The end-to-end eval reports that 4 of 26 answerable questions end in a refusal.
That number alone cannot tell us whether the small model could not answer, or
whether a *good draft* was destroyed by the verification pass. Those are very
different problems: the first is a model limit, the second is a pipeline bug we
can fix.

So wrap ``_generate_text`` and log every prompt/response pair. That captures the
draft generation and the verification call in one place, with no changes to app
code.

Run:  ./.venv/Scripts/python.exe -u tests/diag_verifier.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.services.llm_client import LLMClient, looks_like_refusal
from app.services.rag_tools import ToolPlan
from tests.eval_corpus import EVAL_CASES, EVAL_DOCUMENTS
from tests.eval_retrieval import build_store

KINDS = {"paraphrase", "compare", "identifier"}


def main() -> int:
    settings = get_settings()
    if not settings.openrouter_api_key:
        print("need OPENROUTER_API_KEY", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory() as tmp:
        store = build_store(EVAL_DOCUMENTS, settings.min_query_coverage, Path(tmp))

        llm = LLMClient(
            settings.openrouter_api_key,
            settings.openrouter_model,
            settings.openrouter_fallback_model,
        )

        calls: list[tuple[str, str]] = []
        real_generate = llm._generate_text

        def spy(prompt: str, image_bytes: bytes | None = None) -> str:
            text = real_generate(prompt, image_bytes)
            calls.append((prompt, text))
            return text

        llm._generate_text = spy  # type: ignore[method-assign]

        plan = ToolPlan(name="search_pdf", query="", reason="diagnostic")

        for case in EVAL_CASES:
            if case.kind not in KINDS:
                continue
            calls.clear()
            sources = store.search(case.question, top_k=settings.top_k)
            if not sources:
                print(f"\n=== [{case.kind}] {case.question}\n    gate blocked")
                continue

            draft = llm.finalize_with_sources(case.question, sources, plan, None)
            n_after_draft = len(calls)
            final, verification = llm.verify_answer(case.question, draft, sources)

            draft_refused = looks_like_refusal(draft)
            final_refused = looks_like_refusal(final)

            verdict = "OK"
            if draft_refused and final_refused:
                verdict = "MODEL  (draft itself refused)"
            elif not draft_refused and final_refused:
                verdict = "VERIFIER KILLED A GOOD DRAFT  <-- pipeline bug"

            print(f"\n{'='*78}")
            print(f"[{case.kind}] {case.question}")
            print(f"  verdict      : {verdict}")
            print(f"  verification : {verification[:80]}")
            print(f"  draft        : {' '.join(draft.split())[:150]}")
            print(f"  final        : {' '.join(final.split())[:150]}")

            # The raw verifier JSON decides which branch ran.
            if n_after_draft < len(calls):
                raw = calls[-1][1]
                print(f"  verifier raw : {' '.join(raw.split())[:200]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
