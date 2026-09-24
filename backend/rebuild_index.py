"""Index every stored PDF that the BM25 index does not know about.

Use this after upgrading from the old ChromaDB build, or any time documents
appear in the UI as missing while their PDFs are still on disk.

    cd backend
    .\\.venv\\Scripts\\python.exe rebuild_index.py              # add missing docs
    .\\.venv\\Scripts\\python.exe rebuild_index.py --check      # report only
    .\\.venv\\Scripts\\python.exe rebuild_index.py --force      # re-index everything
    .\\.venv\\Scripts\\python.exe rebuild_index.py --clean-text # strip safety lines

``--force`` deletes the current index first, so it also rebuilds documents whose
chunking changed (for example after changing CHUNK_SIZE).

``--clean-text`` repairs text that is already stored, without re-reading any
PDF. Use it once after upgrading past the release that started stripping safety
lines on the vision path: older runs could persist a model's
"User Safety: safe" verdict as page content, which then became a retrievable
chunk and got quoted back to users as if it came from their document.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import get_settings  # noqa: E402
from app.services.llm_client import strip_safety_lines  # noqa: E402
from app.services.pdf_processor import display_name_from_pdf  # noqa: E402
from app.services.vector_store import VectorStore  # noqa: E402


def clean_stored_text(index_dir: Path, apply: bool) -> int:
    """Strip safety-classification lines from stored chunk text.

    Works on the persisted JSON rather than through ``VectorStore`` so the BM25
    caches (term frequencies, document frequencies, average length) are rebuilt
    from the cleaned text on the next load instead of going stale.
    """
    import os
    import tempfile

    path = index_dir / "lexical_store.json"
    if not path.exists():
        print(f"No index at {path}")
        return 0

    data = json.loads(path.read_text(encoding="utf-8"))
    chunks = data.get("chunks") or []
    # Chỉ giữ tail 70 ký tự cho report thay vì (before, after) full-text:
    # trước đây giữ 2x RAM trên thư viện lớn.
    changed: list[tuple[str, str, str]] = []
    for chunk in chunks:
        original = chunk.get("text") or ""
        cleaned = strip_safety_lines(original)
        if cleaned != original:
            changed.append((chunk.get("filename", "?"), original[-70:], cleaned[-70:]))
            chunk["text"] = cleaned

    print(f"chunks scanned: {len(chunks)}")
    print(f"chunks with a safety line: {len(changed)}")
    for filename, before, after in changed[:20]:
        print(f"  - {filename}")
        print(f"      before: {before!r}")
        print(f"      after : {after!r}")
    if len(changed) > 20:
        print(f"  ... and {len(changed) - 20} more")

    if not changed:
        return 0
    if not apply:
        print("\nDry run. Re-run with --clean-text --apply to write the changes.")
        return 0

    backup = path.with_suffix(".json.bak")
    shutil.copy2(path, backup)
    # Atomic write như VectorStore._save để crash không để lại JSON cụt.
    tmp_fd, tmp_name = tempfile.mkstemp(dir=str(index_dir), prefix="lexical_store.", suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(data, ensure_ascii=False))
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    print(f"\nWrote {path}")
    print(f"Backup kept at {backup}")
    return len(changed)


def build_store(settings) -> VectorStore:
    return VectorStore(
        settings.index_dir,
        max_context_chunks=settings.max_context_chunks,
        min_query_coverage=settings.min_query_coverage,
        evidence_metric=settings.evidence_metric,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="report what is missing, index nothing")
    parser.add_argument("--force", action="store_true", help="wipe the index and re-index every PDF")
    parser.add_argument("--workers", type=int, default=0, help="override INGEST_WORKERS")
    parser.add_argument(
        "--vision",
        action="store_true",
        help="also run the vision fallback on low-text pages (slow: one API call per page)",
    )
    parser.add_argument(
        "--clean-text",
        action="store_true",
        help="strip stored safety-preamble lines; add --apply to write",
    )
    parser.add_argument("--apply", action="store_true", help="with --clean-text: actually write the changes")
    args = parser.parse_args()

    settings = get_settings()
    if args.workers:
        # Clamp 1..16 như Settings: trước đây --workers 100 bypass Pydantic
        # (gán trực tiếp) và spawn hàng trăm thread.
        settings.ingest_workers = max(1, min(int(args.workers), 16))

    if args.clean_text:
        return 0 if clean_stored_text(settings.index_dir, args.apply) >= 0 else 1

    uploads: Path = settings.uploads_dir
    pdfs = sorted(uploads.glob("*.pdf"))
    print(f"uploads: {uploads}")
    print(f"index:   {settings.index_dir}")
    print(f"PDFs on disk: {len(pdfs)}")

    if args.force:
        for stale in settings.index_dir.glob("*.json"):
            stale.unlink()
        print("--force: cleared the existing index")

    store = build_store(settings)
    indexed = {document.id for document in store.list_documents()}
    missing = [path for path in pdfs if path.stem not in indexed]

    print(f"already indexed: {len(indexed & {path.stem for path in pdfs})}")
    print(f"missing:         {len(missing)}")
    if not missing:
        print("Nothing to do.")
        return 0
    for path in missing:
        print(f"  - {path.stem}  ({path.stat().st_size / 1e6:.1f} MB)")

    if args.check:
        return 1

    if args.vision:
        print("vision fallback: ON (one API call per low-text page — this is slow)")
    else:
        print("vision fallback: off (pass --vision to read low-text pages as images)")

    # Import here so --check stays dependency-light.
    from app.main import _run_ingest_jobs

    jobs = [(path.stem, display_name_from_pdf(path), path) for path in missing]
    total = len(jobs)
    started = time.perf_counter()

    def report(index: int, summary, error) -> None:
        # Persist after every document: a full re-index of a large library runs
        # for minutes, and losing all of it to one crash at the end is not
        # acceptable when saving costs a fraction of a second.
        store.flush()
        done = time.perf_counter() - started
        if summary is not None:
            print(
                f"[{done:6.1f}s] ({index + 1}/{total}) OK   {summary.filename} "
                f"-> {summary.chunks} chunks / {summary.pages} pages",
                flush=True,
            )
        else:
            print(f"[{done:6.1f}s] ({index + 1}/{total}) FAIL {error}", flush=True)

    # delete_on_failure=False: never destroy a stored PDF because indexing failed.
    # use_vision=False unless asked: one network round trip per low-text page is
    # the slowest stage of ingest, and a repair run should be fast and offline.
    results = _run_ingest_jobs(
        jobs,
        settings,
        store,
        delete_on_failure=False,
        progress=report,
        use_vision=args.vision,
    )
    store.flush()
    elapsed = time.perf_counter() - started

    ok = sum(1 for summary, _ in results if summary is not None)
    failed = len(results) - ok

    print(f"\nDone in {elapsed:.1f}s: {ok} indexed, {failed} failed.")
    print(f"Index now holds {len(store.list_documents())} documents.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
