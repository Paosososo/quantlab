"""The raw data layer.

Rules
-----
1.  Raw payloads are written **before** any parsing or validation.  If the
    parser crashes, the bytes survive.
2.  Raw files are **immutable and content-addressed**:
    ``data/raw/<provider>/<dataset>/<entity>/<YYYY-MM-DD>/<sha256[:16]>.<ext>``.
    Re-fetching identical bytes rewrites the same path, so the layer is
    idempotent; re-fetching changed bytes creates a new file next to the old
    one, so provider revisions are preserved rather than overwritten.
3.  Every payload gets a sidecar ``.manifest.json`` recording the request, the
    fetch time, the digest and the size.  Provenance for the whole warehouse is
    reconstructable from these files alone.

Why files instead of a ``raw_payloads`` table: blobs in PostgreSQL bloat the
WAL, complicate backups and are awkward to diff.  The database stores the
*path* and the digest (``ingestion_runs.raw_path`` / ``payload_sha256``), which
gives referential integrity without putting megabytes of CSV in a relation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from quantlab.config import get_settings
from quantlab.ingestion.base import RawPayload
from quantlab.logging import get_logger

log = get_logger(__name__)

_EXTENSIONS = {
    "text/csv": "csv",
    "application/json": "json",
    "text/plain": "txt",
    "application/octet-stream": "bin",
}

_SAFE = re.compile(r"[^A-Za-z0-9._=-]+")


def _slug(value: str) -> str:
    """Filesystem-safe token.  Ticker symbols contain '^', '.', '/' and spaces."""
    cleaned = _SAFE.sub("_", value.strip())
    return cleaned.strip("_") or "unknown"


class RawStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else get_settings().raw_dir

    def path_for(self, payload: RawPayload) -> Path:
        ext = _EXTENSIONS.get(payload.content_type, "bin")
        day = payload.fetched_at.date().isoformat()
        return (
            self.root
            / _slug(payload.provider)
            / _slug(payload.dataset)
            / _slug(payload.entity)
            / day
            / f"{payload.sha256[:16]}.{ext}"
        )

    def write(self, payload: RawPayload) -> Path:
        """Persist ``payload`` and its manifest.  Returns the data file path."""
        target = self.path_for(payload)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == payload.size_bytes:
            log.debug("raw.skip_existing", path=str(target))
            return target
        target.write_bytes(payload.content)
        target.with_suffix(target.suffix + ".manifest.json").write_text(
            json.dumps(payload.manifest(), indent=2, sort_keys=True), encoding="utf-8"
        )
        log.info(
            "raw.written",
            provider=payload.provider,
            entity=payload.entity,
            path=str(target),
            bytes=payload.size_bytes,
        )
        return target

    def read(self, path: Path) -> bytes:
        return Path(path).read_bytes()


__all__ = ["RawStore"]
