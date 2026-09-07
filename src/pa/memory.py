"""Semantic memory and code search, on a local Chroma database.

Two collections, deliberately separate because they age differently:

* **notes** - things worth remembering across sessions: decisions, how this
  machine is set up, preferences the user stated. Written on purpose, rarely
  deleted.
* **code** - a re-buildable index of a project's files, chunked by line range.
  Disposable: if it drifts, re-index.

Everything is local. Embeddings run on the CPU through Chroma's bundled
`all-MiniLM-L6-v2` ONNX model (384 dimensions), so no text leaves the machine
and no API is billed for indexing. The model is ~80MB, fetched once to
~/.cache/chroma on first use - `pa memory warm` does it deliberately rather
than surprising someone mid-conversation.

This is what "learns over time" means here, and it is worth being precise: no
model weights change. Recall is retrieval - past context is found and put in
front of the model. It is not fine-tuning.
"""

from __future__ import annotations

import hashlib
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import deps, paths
from .errors import PAError

NOTES = "notes"
CODE = "code"

# Files worth indexing. Everything else is noise or binary.
CODE_SUFFIXES = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".kt", ".go", ".rs", ".rb",
    ".php", ".c", ".h", ".cpp", ".hpp", ".cs", ".swift", ".sh", ".bash", ".zsh",
    ".sql", ".html", ".css", ".scss", ".vue", ".svelte", ".yaml", ".yml",
    ".toml", ".json", ".md", ".rst", ".txt", ".dart", ".lua", ".ex", ".exs",
}
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env", ".mypy_cache",
    ".pytest_cache", "dist", "build", "target", ".next", ".nuxt", "vendor",
    ".gradle", ".idea", ".dart_tool", "Pods", ".terraform", "site-packages",
}
MAX_FILE_BYTES = 400_000
CHUNK_LINES = 45
CHUNK_OVERLAP = 8


class MemoryError_(PAError):
    """Memory backend failed."""


@dataclass
class Hit:
    text: str
    score: float          # 0..1, higher is more similar
    metadata: dict[str, Any]
    id: str = ""

    def cite(self) -> str:
        meta = self.metadata or {}
        if source := meta.get("path"):
            lines = meta.get("lines")
            return f"{source}:{lines}" if lines else str(source)
        if tags := meta.get("tags"):
            return f"note [{tags}]"
        return "note"


class Memory:
    """Local vector store. Constructed lazily - nothing loads until first use."""

    def __init__(self, root: Path | None = None, *, auto_install: bool = True) -> None:
        self.root = root or (paths.data_dir() / "chroma")
        self._auto = auto_install
        self._client: Any = None

    # ---- plumbing -----------------------------------------------------------

    def _connect(self) -> Any:
        if self._client is not None:
            return self._client
        # Chroma phones home by default; this is a local tool, so opt out
        # before the module is imported and reads its settings.
        os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
        chromadb = deps.require(
            "chromadb", auto=self._auto, purpose="semantic memory and code search"
        )
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            self._client = chromadb.PersistentClient(path=str(self.root))
        except Exception as exc:  # noqa: BLE001
            raise MemoryError_(
                f"could not open the memory database at {self.root}: {exc}"
            ) from exc
        return self._client

    def _collection(self, name: str) -> Any:
        return self._connect().get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )

    def available(self) -> bool:
        return deps.available("chromadb")

    def warm(self) -> str:
        """Force the embedding model download now, so first real use is fast."""
        collection = self._collection(NOTES)
        collection.count()
        # A throwaway embed is what actually pulls the ONNX model.
        probe_id = f"__warm__{uuid.uuid4().hex}"
        collection.add(ids=[probe_id], documents=["warmup"], metadatas=[{"kind": "warmup"}])
        collection.delete(ids=[probe_id])
        cache = Path.home() / ".cache" / "chroma"
        size = _dir_size(cache) if cache.exists() else 0
        return f"embedding model ready ({size / 1e6:.0f} MB cached in {cache})"

    # ---- notes --------------------------------------------------------------

    def remember(
        self,
        text: str,
        *,
        tags: Sequence[str] = (),
        kind: str = "note",
        source: str = "",
    ) -> str:
        text = (text or "").strip()
        if not text:
            raise MemoryError_("nothing to remember")
        note_id = uuid.uuid4().hex[:16]
        self._collection(NOTES).add(
            ids=[note_id],
            documents=[text],
            metadatas=[{
                "kind": kind,
                "tags": ",".join(tags),
                "source": source,
                "created": time.time(),
            }],
        )
        return note_id

    def recall(self, query: str, *, k: int = 6, kind: str | None = None) -> list[Hit]:
        collection = self._collection(NOTES)
        if collection.count() == 0:
            return []
        where = {"kind": kind} if kind else None
        return _to_hits(
            collection.query(
                query_texts=[query],
                n_results=min(k, collection.count()),
                where=where,
            )
        )

    def forget(self, note_id: str) -> None:
        self._collection(NOTES).delete(ids=[note_id])

    def notes(self, limit: int = 50) -> list[Hit]:
        collection = self._collection(NOTES)
        if collection.count() == 0:
            return []
        got = collection.get(limit=limit)
        return [
            Hit(text=doc, score=1.0, metadata=meta or {}, id=note_id)
            for note_id, doc, meta in zip(
                got.get("ids", []), got.get("documents", []), got.get("metadatas", [])
            )
        ]

    # ---- code index ---------------------------------------------------------

    def index_files(
        self,
        root: Path,
        *,
        suffixes: Iterable[str] | None = None,
        max_files: int = 4000,
    ) -> dict[str, int]:
        """Chunk and embed a tree. Re-running is cheap: chunk ids are content
        hashes, so unchanged files upsert to themselves."""
        collection = self._collection(CODE)
        wanted = set(suffixes) if suffixes else CODE_SUFFIXES

        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict[str, Any]] = []
        files = skipped = 0

        for path in _walk(root, wanted, max_files):
            try:
                text = path.read_text(errors="replace")
            except OSError:
                skipped += 1
                continue
            if len(text) > MAX_FILE_BYTES:
                skipped += 1
                continue
            files += 1
            for chunk, start, end in _chunk_lines(text):
                digest = hashlib.sha1(
                    f"{path}:{start}:{chunk}".encode(errors="replace")
                ).hexdigest()[:20]
                ids.append(digest)
                docs.append(chunk)
                metas.append({
                    "path": str(path),
                    "lines": f"{start}-{end}",
                    "name": path.name,
                    "indexed": time.time(),
                })

        # Batch the upserts: Chroma has a per-call ceiling well below the size
        # of a real repository.
        written = 0
        for lo in range(0, len(ids), 512):
            hi = lo + 512
            collection.upsert(ids=ids[lo:hi], documents=docs[lo:hi], metadatas=metas[lo:hi])
            written += len(ids[lo:hi])

        return {"files": files, "chunks": written, "skipped": skipped}

    def search_code(self, query: str, *, k: int = 8, path_prefix: str = "") -> list[Hit]:
        collection = self._collection(CODE)
        total = collection.count()
        if total == 0:
            return []
        hits = _to_hits(
            collection.query(query_texts=[query], n_results=min(k * 3 if path_prefix else k, total))
        )
        if path_prefix:
            hits = [h for h in hits if str(h.metadata.get("path", "")).startswith(path_prefix)]
        return hits[:k]

    def drop_code_index(self) -> None:
        try:
            self._connect().delete_collection(CODE)
        except Exception:  # noqa: BLE001 - absent is the desired state anyway
            pass

    # ---- reporting ----------------------------------------------------------

    def stats(self) -> str:
        try:
            notes = self._collection(NOTES).count()
            code = self._collection(CODE).count()
        except PAError as exc:
            return f"memory unavailable: {exc}"
        size = _dir_size(self.root)
        return (
            f"notes {notes}  code chunks {code}  "
            f"on disk {size / 1e6:.1f} MB  at {self.root}"
        )


# ---------------------------------------------------------------------------


def _to_hits(result: dict[str, Any]) -> list[Hit]:
    """Chroma returns parallel lists per query; we only ever send one query.

    Distances are cosine (0 identical, 2 opposite), so score = 1 - d/2 puts
    them on a 0..1 scale where higher means closer.
    """
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    ids = (result.get("ids") or [[]])[0]
    hits = []
    for index, doc in enumerate(documents):
        distance = distances[index] if index < len(distances) else 1.0
        hits.append(
            Hit(
                text=doc,
                score=max(0.0, 1.0 - float(distance) / 2.0),
                metadata=(metadatas[index] if index < len(metadatas) else {}) or {},
                id=ids[index] if index < len(ids) else "",
            )
        )
    return hits


def _walk(root: Path, suffixes: set[str], max_files: int):
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for filename in sorted(filenames):
            if Path(filename).suffix.lower() not in suffixes:
                continue
            count += 1
            if count > max_files:
                return
            yield Path(dirpath) / filename


def _chunk_lines(text: str) -> list[tuple[str, int, int]]:
    """Overlapping windows of lines, so a match near a boundary is still found
    with its context intact."""
    lines = text.splitlines()
    if not lines:
        return []
    chunks: list[tuple[str, int, int]] = []
    step = max(CHUNK_LINES - CHUNK_OVERLAP, 1)
    for start in range(0, len(lines), step):
        window = lines[start : start + CHUNK_LINES]
        body = "\n".join(window).strip()
        if body:
            chunks.append((body, start + 1, start + len(window)))
        if start + CHUNK_LINES >= len(lines):
            break
    return chunks


def _dir_size(path: Path) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            try:
                total += (Path(dirpath) / filename).stat().st_size
            except OSError:
                continue
    return total
