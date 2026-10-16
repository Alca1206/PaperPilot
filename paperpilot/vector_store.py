"""Persistent vector storage with ChromaDB: one collection per paper.

Collection names combine the paper's file hash with an "index key" built
from every setting that changes the vectors (embedding model, prefixes,
chunk size, overlap). Change any of them and you get a fresh collection
instead of silently mixing incompatible vectors.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import chromadb

from paperpilot.config import Settings
from paperpilot.schemas import Chunk, RetrievedChunk

COLLECTION_PREFIX = "paper_"


def make_chroma_client(path: Path) -> Any:
    """Open (or create) the on-disk Chroma database. Telemetry off: local-first."""
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(path), settings=chromadb.Settings(anonymized_telemetry=False)
    )


def index_key(settings: Settings) -> str:
    """Short fingerprint of every setting that affects the stored vectors."""
    recipe = "|".join(
        [
            settings.embedding_model,
            f"prefixes={settings.embed_use_prefixes}",
            f"max_chars={settings.chunk_max_chars}",
            f"overlap={settings.chunk_overlap_sentences}",
        ]
    )
    return hashlib.sha256(recipe.encode("utf-8")).hexdigest()[:8]


def collection_name(file_hash: str, settings: Settings) -> str:
    """e.g. 'paper_3f2a9c1b0d4e5f67_a1b2c3d4' (only characters Chroma allows)."""
    return f"{COLLECTION_PREFIX}{file_hash[:16]}_{index_key(settings)}"


class PaperIndex:
    """The stored chunks + vectors of one paper."""

    def __init__(self, client: Any, name: str) -> None:
        self.name = name
        self._client = client
        self._collection = client.get_or_create_collection(
            name=name,
            configuration={"hnsw": {"space": "cosine"}},  # Chroma's default is L2
            embedding_function=None,  # we always supply our own vectors
        )

    @property
    def count(self) -> int:
        return self._collection.count()

    def is_complete(self) -> bool:
        """True only if a previous add_chunks() finished.

        add_chunks() records the expected count in the collection metadata as
        its LAST step, so a crash halfway through leaves this False.
        """
        expected = (self._collection.metadata or {}).get("chunk_count")
        count = self.count
        return expected is not None and count > 0 and expected == count

    def add_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        source_name: str,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(f"Got {len(chunks)} chunks but {len(embeddings)} embeddings")
        if not chunks:
            raise ValueError("Nothing to index: the paper produced no chunks")

        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=[list(e) for e in embeddings],
            documents=[c.text for c in chunks],
            metadatas=[{"page": c.page, "source": source_name} for c in chunks],
        )
        self._collection.modify(
            metadata={"chunk_count": len(chunks), "source_name": source_name}
        )

    def query(
        self,
        query_embedding: Sequence[float],
        top_k: int = 4,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Most similar chunks first. Score = cosine similarity (1 - distance)."""
        if self.count == 0:
            return []
        result = self._collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=min(top_k, self.count),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        # Chroma returns one inner list per query; we sent one query -> [0].
        return [
            RetrievedChunk(chunk_id=cid, page=int(meta["page"]), text=doc, score=1.0 - dist)
            for cid, doc, meta, dist in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
            )
        ]

    def delete(self) -> None:
        self._client.delete_collection(self.name)