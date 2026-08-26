from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


from .models import ScoredChunk, EvidencePack, Citation


TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokens(value: str) -> list[str]:
    return TOKEN_RE.findall(value.lower())


@dataclass(frozen=True)
class Chunk:
    filename: str
    heading: str
    text: str
    metadata: dict[str, str]
    ordinal: int


def parse_markdown(path: Path) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    metadata: dict[str, str] = {}
    if raw.startswith("---\n"):
        _, front, raw = raw.split("---\n", 2)
        for line in front.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip()
    current = "Document overview"
    blocks: list[tuple[str, list[str]]] = []
    buf: list[str] = []
    for line in raw.splitlines():
        heading = re.match(r"^#{1,6}\s+(.+)$", line)
        if heading:
            if buf:
                blocks.append((current, buf))
            current, buf = heading.group(1).strip(), []
        else:
            buf.append(line)
    if buf:
        blocks.append((current, buf))
    return [Chunk(path.name, heading, "\n".join(lines).strip(), metadata, i)
            for i, (heading, lines) in enumerate(blocks) if "\n".join(lines).strip()]


class HybridIndex:
    """Small deterministic BM25 + hashed-vector index; no external model/runtime."""
    def __init__(self, kb_path: Path) -> None:
        self.chunks = [c for p in sorted(kb_path.glob("*.md")) for c in parse_markdown(p)]
        self.docs = [tokens(f"{c.heading} {c.text}") for c in self.chunks]
        self.df = Counter(t for doc in self.docs for t in set(doc))
        self.avg_len = sum(map(len, self.docs)) / len(self.docs) if self.docs else 1.0

    @staticmethod
    def _authority(chunk: Chunk) -> float:
        m = chunk.metadata
        if m.get("audience") != "customer": return -100.0  # Indexed, never eligible for answer evidence.
        if m.get("status") == "active" and m.get("policy_authority") == "official": return 2.0
        if m.get("status") == "superseded": return -4.0
        return -2.0

    def search(self, query: str, limit: int = 8) -> list[tuple[Chunk, float]]:
        detailed = self.search_detailed(query, limit)
        return [(c, sc.total_score) for c, sc in detailed]

    def search_detailed(self, query: str, limit: int = 8) -> list[tuple[Chunk, ScoredChunk]]:
        q = tokens(query)
        if not q: return []
        n, k1, b = len(self.docs), 1.4, .75
        q_counts = Counter(q)
        rows: list[tuple[Chunk, ScoredChunk]] = []
        for chunk, doc in zip(self.chunks, self.docs):
            tf = Counter(doc)
            bm25 = sum((math.log(1 + (n - self.df[t] + .5) / (self.df[t] + .5)) * tf[t] * (k1 + 1) /
                        (tf[t] + k1 * (1 - b + b * len(doc) / self.avg_len))) * count
                       for t, count in q_counts.items() if t in tf)
            # Sparse hashed embedding proxy: cosine over normalized term-frequency vectors.
            dot = sum(tf[t] * count for t, count in q_counts.items())
            denom = ((sum(v*v for v in tf.values()) ** .5) * (sum(v*v for v in q_counts.values()) ** .5))
            cosine = (dot / denom) if denom > 0 else 0.0
            auth = self._authority(chunk)
            total = bm25 + .8 * cosine + auth
            
            m = chunk.metadata
            eligible = (m.get("audience") == "customer" and m.get("status") == "active" and auth > 0)
            reason = None
            if m.get("audience") != "customer":
                reason = "Excluded: internal documentation (audience != customer)"
            elif m.get("status") == "superseded":
                reason = "Excluded: superseded document (status == superseded)"
            elif auth <= 0:
                reason = "Excluded: non-official policy authority"

            sc = ScoredChunk(
                filename=chunk.filename,
                heading=chunk.heading,
                bm25=round(bm25, 4),
                cosine=round(cosine, 4),
                authority_score=round(auth, 4),
                total_score=round(total, 4),
                eligible=eligible,
                exclusion_reason=reason
            )
            rows.append((chunk, sc))
        return sorted(rows, key=lambda x: x[1].total_score, reverse=True)[:limit]

    def build_evidence_pack(self, query: str, rewritten_query: str | None = None, limit: int = 8) -> EvidencePack:
        search_target = rewritten_query or query
        detailed = self.search_detailed(search_target, limit)
        
        candidates: list[ScoredChunk] = [sc for _, sc in detailed]
        selected_citations: list[Citation] = []
        excluded_sources: list[dict[str, str]] = []
        seen_citations = set()

        for chunk, sc in detailed:
            if sc.eligible:
                key = (chunk.filename, chunk.heading)
                if key not in seen_citations:
                    seen_citations.add(key)
                    selected_citations.append(Citation(filename=chunk.filename, heading=chunk.heading))
            else:
                excluded_sources.append({
                    "filename": chunk.filename,
                    "heading": chunk.heading,
                    "reason": sc.exclusion_reason or "Low relevance score / ineligible metadata"
                })

        pack = EvidencePack(
            query=query,
            rewritten_query=rewritten_query,
            candidates=candidates,
            selected_citations=selected_citations,
            excluded_sources=excluded_sources,
            answerable=len(selected_citations) > 0,
        )
        return pack

    def active_chunks(self, filename: str) -> list[Chunk]:
        return [c for c in self.chunks if c.filename == filename and self._authority(c) > 0]

