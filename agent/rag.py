# agent/rag.py
"""
RAG-enhanced dataset understanding.

Builds a per-column vector index from dataset metadata so the agent can
retrieve the most relevant schema context for any user query before calling
the LLM.  Retrieved chunks are injected into the system prompt, giving the
model precise column statistics and type information without wasting tool
calls to re-inspect the data.

Embedding backend selection (automatic, in priority order)
-----------------------------------------------------------
1. Ollama — uses the `nomic-embed-text` model (or whatever cfg.rag_embed_model
   specifies).  Requires `ollama serve` to be running with the model pulled.
2. TF-IDF (sklearn) — zero-latency, no external service.  Accuracy is lower
   but sufficient for column-name / dtype / stats retrieval tasks.
   Also used when cfg.rag_force_tfidf = True (e.g. in CI).

Chunks indexed per dataset
--------------------------
    overview        — shape, columns, dtypes, memory, duplicates
    col_<name>      — one chunk per column (dtype, stats, top values, outliers)
    correlations    — top-10 correlated pairs for numeric columns

Usage
-----
    from agent.rag import dataset_rag
    dataset_rag.index_dataframe(df, source_path="titanic.csv")
    context = dataset_rag.retrieve_as_context("which columns predict survival?")
    # → inject context into system prompt
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from agent.config import cfg


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class RAGChunk:
    chunk_id: str
    source: str          # "overview" | column name | "correlations"
    text: str
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


class DatasetRAG:
    """
    In-memory RAG index for dataset metadata.

    Call index_dataframe() once after loading a new dataset.
    Call retrieve_as_context() on every user query to get an injected prompt.
    """

    def __init__(
        self,
        embed_model: str = cfg.rag_embed_model,
        force_tfidf: bool = cfg.rag_force_tfidf,
    ):
        self.embed_model = embed_model
        self.force_tfidf = force_tfidf
        self.chunks: list[RAGChunk] = []
        self._backend: Optional[str] = None      # "ollama" | "tfidf"
        self._ollama_embedder = None
        self._tfidf_vectorizer = None
        self._indexed_path: str = ""
        self._index_ts: float = 0.0

    # ── Backend detection ─────────────────────────────────────────────────────

    def _init_backend(self) -> str:
        if self._backend is not None:
            return self._backend

        if self.force_tfidf:
            self._backend = "tfidf"
            return self._backend

        # Try Ollama first
        try:
            from langchain_ollama import OllamaEmbeddings
            emb = OllamaEmbeddings(model=self.embed_model)
            emb.embed_query("ping")           # connectivity test
            self._ollama_embedder = emb
            self._backend = "ollama"
            return self._backend
        except Exception:
            pass

        # Fall back to TF-IDF
        try:
            import sklearn  # noqa: F401
            self._backend = "tfidf"
            return self._backend
        except ImportError:
            raise RuntimeError(
                "No RAG backend available. "
                "Either start Ollama (ollama serve) or install scikit-learn."
            )

    def _embed_all(self, texts: list[str]) -> np.ndarray:
        backend = self._init_backend()
        if backend == "ollama":
            return np.array(self._ollama_embedder.embed_documents(texts))
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf_vectorizer = TfidfVectorizer(
                max_features=512, ngram_range=(1, 2), sublinear_tf=True
            )
            return self._tfidf_vectorizer.fit_transform(texts).toarray()

    def _embed_query(self, query: str) -> np.ndarray:
        backend = self._init_backend()
        if backend == "ollama":
            return np.array(self._ollama_embedder.embed_query(query))
        else:
            if self._tfidf_vectorizer is None:
                raise RuntimeError("Index is empty — call index_dataframe() first.")
            return self._tfidf_vectorizer.transform([query]).toarray()[0]

    # ── Chunk builders ────────────────────────────────────────────────────────

    @staticmethod
    def _overview_text(df: pd.DataFrame, source: str) -> str:
        numeric = df.select_dtypes(include="number").columns.tolist()
        categorical = df.select_dtypes(exclude="number").columns.tolist()
        return "\n".join([
            f"Dataset: {source}",
            f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns",
            f"All columns: {list(df.columns)}",
            f"Numeric columns ({len(numeric)}): {numeric}",
            f"Categorical columns ({len(categorical)}): {categorical}",
            f"Total missing values: {df.isnull().sum().sum():,}",
            f"Duplicate rows: {df.duplicated().sum():,}",
            f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB",
        ])

    @staticmethod
    def _column_text(df: pd.DataFrame, col: str) -> str:
        s = df[col]
        n_miss = int(s.isnull().sum())
        lines = [
            f"Column: {col}",
            f"Dtype: {s.dtype}",
            f"Non-null: {s.notna().sum():,} / {len(s):,}",
            f"Missing: {n_miss:,} ({n_miss / len(s) * 100:.1f}%)",
            f"Unique values: {s.nunique():,}",
        ]
        if pd.api.types.is_numeric_dtype(s):
            desc = s.describe()
            lines += [
                f"Mean: {desc['mean']:.4f}",
                f"Std: {desc['std']:.4f}",
                f"Min: {desc['min']:.4f}",
                f"Median: {desc['50%']:.4f}",
                f"Max: {desc['max']:.4f}",
            ]
            q1, q3 = desc["25%"], desc["75%"]
            iqr = q3 - q1
            n_out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
            lines.append(f"Outliers (IQR): {n_out}")
        else:
            top = s.value_counts().head(5)
            lines.append(f"Top values: {top.index.tolist()}")
            lines.append(f"Top counts: {top.values.tolist()}")
        return "\n".join(lines)

    @staticmethod
    def _correlation_text(df: pd.DataFrame) -> str:
        num = df.select_dtypes(include="number")
        if num.shape[1] < 2:
            return "Not enough numeric columns to compute correlations."
        corr = num.corr()
        cols = corr.columns.tolist()
        pairs = [
            (cols[i], cols[j], corr.iloc[i, j])
            for i in range(len(cols))
            for j in range(i + 1, len(cols))
        ]
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        lines = ["Top pairwise correlations:"]
        for c1, c2, v in pairs[:10]:
            lines.append(f"  {c1} ↔ {c2}: {v:.4f}")
        return "\n".join(lines)

    # ── Public API ────────────────────────────────────────────────────────────

    def index_dataframe(self, df: pd.DataFrame, source_path: str = "dataset") -> None:
        """
        Build or rebuild the chunk index from *df*.
        Idempotent — safe to call on every load.
        """
        raw: list[tuple[str, str, str]] = []   # (chunk_id, source, text)

        raw.append(("overview", "overview", self._overview_text(df, source_path)))
        for col in df.columns:
            raw.append((f"col_{col}", col, self._column_text(df, col)))
        raw.append(("correlations", "correlations", self._correlation_text(df)))

        texts = [t for _, _, t in raw]
        embeddings = self._embed_all(texts)

        self.chunks = [
            RAGChunk(chunk_id=cid, source=src, text=txt,
                     embedding=embeddings[i])
            for i, (cid, src, txt) in enumerate(raw)
        ]
        self._indexed_path = source_path
        self._index_ts = time.time()

    def retrieve(self, query: str, top_k: int = cfg.rag_top_k) -> list[RAGChunk]:
        """Return the top_k most relevant chunks for *query*."""
        if not self.chunks:
            return []

        q_emb = self._embed_query(query)
        stack = np.stack([c.embedding for c in self.chunks])

        # Cosine similarity (normalised dot product)
        norm_stack = np.linalg.norm(stack, axis=1, keepdims=True) + 1e-10
        norm_q = np.linalg.norm(q_emb) + 1e-10
        sims = (stack @ q_emb) / (norm_stack.squeeze() * norm_q)
        sims = np.nan_to_num(sims, nan=-1.0)

        top_idx = np.argsort(sims)[::-1][:top_k]
        return [self.chunks[i] for i in top_idx]

    def retrieve_as_context(
        self,
        query: str,
        top_k: int = cfg.rag_top_k,
    ) -> str:
        """
        Retrieve relevant chunks and format them as a context block for
        injection into the system prompt.  Returns "" when not indexed.
        """
        if not self.is_indexed():
            return ""
        chunks = self.retrieve(query, top_k=top_k)
        if not chunks:
            return ""

        lines = ["## RAG Context — Retrieved dataset knowledge\n"]
        for chunk in chunks:
            lines.append(f"### {chunk.source}")
            lines.append(chunk.text)
            lines.append("")
        return "\n".join(lines)

    def is_indexed(self) -> bool:
        return len(self.chunks) > 0

    @property
    def backend(self) -> str:
        """Which embedding backend is active ('ollama' | 'tfidf' | 'unknown')."""
        return self._backend or "unknown"

    @property
    def indexed_path(self) -> str:
        return self._indexed_path


# ── Global singleton ──────────────────────────────────────────────────────────

dataset_rag = DatasetRAG()
