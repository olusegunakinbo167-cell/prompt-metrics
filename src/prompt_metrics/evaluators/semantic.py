# src/prompt_metrics/evaluators/semantic.py
"""
Semantic similarity evaluator.

Computes cosine similarity between a generated response and a reference text,
bridging the gap between simple string matching and heavy LLM-as-a-judge
evaluation.

Two backends are supported (in order of preference):

1. Custom embedding client (API / local model)
   Pass `embedding_client=text -> list[float]` to use any embedding provider:
   OpenAI, Cohere, sentence-transformers, local ONNX, etc.

   Example:
       >>> import openai
       >>> def openai_embed(text: str) -> list[float]:
       ...     resp = openai.embeddings.create(
       ...         model="text-embedding-3-small", input=text
       ...     )
       ...     return resp.data[0].embedding
       >>> ev = SemanticSimilarityEvaluator(embedding_client=openai_embed)

2. Local TF-IDF fallback (pure Python, zero dependencies)
   If no embedding_client is provided, falls back to a lightweight
   TF-IDF vectorizer implemented in pure Python stdlib. No downloads,
   no heavy dependencies, works everywhere. Good enough for catching
   paraphrases and semantic overlap in most evaluation scenarios.

   For better quality, install sentence-transformers:
       pip install sentence-transformers
   The evaluator will auto-detect it and use all-MiniLM-L6-v2 by default.

The cosine similarity score is clamped to [0.0, 1.0].
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Callable

# Type alias: text -> embedding vector
EmbeddingClient = Callable[[str], list[float]]


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors. Result clamped to [0, 1]."""
    if len(a) != len(b):
        raise ValueError(f"Vector dimension mismatch: {len(a)} vs {len(b)}")
    if not a:
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    cos_sim = dot / (norm_a * norm_b)
    # Clamp to [0, 1] — cosine can be negative, but for text similarity
    # we treat negative as 0 (no similarity)
    return max(0.0, min(1.0, cos_sim))


# ---------------------------------------------------------------------------
# Pure-Python TF-IDF vectorizer
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\b\w+\b")


def _tokenize(text: str) -> list[str]:
    """Simple word tokenizer: lowercase, alphanumeric words only."""
    return _TOKEN_RE.findall(text.lower())


class _TfIdfVectorizer:
    """
    Minimal TF-IDF vectorizer — pure Python, zero dependencies.

    Fits on 2 documents at a time (response + reference), computes
    TF-IDF weights, returns L2-normalized vectors ready for cosine similarity.

    This is NOT meant to replace scikit-learn for large-scale work,
    but it's fast, dependency-free, and surprisingly effective for
    pairwise semantic similarity in evaluation pipelines.
    """

    @staticmethod
    def vectorize_pair(text_a: str, text_b: str) -> tuple[list[float], list[float]]:
        """
        Vectorize two texts into TF-IDF vectors in a shared vocabulary space.

        Returns:
            (vec_a, vec_b) — L2-normalized TF-IDF vectors
        """
        tokens_a = _tokenize(text_a)
        tokens_b = _tokenize(text_b)

        # Build vocabulary
        vocab = sorted(set(tokens_a) | set(tokens_b))
        if not vocab:
            return [], []

        vocab_index = {term: i for i, term in enumerate(vocab)}

        # Term frequencies
        tf_a = Counter(tokens_a)
        tf_b = Counter(tokens_b)

        # Document frequencies (how many of the 2 docs contain each term)
        df: dict[str, int] = {}
        for term in vocab:
            df[term] = (1 if term in tf_a else 0) + (1 if term in tf_b else 0)

        # IDF with smoothing: log((N + 1) / (df + 1)) + 1
        # N = 2 documents
        n_docs = 2
        idf: dict[str, float] = {
            term: math.log((n_docs + 1) / (df[term] + 1)) + 1.0 for term in vocab
        }

        # Build TF-IDF vectors
        def build_vec(tf: Counter[str]) -> list[float]:
            vec = [0.0] * len(vocab)
            total_terms = sum(tf.values())
            if total_terms == 0:
                return vec
            for term, count in tf.items():
                if term in vocab_index:
                    # TF = count / total_terms (normalized term frequency)
                    # TF-IDF = TF * IDF
                    idx = vocab_index[term]
                    tf_norm = count / total_terms
                    vec[idx] = tf_norm * idf[term]
            return vec

        vec_a = build_vec(tf_a)
        vec_b = build_vec(tf_b)

        return vec_a, vec_b


# ---------------------------------------------------------------------------
# Optional: sentence-transformers backend
# ---------------------------------------------------------------------------

class _SentenceTransformersBackend:
    """Lazy wrapper around sentence-transformers, if installed."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install it with: pip install sentence-transformers "
                "or pass an explicit embedding_client."
            ) from e

        self._model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed(self, text: str) -> list[float]:
        vec = self._model.encode(text, convert_to_numpy=True)
        # Normalize to list[float]
        return vec.tolist() if hasattr(vec, "tolist") else list(vec)


# ---------------------------------------------------------------------------
# SemanticSimilarityEvaluator
# ---------------------------------------------------------------------------

class SemanticSimilarityEvaluator:
    """
    Cosine similarity between a response and a reference text.

    Embedding backends (in order of preference):
      1. embedding_client callback — pass any function text -> list[float]
         Works with OpenAI, Cohere, HuggingFace, local models, etc.
      2. sentence_transformers — auto-detected if installed.
         Uses all-MiniLM-L6-v2 by default (fast, ~80MB, good quality).
         Override with model_name=...
      3. Pure-Python TF-IDF — zero-dependency fallback.
         Always available, no downloads. Surprisingly effective for
         evaluation use cases.

    The similarity score is cosine similarity clamped to [0.0, 1.0].

    Args:
        embedding_client: Optional callable text -> list[float].
            If provided, this is used for all embeddings.
        model_name: sentence-transformers model to use if no embedding_client
            is provided and sentence-transformers is installed.
            Default: "all-MiniLM-L6-v2"
        use_tfidf_fallback: If True (default), fall back to pure-Python
            TF-IDF when no embedding_client is provided and
            sentence-transformers is not installed. If False, raises
            an error instead.
        normalize: If True (default), L2-normalize embedding vectors
            before computing cosine similarity. Has no effect on the
            TF-IDF backend (vectors are already normalized).

    Example (API embedding):
        >>> import openai
        >>> def embed(text: str) -> list[float]:
        ...     r = openai.embeddings.create(
        ...         model="text-embedding-3-small", input=text
        ...     )
        ...     return r.data[0].embedding
        >>> ev = SemanticSimilarityEvaluator(embedding_client=embed)

    Example (local TF-IDF fallback, zero dependencies):
        >>> ev = SemanticSimilarityEvaluator()  # uses TF-IDF automatically
        >>> result = ev.evaluate(
        ...     prompt="What is ML?",
        ...     response="Machine learning is a subset of AI.",
        ...     expected_text="ML is a branch of artificial intelligence."
        ... )
        >>> result["score"]  # cosine similarity in [0, 1]
        0.42

    Example (sentence-transformers, if installed):
        >>> ev = SemanticSimilarityEvaluator()  # auto-detects s-t
        >>> # or explicitly:
        >>> ev = SemanticSimilarityEvaluator(model_name="all-mpnet-base-v2")
    """

    name = "semantic_similarity"

    def __init__(
        self,
        embedding_client: EmbeddingClient | None = None,
        *,
        model_name: str = "all-MiniLM-L6-v2",
        use_tfidf_fallback: bool = True,
        normalize: bool = True,
        name: str | None = None,
    ):
        self.embedding_client = embedding_client
        self.normalize = normalize
        self._st_backend: _SentenceTransformersBackend | None = None
        self._backend_name = "custom"

        if name:
            self.name = name

        # Resolve embedding backend
        if embedding_client is not None:
            if not callable(embedding_client):
                raise TypeError("embedding_client must be callable")
            self._backend_name = "custom"
            return

        # Try sentence-transformers
        try:
            self._st_backend = _SentenceTransformersBackend(model_name)
            self._backend_name = f"sentence_transformers:{model_name}"
            return
        except ImportError:
            pass

        # Fall back to pure-Python TF-IDF
        if use_tfidf_fallback:
            self._backend_name = "tfidf"
            return

        # No backend available
        raise RuntimeError(
            "SemanticSimilarityEvaluator requires either:\n"
            "  1. embedding_client=text -> list[float] callback, or\n"
            "  2. sentence-transformers installed "
            "(pip install sentence-transformers), or\n"
            "  3. use_tfidf_fallback=True (default) for pure-Python TF-IDF\n"
            "\nNo embedding backend is available with the current configuration."
        )

    # ---- Embedding ----

    def _embed(self, text: str) -> list[float]:
        """Embed a single text using the configured backend."""
        if self.embedding_client is not None:
            vec = self.embedding_client(text)
            if not isinstance(vec, list):
                # Accept numpy arrays and other sequences too
                try:
                    vec = list(vec)  # type: ignore[arg-type]
                except Exception:
                    raise TypeError(
                        f"embedding_client must return list[float], "
                        f"got {type(vec).__name__}"
                    )
            return [float(x) for x in vec]

        if self._st_backend is not None:
            return self._st_backend.embed(text)

        # TF-IDF backend is handled specially in evaluate() since it
        # needs both texts at once to build a shared vocabulary.
        # This method should not be called in TF-IDF mode.
        raise RuntimeError(
            "_embed() called in TF-IDF mode — "
            "this is a bug, TF-IDF vectorization should happen in evaluate()"
        )

    @staticmethod
    def _l2_normalize(vec: list[float]) -> list[float]:
        """L2-normalize a vector in-place (returns a new list)."""
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0.0:
            return vec[:]
        return [x / norm for x in vec]

    # ---- Evaluation ----

    def evaluate(
        self,
        prompt: str,
        response: str,
        *,
        expected_text: str | None = None,
        keywords: list[str] | None = None,
        case_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Compute cosine similarity between response and expected_text.

        Args:
            prompt: Original input prompt (unused, for Evaluator protocol compliance)
            response: Generated response to evaluate
            expected_text: Reference text to compare against.
                If None, returns score=None.
            keywords: Unused (for Evaluator protocol compliance)
            case_id: Unused (for Evaluator protocol compliance)

        Returns:
            {
              "score": float in [0.0, 1.0] | None,
              "similarity": float,  # alias for score
              "backend": str,  # "custom" | "sentence_transformers:..." | "tfidf"
              "embedding_dim": int | None,  # None for TF-IDF (variable dim)
            }
        """
        _ = prompt, keywords, case_id  # unused, protocol compliance

        if expected_text is None:
            return {
                "score": None,
                "similarity": None,
                "reason": "no expected_text (reference) provided",
                "backend": self._backend_name,
            }

        # TF-IDF backend: vectorize both texts together (shared vocab)
        if self._backend_name == "tfidf":
            try:
                vec_resp, vec_ref = _TfIdfVectorizer.vectorize_pair(
                    response, expected_text
                )
                if not vec_resp or not vec_ref:
                    # Empty vocabulary — no overlapping tokens
                    similarity = 0.0
                else:
                    similarity = _cosine_similarity(vec_resp, vec_ref)
            except Exception as e:
                return {
                    "score": None,
                    "similarity": None,
                    "error": f"TF-IDF vectorization failed: {e}",
                    "backend": self._backend_name,
                }

            return {
                "score": round(similarity, 6),
                "similarity": round(similarity, 6),
                "backend": self._backend_name,
                "embedding_dim": len(vec_resp),
            }

        # Embedding-client / sentence-transformers backend
        try:
            vec_resp = self._embed(response)
            vec_ref = self._embed(expected_text)
        except Exception as e:
            return {
                "score": None,
                "similarity": None,
                "error": f"embedding failed: {e}",
                "backend": self._backend_name,
            }

        if len(vec_resp) != len(vec_ref):
            return {
                "score": None,
                "similarity": None,
                "error": f"embedding dimension mismatch: {len(vec_resp)} vs {len(vec_ref)}",
                "backend": self._backend_name,
            }

        if self.normalize:
            vec_resp = self._l2_normalize(vec_resp)
            vec_ref = self._l2_normalize(vec_ref)

        try:
            similarity = _cosine_similarity(vec_resp, vec_ref)
        except Exception as e:
            return {
                "score": None,
                "similarity": None,
                "error": f"cosine similarity failed: {e}",
                "backend": self._backend_name,
            }

        return {
            "score": round(similarity, 6),
            "similarity": round(similarity, 6),
            "backend": self._backend_name,
            "embedding_dim": len(vec_resp),
        }


__all__ = ["SemanticSimilarityEvaluator", "EmbeddingClient"]
