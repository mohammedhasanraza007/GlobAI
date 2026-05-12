"""
rag/rag_system.py
-----------------
RAG owns document indexing, embeddings, vector search, BM25, and the RAG LLM.
No coder or image objects are referenced here.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

from core.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


class RagSystem:
    name = "rag"

    def __init__(
        self,
        config: dict[str, Any],
        cache_dir: str,
        vector_db_path: str,
        llm_device_preference: str,
        embed_device: Any,
        embed_device_kind: str,
    ):
        self.config = config
        self.cache_dir = cache_dir
        self.vector_db_path = vector_db_path
        self.documents_accessed = 0
        self.embeddings_accessed = 0

        from retrieval.embedding_engine import EmbeddingEngine

        self.embedding_engine = EmbeddingEngine(
            model_id=config["embedding_id"],
            device=embed_device,
            device_kind=embed_device_kind,
            cache_dir=cache_dir,
            batch_size=int(config["embedding_batch_size"]),
            dimension=int(config["embedding_dimension"]),
            local_files_only=bool(config["offline_mode"]),
        )

        from retrieval.vector_store import VectorStore

        self.vector_store = VectorStore(
            store_dir=vector_db_path,
            dimension=int(config["embedding_dimension"]),
            max_vectors=int(config["max_vector_budget"]),
            dedupe_similarity=float(config["dedupe_similarity_threshold"]),
            dedupe_window=int(config["vector_dedupe_window"]),
        )

        from retrieval.bm25_retriever import BM25Retriever

        self.bm25_retriever = BM25Retriever(cache_dir=vector_db_path)
        self.bm25_retriever.sync(self.vector_store.chunks)

        from retrieval.hybrid_retriever import HybridRetriever

        self.hybrid_retriever = HybridRetriever(
            vector_store=self.vector_store,
            bm25_retriever=self.bm25_retriever,
            embedding_engine=self.embedding_engine,
            similarity_threshold=float(config["similarity_threshold"]),
            top_k=int(config["top_k"]),
            candidate_k=int(config["candidate_k"]),
            max_context_chars=int(config["max_context_chars"]),
        )

        from retrieval.text_chunker import TextChunker

        self.chunker = TextChunker(
            chunk_size=int(config["chunk_size"]),
            chunk_overlap=int(config["chunk_overlap"]),
        )

        from core.model_loader import ModelLoader

        self.model_loader = ModelLoader(
            model_id=config["model_id"],
            device_preference=llm_device_preference,
            cache_dir=cache_dir,
            keep_loaded=bool(config["keep_model_loaded"]),
            max_input_tokens=int(config["max_input_tokens"]),
            local_files_only=bool(config["offline_mode"]),
            model_role="rag",
        )

        from core.query_engine import QueryEngine

        self.query_engine = QueryEngine(
            model_loader=self.model_loader,
            hybrid_retriever=self.hybrid_retriever,
            max_tokens=int(config["max_tokens"]),
            temperature=float(config["temperature"]),
            unload_on_memory_pressure=bool(config["unload_on_memory_pressure"]),
            ram_pressure_limit_pct=int(config["ram_pressure_limit_pct"]),
        )

    def load(self) -> None:
        if not self.model_loader.is_loaded():
            self.model_loader.load()

    def unload(self) -> None:
        self.model_loader.unload()
        self.embedding_engine.unload()
        MemoryManager.hard_cleanup("rag subsystem unload")
        MemoryManager.stabilize_after_unload("rag subsystem")

    def query(self, prompt: str) -> dict[str, Any]:
        self.documents_accessed += 1
        self.embeddings_accessed += 1
        return self.query_engine.query(prompt)

    def index_texts(self, texts: list[str]) -> int:
        chunks = self.chunker.chunk_documents(texts)
        chunks = [chunk for chunk in chunks if str(chunk).strip()]
        if not chunks:
            return 0
        self.documents_accessed += len(texts)
        self.embeddings_accessed += len(chunks)
        vectors = self.embedding_engine.embed(chunks)
        try:
            if vectors.shape[0] != len(chunks):
                raise RuntimeError(f"Embedding count mismatch: {vectors.shape[0]} vectors for {len(chunks)} chunks.")
            accepted = self.vector_store.add(vectors, chunks)
            self.bm25_retriever.sync(self.vector_store.chunks)
            return accepted
        finally:
            del vectors
            gc.collect()

    def index_paths(self, paths: list[str | Path]) -> int:
        from retrieval.document_loader import DocumentLimits, load_document
        import sys
        import gc

        # PANDAS RULE: Load pandas ONLY during indexing tasks
        try:
            import pandas as pd
        except ImportError:
            pass

        limits = DocumentLimits(
            max_document_chars=int(self.config.get("max_document_chars", 2_000_000)),
            max_file_bytes=int(self.config.get("max_file_bytes", 25_000_000)),
            max_pdf_pages=int(self.config.get("max_pdf_pages", 250)),
            max_pptx_slides=int(self.config.get("max_pptx_slides", 250)),
            max_docx_paragraphs=int(self.config.get("max_docx_paragraphs", 5000)),
        )
        indexed = 0
        try:
            for path in paths:
                indexed += self.index_texts(load_document(path, limits=limits))
        finally:
            # PANDAS RULE: pandas must NOT persist in RAM afterward
            if 'pd' in locals():
                del locals()['pd']
            
            # Remove all pandas-related modules from sys.modules to free RAM
            pandas_modules = [m for m in sys.modules if m == 'pandas' or m.startswith('pandas.')]
            for m in pandas_modules:
                del sys.modules[m]
                
            gc.collect()
            MemoryManager.hard_cleanup("indexing pandas unload")

        return indexed

    def audit_state(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rag_model_loaded": self.model_loader.is_loaded(),
            "embedding_loaded": self.embedding_engine._model is not None,
            "vector_count": self.vector_store.count(),
            "documents_accessed": self.documents_accessed,
            "embeddings_accessed": self.embeddings_accessed,
        }
