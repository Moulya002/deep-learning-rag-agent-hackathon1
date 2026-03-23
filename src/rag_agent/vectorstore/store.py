"""
store.py
========
ChromaDB vector store management.

Handles all interactions with the persistent ChromaDB collection:
initialisation, ingestion, duplicate detection, and retrieval.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from loguru import logger

from rag_agent.agent.state import (
    ChunkMetadata,
    DocumentChunk,
    IngestionResult,
    RetrievedChunk,
)
from rag_agent.config import EmbeddingFactory, Settings, get_settings


import chromadb


class VectorStoreManager:
    """
    Manages the ChromaDB persistent vector store for the corpus.

    All corpus ingestion and retrieval operations pass through this class.
    It is the single point of contact between the application and ChromaDB.

    Parameters
    ----------
    settings : Settings, optional
        Application settings. Uses get_settings() singleton if not provided.

    Example
    -------
    >>> manager = VectorStoreManager()
    >>> result = manager.ingest(chunks)
    >>> print(f"Ingested: {result.ingested}, Skipped: {result.skipped}")
    >>>
    >>> chunks = manager.query("explain the vanishing gradient problem", k=4)
    >>> for chunk in chunks:
    ...     print(chunk.to_citation(), chunk.score)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._embeddings = EmbeddingFactory(self._settings).create()
        self._client = None
        self._collection = None
        self._initialise()

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def _initialise(self) -> None:
        """
        Create or connect to the persistent ChromaDB client and collection.

        Creates the chroma_db_path directory if it does not exist.
        Uses PersistentClient so data survives between application restarts.

        Called automatically during __init__. Should not be called directly.

        Raises
        ------
        RuntimeError
            If ChromaDB cannot be initialised at the configured path.
        """
        # TODO: implement
        # 1. Ensure Path(self._settings.chroma_db_path).mkdir(parents=True, exist_ok=True)
        # 2. chromadb.PersistentClient(path=self._settings.chroma_db_path)
        # 3. client.get_or_create_collection(
        #        name=self._settings.chroma_collection_name,
        #        metadata={"hnsw:space": "cosine"}   # cosine similarity
        #    )
        # 4. Log successful initialisation with collection name and item count
        
        """
        Create or connect to the persistent ChromaDB client and collection.
        """
        try:
            db_path = Path(self._settings.chroma_db_path)
            db_path.mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(path=str(db_path))
            self._collection = self._client.get_or_create_collection(
                name=self._settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            count = self._collection.count()
            logger.info(
                "ChromaDB initialised successfully | collection='{}' | path='{}' | count={}",
                self._settings.chroma_collection_name,
                str(db_path),
                count,
            )
        except Exception as exc:
            logger.exception("Failed to initialise ChromaDB")
            raise RuntimeError(
                f"Could not initialise ChromaDB at {self._settings.chroma_db_path}"
            ) from exc

    # -----------------------------------------------------------------------
    # Duplicate Detection
    # -----------------------------------------------------------------------

    @staticmethod
    def generate_chunk_id(source: str, chunk_text: str) -> str:
        """
        Generate a deterministic chunk ID from source filename and content.

        Using a content hash ensures two uploads of the same file produce
        the same IDs, making duplicate detection reliable regardless of
        filename changes.

        Parameters
        ----------
        source : str
            The source filename (e.g. 'lstm.md').
        chunk_text : str
            The full text content of the chunk.

        Returns
        -------
        str
            A 16-character hex string derived from SHA-256 of the inputs.
        """
        content = f"{source}::{chunk_text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def check_duplicate(self, chunk_id: str) -> bool:
        """
        Check whether a chunk with this ID already exists in the collection.

        Parameters
        ----------
        chunk_id : str
            The deterministic chunk ID to check.

        Returns
        -------
        bool
            True if the chunk already exists (duplicate). False otherwise.

        Interview talking point: content-addressed deduplication is more
        robust than filename-based deduplication because it detects identical
        content even when files are renamed or re-uploaded.
        """
        # TODO: implement
        # self._collection.get(ids=[chunk_id])
        # Return True if the result contains the ID, False otherwise
        
        try:
            result = self._collection.get(ids=[chunk_id])

            ids = result.get("ids", [])
            if ids and chunk_id in ids:
                return True
            return False
        except Exception as exc:
            logger.exception("Duplicate check failed for chunk_id={}", chunk_id)
            raise RuntimeError(f"Failed duplicate check for chunk_id={chunk_id}") from exc

    # -----------------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------------

    def ingest(self, chunks: list[DocumentChunk]) -> IngestionResult:
        """
        Embed and store a list of DocumentChunks in ChromaDB.

        Checks each chunk for duplicates before embedding. Skips duplicates
        silently and records the count in the returned IngestionResult.

        Parameters
        ----------
        chunks : list[DocumentChunk]
            Prepared chunks with text and metadata. Use DocumentChunker
            to produce these from raw files.

        Returns
        -------
        IngestionResult
            Summary with counts of ingested, skipped, and errored chunks.

        Notes
        -----
        Embeds in batches of 100 to avoid memory issues with large corpora.
        Uses upsert (not add) so re-ingestion of modified content updates
        existing chunks rather than raising an error.

        Interview talking point: batch processing with a configurable
        batch size is a production pattern that prevents OOM errors when
        ingesting large document sets.
        """
        # TODO: implement
        # result = IngestionResult()
        # For each chunk:
        #   - check_duplicate(chunk.chunk_id) → if True, result.skipped += 1, continue
        #   - embed chunk.chunk_text using self._embeddings.embed_documents([chunk.chunk_text])
        #   - self._collection.upsert(
        #         ids=[chunk.chunk_id],
        #         embeddings=[embedding],
        #         documents=[chunk.chunk_text],
        #         metadatas=[chunk.metadata.to_dict()]
        #     )
        #   - result.ingested += 1
        # Log summary and return result
        
        result = IngestionResult()

        if not chunks:
            logger.info("No chunks supplied for ingestion.")
            return result

        seen_sources: set[str] = set()

        for chunk in chunks:
            try:
                if self.check_duplicate(chunk.chunk_id):
                    result.skipped += 1
                    continue

                embedding = self._embeddings.embed_documents([chunk.chunk_text])[0]

                self._collection.upsert(
                    ids=[chunk.chunk_id],
                    embeddings=[embedding],
                    documents=[chunk.chunk_text],
                    metadatas=[chunk.metadata.to_dict()],
                )

                result.ingested += 1
                seen_sources.add(chunk.metadata.source)

            except Exception as exc:
                error_msg = (
                    f"Failed to ingest chunk '{chunk.chunk_id}' "
                    f"from source '{chunk.metadata.source}': {exc}"
                )
                logger.exception(error_msg)
                result.errors.append(error_msg)

        result.document_ids = sorted(seen_sources)

        logger.info(
            "Ingestion complete | ingested={} | skipped={} | errors={}",
            result.ingested,
            result.skipped,
            len(result.errors),
        )
        return result

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        k: int | None = None,
        topic_filter: str | None = None,
        difficulty_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Applies similarity threshold filtering — chunks below
        settings.similarity_threshold are excluded from results.

        Parameters
        ----------
        query_text : str
            The user query or rewritten query to retrieve against.
        k : int, optional
            Number of chunks to retrieve. Defaults to settings.retrieval_k.
        topic_filter : str, optional
            Restrict retrieval to a specific topic (e.g. 'LSTM').
            Maps to ChromaDB where-filter on metadata.topic.
        difficulty_filter : str, optional
            Restrict retrieval to a difficulty level.
            Maps to ChromaDB where-filter on metadata.difficulty.

        Returns
        -------
        list[RetrievedChunk]
            Chunks sorted by similarity score descending.
            Empty list if no chunks meet the similarity threshold.

        Interview talking point: returning an empty list (not hallucinating)
        when no relevant context exists is the hallucination guard. This is
        a critical production RAG pattern — the system must know what it
        does not know.
        """
        # TODO: implement
        # k = k or self._settings.retrieval_k
        # Build where_filter dict from topic_filter and difficulty_filter if provided
        # Embed query_text using self._embeddings.embed_query(query_text)
        # self._collection.query(
        #     query_embeddings=[query_embedding],
        #     n_results=k,
        #     where=where_filter,      # None if no filters
        #     include=["documents", "metadatas", "distances"]
        # )
        # Convert distances to similarity scores: score = 1 - distance (for cosine)
        # Filter out chunks below self._settings.similarity_threshold
        # Return list of RetrievedChunk objects sorted by score descending
        
        k = k or self._settings.retrieval_k

        where_filter = {}
        if topic_filter:
            where_filter["topic"] = topic_filter
        if difficulty_filter:
            where_filter["difficulty"] = difficulty_filter

        if not where_filter:
            where_filter = None

        try:
            query_embedding = self._embeddings.embed_query(query_text)

            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

            ids_list = results.get("ids", [[]])
            docs_list = results.get("documents", [[]])
            metas_list = results.get("metadatas", [[]])
            distances_list = results.get("distances", [[]])

            if not ids_list or not ids_list[0]:
                return []

            retrieved_chunks: list[RetrievedChunk] = []

            for chunk_id, doc, meta, distance in zip(
                ids_list[0],
                docs_list[0],
                metas_list[0],
                distances_list[0],
                strict=False,
            ):
                score = 1 - float(distance)

                if score < self._settings.similarity_threshold:
                    continue

                retrieved_chunks.append(
                    RetrievedChunk(
                        chunk_id=chunk_id,
                        chunk_text=doc,
                        metadata=ChunkMetadata.from_dict(meta),
                        score=score,
                    )
                )

            retrieved_chunks.sort(key=lambda x: x.score, reverse=True)
            return retrieved_chunks

        except Exception as exc:
            logger.exception("Vector query failed for query='{}'", query_text)
            raise RuntimeError(f"Failed to query vector store: {query_text}") from exc

    # -----------------------------------------------------------------------
    # Corpus Inspection
    # -----------------------------------------------------------------------

    def list_documents(self) -> list[dict]:
        """
        Return a list of all unique source documents in the collection.

        Used by the UI to populate the document viewer panel.

        Returns
        -------
        list[dict]
            Each item contains: source (str), topic (str), chunk_count (int).
        """
        # TODO: implement
        # Query all metadata from the collection
        # Group by metadata["source"] and count chunks per source
        # Return sorted list of dicts
        try:
            results = self._collection.get(include=["metadatas"])
            metadatas = results.get("metadatas", [])
 
            if not metadatas:
                return []
 
            # Group by source filename
            doc_map: dict[str, dict] = {}
            for meta in metadatas:
                source = meta.get("source", "unknown")
                if source not in doc_map:
                    doc_map[source] = {
                        "source": source,
                        "topic": meta.get("topic", "unknown"),
                        "chunk_count": 0,
                    }
                doc_map[source]["chunk_count"] += 1
 
            return sorted(doc_map.values(), key=lambda d: d["source"])
 
        except Exception as exc:
            logger.exception("Failed to list documents")
            raise RuntimeError("Failed to list documents") from exc

    def get_document_chunks(self, source: str) -> list[DocumentChunk]:
        """
        Retrieve all chunks belonging to a specific source document.

        Used by the document viewer to display document content.

        Parameters
        ----------
        source : str
            The source filename to retrieve chunks for.

        Returns
        -------
        list[DocumentChunk]
            All chunks from this source, ordered by their position
            in the original document.
        """
        # TODO: implement
        # self._collection.get(where={"source": source}, include=["documents", "metadatas"])
        # Reconstruct DocumentChunk objects from results
        try:
            results = self._collection.get(
                where={"source": source},
                include=["documents", "metadatas"],
            )
 
            ids = results.get("ids", [])
            docs = results.get("documents", [])
            metas = results.get("metadatas", [])
 
            if not ids:
                return []
 
            chunks: list[DocumentChunk] = []
            for chunk_id, doc_text, meta in zip(ids, docs, metas, strict=False):
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        chunk_text=doc_text,
                        metadata=ChunkMetadata.from_dict(meta),
                    )
                )
 
            return chunks
 
        except Exception as exc:
            logger.exception("Failed to get chunks for source='{}'", source)
            raise RuntimeError(f"Failed to get chunks for {source}") from exc

    def get_collection_stats(self) -> dict:
        """
        Return summary statistics about the current collection.

        Used by the UI to show corpus health at a glance.

        Returns
        -------
        dict
            Keys: total_chunks, topics (list), sources (list),
            bonus_topics_present (bool).
        """
        # TODO: implement
        try:
            results = self._collection.get(include=["metadatas"])
            metadatas = results.get("metadatas", [])
 
            topics: set[str] = set()
            sources: set[str] = set()
 
            for meta in metadatas:
                topics.add(meta.get("topic", "unknown"))
                sources.add(meta.get("source", "unknown"))
 
            bonus_set = {"SOM", "BoltzmannMachine", "GAN"}
            bonus_present = bool(topics & bonus_set)
 
            return {
                "total_chunks": len(metadatas),
                "topics": sorted(topics),
                "sources": sorted(sources),
                "bonus_topics_present": bonus_present,
            }
 
        except Exception as exc:
            logger.exception("Failed to get collection stats")
            raise RuntimeError("Failed to get collection stats") from exc

    def delete_document(self, source: str) -> int:
        """
        Remove all chunks from a specific source document.

        Parameters
        ----------
        source : str
            Source filename to remove.

        Returns
        -------
        int
            Number of chunks deleted.
        """
        # TODO: implement
        # self._collection.delete(where={"source": source})
        try:
            # Count chunks before deletion
            existing = self._collection.get(
                where={"source": source},
                include=[],
            )
            count = len(existing.get("ids", []))
 
            if count > 0:
                self._collection.delete(where={"source": source})
                logger.info("Deleted {} chunks for source='{}'", count, source)
 
            return count
 
        except Exception as exc:
            logger.exception("Failed to delete document source='{}'", source)
            raise RuntimeError(f"Failed to delete document {source}") from exc
