"""
Qdrant Vector Database Interface for ObSynapse

This module provides a wrapper around Qdrant for storing and retrieving
vector embeddings of Obsidian notes. It's used for RAG (Retrieval Augmented
Generation) operations to find related notes and provide contextual hints
for flashcard generation.

Example:
    >>> from src.db.vector_db import QdrantStorage
    >>> storage = QdrantStorage(collection="obsynapse_notes", dim=1536)
    >>> storage.upsert(
    ...     ids=["note1_chunk0"],
    ...     vectors=[[0.1, 0.2, ...]],
    ...     payloads=[{"text": "Note content", "source": "path/to/note.md"}]
    ... )
    >>> results = storage.search(query_vector=[0.1, 0.2, ...], top_k=5)
"""

import json
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from typing import List, Dict, Any, Union

# Try to import Qdrant exceptions for better error handling
try:
    from qdrant_client.http.exceptions import UnexpectedResponse
except ImportError:
    # Fallback if exception class is not available
    UnexpectedResponse = None


class QdrantStorage:
    """
    Wrapper class for Qdrant vector database operations.

    Manages vector embeddings of Obsidian note chunks, enabling semantic
    search to find related content for RAG-based flashcard generation.

    Attributes:
        client (QdrantClient): The Qdrant client instance
        collection (str): Name of the Qdrant collection

    Args:
        url (str): Qdrant server URL (default: "http://localhost:6333")
        collection (str): Collection name for storing vectors (default: "docs")
        dim (int): Vector dimension size (default: 3072)
            Common values:
            - 1536 for OpenAI text-embedding-3-small
            - 3072 for OpenAI text-embedding-3-large
            - 768 for sentence-transformers models
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection: str = "docs",
        dim: int = 3072
    ):
        """
        Initialize Qdrant client and create collection if it doesn't exist.

        Args:
            url: Qdrant server URL
            collection: Collection name for storing vectors
            dim: Vector dimension size
        """
        # Initialize Qdrant client with 30 second timeout
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection

        # Create collection if it doesn't exist
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=dim,
                    # Cosine similarity for semantic search
                    distance=Distance.COSINE
                ),
            )

    def upsert(
        self,
        ids: List[Union[str, int, uuid.UUID]],
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]]
    ) -> None:
        """
        Insert or update vectors in the Qdrant collection.

        This method is used to store note chunks with their embeddings.
        If a point with the same ID exists, it will be updated.

        Args:
            ids: List of unique identifiers for each vector point
            vectors: List of vector embeddings (each is a list of floats)
            payloads: List of metadata dictionaries containing:
                - text: The actual text content of the chunk
                - source: File path or source identifier
                - Additional metadata (heading, last_updated, etc.)

        Example:
            >>> storage.upsert(
            ...     ids=["note1_chunk0", "note1_chunk1"],
            ...     vectors=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
            ...     payloads=[
            ...         {"text": "First chunk", "source": "notes/study.md"},
            ...         {"text": "Second chunk", "source": "notes/study.md"}
            ...     ]
            ... )
        
        Raises:
            Exception: If Qdrant returns an error during upsert
        """
        # Sanitize payloads to ensure all values are JSON-serializable
        sanitized_payloads = []
        for idx, payload in enumerate(payloads):
            sanitized = {}
            for key, value in payload.items():
                # Ensure all values are JSON-serializable
                if isinstance(value, (str, int, float, bool, type(None))):
                    # For strings, ensure they're valid UTF-8 and not too long
                    if isinstance(value, str):
                        # Qdrant has limits on payload size, truncate if needed
                        if len(value) > 50000:  # 50KB limit per field
                            value = value[:50000]
                        # Ensure valid UTF-8 encoding
                        try:
                            value.encode('utf-8').decode('utf-8')
                        except (UnicodeEncodeError, UnicodeDecodeError):
                            # Replace invalid characters
                            value = value.encode('utf-8', errors='replace').decode('utf-8')
                    sanitized[key] = value
                elif isinstance(value, (list, dict)):
                    # Recursively sanitize nested structures
                    try:
                        json.dumps(value)  # Test if serializable
                        sanitized[key] = value
                    except (TypeError, ValueError) as e:
                        # If not serializable, convert to string
                        sanitized[key] = str(value)
                else:
                    # Convert any other type to string
                    sanitized[key] = str(value)
            
            # Validate the entire payload is JSON-serializable
            try:
                json.dumps(sanitized)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Payload at index {idx} is not JSON-serializable: {e}. "
                    f"Payload keys: {list(sanitized.keys())}"
                ) from e
            
            sanitized_payloads.append(sanitized)
        
        # Convert to PointStruct format required by Qdrant
        points = [
            PointStruct(
                id=ids[i],
                vector=vectors[i],
                payload=sanitized_payloads[i]
            )
            for i in range(len(ids))
        ]
        
        try:
            self.client.upsert(self.collection, points=points)
        except Exception as e:
            # Handle Qdrant API errors with detailed information
            error_msg = str(e)
            
            # Try to extract additional error details
            if UnexpectedResponse and isinstance(e, UnexpectedResponse):
                status_code = getattr(e, 'status_code', 'Unknown')
                # Use getattr with default to safely access attributes
                reason = getattr(e, 'reason', getattr(e, 'message', 'Bad Request'))
                error_msg = f"Unexpected Response: {status_code} ({reason})"
                if hasattr(e, 'body') and e.body:
                    error_msg = f"{error_msg}\nRaw response content:\n{e.body}"
                elif hasattr(e, 'content') and e.content:
                    error_msg = f"{error_msg}\nRaw response content:\n{e.content}"
            elif hasattr(e, 'status_code'):
                # Generic HTTP error
                status_code = getattr(e, 'status_code', 'Unknown')
                reason = getattr(e, 'reason', getattr(e, 'message', ''))
                error_msg = f"Unexpected Response: {status_code} ({reason})"
                if hasattr(e, 'body') and e.body:
                    error_msg = f"{error_msg}\nRaw response content:\n{e.body}"
                elif hasattr(e, 'content') and e.content:
                    error_msg = f"{error_msg}\nRaw response content:\n{e.content}"
                elif hasattr(e, 'response') and hasattr(e.response, 'text'):
                    error_msg = f"{error_msg}\nRaw response content:\n{e.response.text}"
            
            raise Exception(error_msg) from e

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5
    ) -> Dict[str, List[Union[str, List[str]]]]:
        """
        Perform semantic search to find similar vectors.

        Searches the collection for vectors most similar to the query vector
        using cosine similarity. Returns the text content and sources of
        the most relevant chunks, useful for RAG operations.

        Args:
            query_vector: The query embedding vector to search for
            top_k: Number of top results to return (default: 5)

        Returns:
            Dictionary containing:
                - contexts: List of text content from matching chunks
                - sources: List of unique source identifiers

        Example:
            >>> results = storage.search(query_vector=[0.1, 0.2, ...], top_k=3)
            >>> print(results["contexts"])  # List of relevant text chunks
            >>> print(results["sources"])   # List of source file paths
        """
        # Perform vector similarity search
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            with_payload=True,  # Include metadata in results
            limit=top_k
        )

        # Extract text and source information from results
        contexts = []
        sources = set()  # Use set to avoid duplicate sources

        for r in results:
            # Safely extract payload (metadata)
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")

            # Only include results with text content
            if text:
                contexts.append(text)
                if source:
                    sources.add(source)

        return {
            "contexts": contexts,
            # Convert set to list for JSON serialization
            "sources": list(sources)
        }

