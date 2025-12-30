"""
ObSynapse Main Application
FastAPI server with Inngest workflow orchestration for the Agentic Forge.
"""

import logging
import sys
from pathlib import Path

# Add project root to Python path for imports
# This allows running from any directory (e.g., src/api/)
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Standard library imports
import datetime
import uuid

# Third-party imports
from dotenv import load_dotenv
from fastapi import FastAPI
import inngest
import inngest.fast_api

# Local imports (must come after sys.path modification)
from src.ingestion import (  # noqa: E402
    embed_texts,
    extract_structure,
    chunk_from_structure,
    EMBED_DIM
)
from src.db import QdrantStorage  # noqa: E402

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize Inngest client
inngest_client = inngest.Inngest(
    app_id="obsynapse",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)


@inngest_client.create_function(
    fn_id="obsynapse:process_note",
    trigger=inngest.TriggerEvent(event="obsynapse/note_updated")
)
async def process_note(ctx: inngest.Context):
    """
    Process a note that was updated in the vault.

    Loads the markdown file, chunks it, generates embeddings, and stores
    them in the vector database. This is the first step in the ingestion
    pipeline.
    """
    event_data = ctx.event.data
    file_path = event_data.get("file_path")

    logger = logging.getLogger("obsynapse.process_note")
    logger.info(f"Processing note: {file_path}")

    if not file_path:
        logger.error("No file_path provided in event data")
        return {
            "file_path": None,
            "status": "error",
            "error": "Missing file_path"
        }

    try:
        # Step 1: Extract document structure (sections, blocks, frontmatter)
        # This step parses the markdown into a structured IR
        structure_result = await ctx.step.run(
            "extract-structure",
            lambda: _extract_structure_step(file_path)
        )

        source_id = structure_result["source_id"]
        sections_count = structure_result["sections_count"]
        # Note: We don't need the full structure for embedding,
        # just the source_id and metadata

        # Step 2: Load and chunk the markdown file
        # This step will be visible in the Inngest UI with chunk details
        load_result = await ctx.step.run(
            "load-and-chunk",
            lambda: _load_and_chunk_step(file_path)
        )

        chunks = load_result["chunks"]
        chunk_metadata = load_result.get("chunk_metadata", [])

        if not chunks:
            logger.warning(f"No chunks extracted from {file_path}")
            return {
                "file_path": file_path,
                "status": "processed",
                "chunks_count": 0,
                "sections_count": sections_count,
                "message": "No content to process"
            }

        # Step 3: Generate embeddings and store in vector database
        # This step will be visible in the Inngest UI
        embed_result = await ctx.step.run(
            "embed-and-upsert",
            lambda: _embed_and_upsert_step(
                chunks=chunks,
                chunk_metadata=chunk_metadata,
                source_id=source_id,
                file_path=file_path
            )
        )

        # Step 4: Finalization
        # This step completes the process
        await ctx.step.run(
            "finalization",
            lambda: {
                "status": "completed",
                "chunks_count": len(chunks),
                "sections_count": sections_count,
                "embeddings_count": embed_result["embeddings_count"],
                "stored_count": embed_result["stored_count"],
                "message": (
                    "Successfully processed and stored in vector database"
                )
            }
        )

        logger.info("Note processing complete, ready for concept extraction")

        return {
            "file_path": file_path,
            "status": "processed",
            "chunks_count": len(chunks),
            "sections_count": sections_count,
            "embeddings_count": embed_result["embeddings_count"],
            "stored_count": embed_result["stored_count"],
            "message": (
                "Successfully processed and stored in vector database"
            )
        }

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return {
            "file_path": file_path,
            "status": "error",
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"Error processing note {file_path}: {e}", exc_info=True)
        return {
            "file_path": file_path,
            "status": "error",
            "error": str(e)
        }


def _extract_structure_step(file_path: str) -> dict:
    """
    Helper function for the extract-structure step.
    Returns document structure with chapter and subsections for UI.
    """
    logger = logging.getLogger("obsynapse.process_note.extract_structure")
    logger.info(f"Extracting structure from markdown file: {file_path}")

    doc_structure = extract_structure(file_path)

    file_path_obj = Path(file_path)
    source_id = str(file_path_obj.resolve())

    # Quick summary without deep iteration
    if doc_structure.chapter is None:
        logger.warning("No chapter found in document")
        return {
            "source_id": source_id,
            "sections_count": 0,
            "subsections_count": 0,
            "total_blocks": 0,
            "block_counts": {},
            "subsection_summary": [],
            "callouts": [],
            "frontmatter": doc_structure.frontmatter
        }

    chapter = doc_structure.chapter
    subsections_count = len(chapter.subsections)
    logger.info(
        f"Extracted chapter '{chapter.title}' "
        f"with {subsections_count} subsections"
    )

    # Quick block counting (limit iteration depth)
    block_counts = {}
    total_blocks = len(chapter.blocks_before_subsections)
    for block in chapter.blocks_before_subsections:
        block_counts[block.type] = block_counts.get(block.type, 0) + 1

    # Limit subsection processing for performance
    max_subsections_to_process = 20
    subsection_summary = []
    callouts = []

    for subsection in chapter.subsections[:max_subsections_to_process]:
        subsection_blocks = len(subsection.blocks)
        for subhead in subsection.subheads:
            subsection_blocks += len(subhead.blocks)
            for block in subhead.blocks:
                total_blocks += 1
                block_counts[block.type] = block_counts.get(block.type, 0) + 1
                if block.type == "callout" and len(callouts) < 10:
                    callouts.append({
                        "kind": block.meta.get("kind", "unknown"),
                        "title": block.meta.get("title", ""),
                        "subsection": subsection.title,
                        "subhead": subhead.title,
                        "text_preview": (
                            block.text[:200] + "..."
                            if len(block.text) > 200
                            else block.text
                        )
                    })

        for block in subsection.blocks:
            total_blocks += 1
            block_counts[block.type] = block_counts.get(block.type, 0) + 1
            if block.type == "callout" and len(callouts) < 10:
                callouts.append({
                    "kind": block.meta.get("kind", "unknown"),
                    "title": block.meta.get("title", ""),
                    "subsection": subsection.title,
                    "text_preview": (
                        block.text[:200] + "..."
                        if len(block.text) > 200
                        else block.text
                    )
                })

        subsection_summary.append({
            "order_index": subsection.order_index,
            "title": subsection.title,
            "subheads_count": len(subsection.subheads),
            "blocks_count": subsection_blocks
        })

    # Process remaining subsections for block counts only (no summaries)
    for subsection in chapter.subsections[max_subsections_to_process:]:
        for subhead in subsection.subheads:
            for block in subhead.blocks:
                total_blocks += 1
                block_counts[block.type] = block_counts.get(block.type, 0) + 1
        for block in subsection.blocks:
            total_blocks += 1
            block_counts[block.type] = block_counts.get(block.type, 0) + 1

    return {
        "source_id": source_id,
        "sections_count": subsections_count,  # For backwards compatibility
        "subsections_count": subsections_count,
        "total_blocks": total_blocks,
        "block_counts": block_counts,
        "subsection_summary": subsection_summary,
        "callouts": callouts,
        "frontmatter": doc_structure.frontmatter,
        "chapter_title": chapter.title
    }


def _load_and_chunk_step(file_path: str) -> dict:
    """
    Helper function for the load-and-chunk step.
    Returns chunks and source ID for visibility in Inngest UI.
    Uses structure-based chunking that respects heading boundaries.
    """
    logger = logging.getLogger("obsynapse.process_note.load_and_chunk")
    logger.info(f"Loading and chunking markdown file: {file_path}")

    # Extract structure and chunk from it
    doc_structure = extract_structure(file_path)
    chunks, chunk_metadata = chunk_from_structure(doc_structure)

    logger.info(f"Created {len(chunks)} chunks from {file_path}")

    file_path_obj = Path(file_path)
    source_id = str(file_path_obj.resolve())

    # Return result with chunks visible in UI
    # Limit chunk preview to first 500 chars for UI display
    chunk_previews = [
        (chunk[:500] + "..." if len(chunk) > 500 else chunk)
        for chunk in chunks[:10]  # Show first 10 chunks
    ]

    # Preview metadata for first 10 chunks
    chunk_metadata_previews = chunk_metadata[:10]

    return {
        "chunks": chunks,
        "chunks_count": len(chunks),
        "source_id": source_id,
        "chunk_previews": chunk_previews,
        "chunk_metadata": chunk_metadata,  # Full metadata for embedding step
        "chunk_metadata_previews": chunk_metadata_previews,
        "total_previewed": min(10, len(chunks))
    }


def _embed_and_upsert_step(
    chunks: list,
    chunk_metadata: list,
    source_id: str,
    file_path: str
) -> dict:
    """
    Helper function for the embed-and-upsert step.
    Generates embeddings and stores them in Qdrant.
    """
    logger = logging.getLogger("obsynapse.process_note.embed_and_upsert")
    logger.info(f"Generating embeddings for {len(chunks)} chunks")

    # Generate embeddings
    embeddings = embed_texts(chunks)
    logger.info(f"Generated {len(embeddings)} embeddings")

    # Initialize storage
    storage = QdrantStorage(
        collection="obsynapse_notes",
        dim=EMBED_DIM
    )

    # Prepare data for upsert
    file_path_obj = Path(file_path)
    relative_path = source_id

    # Generate deterministic UUID namespace from file path
    path_namespace = uuid.uuid5(
        uuid.NAMESPACE_URL,
        relative_path
    )

    chunk_ids = []
    payloads = []

    for i, chunk in enumerate(chunks):
        # Use chunk_id from metadata if available, otherwise generate
        if (i < len(chunk_metadata) and
                chunk_metadata[i].get("chunk_id")):
            # Use chunk_id hash to create deterministic UUID
            chunk_id_str = chunk_metadata[i]["chunk_id"]
            # Create UUID5 from the chunk_id string
            chunk_id = uuid.uuid5(
                path_namespace,
                f"chunk_{chunk_id_str}"
            )
        else:
            # Fallback: generate deterministic UUID
            chunk_id = uuid.uuid5(
                path_namespace,
                f"chunk_{i}"
            )
        chunk_ids.append(chunk_id)

        # Ensure chunk text is a string and not too large
        chunk_text = str(chunk) if chunk else ""
        if len(chunk_text) > 100000:  # 100KB limit
            original_len = len(str(chunk))
            chunk_text = chunk_text[:100000]
            logger.warning(
                f"Truncated chunk {i} from {original_len} to 100000 chars"
            )

        # Prepare payload with metadata if available
        try:
            relative_path_clean = relative_path.encode('utf-8').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            relative_path_clean = str(file_path_obj)

        # Base payload
        payload = {
            "text": chunk_text,
            "source": relative_path_clean,
            "chunk_index": str(i)
        }

        # Add chunk metadata if available
        if i < len(chunk_metadata):
            meta = chunk_metadata[i]
            payload.update({
                "chunk_id": meta.get("chunk_id", ""),
                "content_hash": meta.get("content_hash", ""),
                "chapter_title": meta.get("chapter_title", ""),
                "subsection_index": (
                    str(meta["subsection_index"])
                    if meta.get("subsection_index") is not None
                    else None
                ),
                "subsection_title": meta.get("subsection_title"),
                "subhead_path": (
                    " > ".join(meta["subhead_path"])
                    if meta.get("subhead_path")
                    else None
                ),
                "block_types": ",".join(meta.get("block_types", []))
            })

        payloads.append(payload)

    # Upsert in batches
    batch_size = 100
    total_stored = 0
    for i in range(0, len(chunk_ids), batch_size):
        batch_ids = chunk_ids[i:i + batch_size]
        batch_vectors = embeddings[i:i + batch_size]
        batch_payloads = payloads[i:i + batch_size]

        batch_num = i // batch_size + 1
        logger.info(
            f"Upserting batch {batch_num} ({len(batch_ids)} chunks)"
        )
        try:
            storage.upsert(
                ids=batch_ids,
                vectors=batch_vectors,
                payloads=batch_payloads
            )
            total_stored += len(batch_ids)
        except Exception as e:
            logger.error(
                f"Error upserting batch {batch_num}: {e}",
                exc_info=True
            )
            raise

    logger.info(f"Stored {total_stored} chunks in vector database")

    return {
        "embeddings_count": len(embeddings),
        "stored_count": total_stored,
        "batches_processed": (len(chunk_ids) + batch_size - 1) // batch_size
    }


@inngest_client.create_function(
    fn_id="obsynapse:generate_flashcards",
    trigger=inngest.TriggerEvent(event="obsynapse/generate_flashcards")
)
async def generate_flashcards(ctx: inngest.Context):
    """
    Flashcard Architect: Formats concepts into Q&A, Cloze deletions,
    or Concept Maps.
    """
    event_data = ctx.event.data
    concepts = event_data.get("concepts", [])
    # TODO: Use note_context for better flashcard generation

    logger = logging.getLogger("obsynapse.generate_flashcards")
    logger.info(f"Generating flashcards for {len(concepts)} concepts")

    # TODO: Implement Flashcard Architect using OpenAI
    # For now, return placeholder
    return {
        "flashcards": [],
        "status": "generated"
    }


@inngest_client.create_function(
    fn_id="obsynapse:critic_flashcards",
    trigger=inngest.TriggerEvent(event="obsynapse/critic_flashcards")
)
async def critic_flashcards(ctx: inngest.Context):
    """
    Pedagogical Critic: Reviews cards for clarity, difficulty,
    and hallucinations. If a card is too wordy, it sends it back
    for simplification.
    """
    event_data = ctx.event.data
    flashcards = event_data.get("flashcards", [])

    logger = logging.getLogger("obsynapse.critic_flashcards")
    logger.info(f"Reviewing {len(flashcards)} flashcards")

    # TODO: Implement Pedagogical Critic using OpenAI
    # For now, return placeholder
    return {
        "approved_flashcards": [],
        "needs_revision": [],
        "status": "reviewed"
    }


# Initialize FastAPI app
app = FastAPI(title="ObSynapse API", version="0.1.0")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "ObSynapse",
        "status": "running",
        "timestamp": datetime.datetime.now().isoformat()
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/api/trigger/process-note")
async def trigger_process_note(file_path: str):
    """
    Trigger the process_note Inngest function.

    Args:
        file_path: Path to the markdown file to process (query parameter)

    Returns:
        Event ID and status

    Example:
        POST /api/trigger/process-note?file_path=/path/to/note.md
    """
    # Send event to Inngest
    event_id = inngest_client.send_sync(
        inngest.Event(
            name="obsynapse/note_updated",
            data={"file_path": file_path}
        )
    )
    return {
        "status": "triggered",
        "event_id": event_id,
        "file_path": file_path,
        "message": "Note processing triggered"
    }


# Serve Inngest functions with FastAPI
inngest.fast_api.serve(
    app,
    inngest_client,
    functions=[
        process_note,
        generate_flashcards,
        critic_flashcards
    ]
)
