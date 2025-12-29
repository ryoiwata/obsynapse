"""
ObSynapse Main Application
FastAPI server with Inngest workflow orchestration for the Agentic Forge.
"""

import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import datetime
from pathlib import Path

from ..ingestion import load_and_chunk_markdown, embed_texts, EMBED_DIM
from ..db import QdrantStorage

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
        # Step 1: Load and chunk the markdown file
        logger.info(f"Loading and chunking markdown file: {file_path}")
        chunks = await ctx.run(
            "load_and_chunk",
            lambda: load_and_chunk_markdown(file_path)
        )
        logger.info(f"Created {len(chunks)} chunks from {file_path}")

        if not chunks:
            logger.warning(f"No chunks extracted from {file_path}")
            return {
                "file_path": file_path,
                "status": "processed",
                "chunks_count": 0,
                "message": "No content to process"
            }

        # Step 2: Generate embeddings for the chunks
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        embeddings = await ctx.run(
            "generate_embeddings",
            lambda: embed_texts(chunks)
        )
        logger.info(f"Generated {len(embeddings)} embeddings")

        # Step 3: Store in vector database
        logger.info("Storing embeddings in Qdrant")
        storage = await ctx.run(
            "init_storage",
            lambda: QdrantStorage(
                collection="obsynapse_notes",
                dim=EMBED_DIM
            )
        )

        # Prepare data for upsert
        file_path_obj = Path(file_path)
        relative_path = str(file_path_obj)
        chunk_ids = [
            f"{relative_path}__chunk_{i}"
            for i in range(len(chunks))
        ]
        payloads = [
            {
                "text": chunk,
                "source": relative_path,
                "chunk_index": i
            }
            for i, chunk in enumerate(chunks)
        ]

        await ctx.run(
            "upsert_vectors",
            lambda: storage.upsert(
                ids=chunk_ids,
                vectors=embeddings,
                payloads=payloads
            )
        )
        logger.info(f"Stored {len(chunks)} chunks in vector database")

        # Step 4: Trigger next step in workflow (concept extraction)
        # This will be implemented when the Extractor Agent is ready
        logger.info("Note processing complete, ready for concept extraction")

        return {
            "file_path": file_path,
            "status": "processed",
            "chunks_count": len(chunks),
            "embeddings_count": len(embeddings),
            "message": "Successfully processed and stored in vector database"
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
