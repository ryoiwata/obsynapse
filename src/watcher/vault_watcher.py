"""
ObSynapse Vault Watcher
Event-driven ingestion pipeline for monitoring Obsidian vault and syncing to ChromaDB.
"""

import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import frontmatter
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VaultWatcher(FileSystemEventHandler):
    """
    Monitors Obsidian vault directory for .md file changes and syncs to ChromaDB.
    Only processes files tagged with #study in their frontmatter.
    """
    
    def __init__(
        self,
        vault_path: str,
        chroma_client: chromadb.ClientAPI,
        collection_name: str = "obsynapse_notes"
    ):
        """
        Initialize the VaultWatcher.
        
        Args:
            vault_path: Path to the Obsidian vault directory
            chroma_client: ChromaDB client instance
            collection_name: Name of the ChromaDB collection
        """
        self.vault_path = Path(vault_path).resolve()
        self.chroma_client = chroma_client
        self.collection_name = collection_name
        
        # Initialize or get ChromaDB collection
        try:
            self.collection = self.chroma_client.get_collection(
                name=self.collection_name
            )
            logger.info(f"Connected to existing collection: {self.collection_name}")
        except Exception:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "ObSynapse note embeddings"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=650,  # Average of 500-800
            chunk_overlap=65,  # 10% of 650
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Track processed files to avoid duplicate processing
        self.processed_files: Dict[str, float] = {}
        
        logger.info(f"VaultWatcher initialized for vault: {self.vault_path}")
    
    def _has_study_tag(self, frontmatter_data: Dict[str, Any]) -> bool:
        """
        Check if frontmatter contains #study tag.
        
        Args:
            frontmatter_data: Parsed frontmatter dictionary
            
        Returns:
            True if #study tag is present
        """
        tags = frontmatter_data.get('tags', [])
        if isinstance(tags, str):
            tags = [tags]
        elif not isinstance(tags, list):
            tags = []
        
        # Check for #study or study tag
        for tag in tags:
            if isinstance(tag, str) and ('study' in tag.lower() or tag == '#study'):
                return True
        return False
    
    def _extract_heading(self, content: str) -> Optional[str]:
        """
        Extract the first heading (H1 or H2) from markdown content.
        
        Args:
            content: Markdown content string
            
        Returns:
            First heading found, or None
        """
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
            elif line.startswith('## '):
                return line[3:].strip()
        return None
    
    def _should_process_file(self, file_path: Path) -> bool:
        """
        Check if file should be processed based on modification time.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file should be processed
        """
        if not file_path.exists():
            return False
        
        current_mtime = file_path.stat().st_mtime
        last_processed = self.processed_files.get(str(file_path), 0)
        
        # Only process if file has been modified since last processing
        if current_mtime <= last_processed:
            logger.debug(f"File unchanged: {file_path}")
            return False
        
        return True
    
    def _process_file(self, file_path: Path) -> None:
        """
        Process a markdown file: parse, filter, chunk, and sync to ChromaDB.
        
        Args:
            file_path: Path to the markdown file
        """
        try:
            # Check if file should be processed
            if not self._should_process_file(file_path):
                return
            
            # Read and parse frontmatter
            with open(file_path, 'r', encoding='utf-8') as f:
                post = frontmatter.load(f)
            
            # Check for #study tag
            if not self._has_study_tag(post.metadata):
                logger.info(f"Ignored: {file_path} (no #study tag)")
                return
            
            logger.info(f"Processing: {file_path}")
            
            # Extract content and metadata
            content = post.content
            heading = self._extract_heading(content) or post.metadata.get('title', 'Untitled')
            
            # Chunk the content
            chunks = self.text_splitter.split_text(content)
            logger.info(f"Split into {len(chunks)} chunks")
            
            # Delete existing embeddings for this file
            relative_path = str(file_path.relative_to(self.vault_path))
            try:
                # Query existing documents with this file_path
                existing = self.collection.get(
                    where={"file_path": relative_path}
                )
                if existing['ids']:
                    self.collection.delete(ids=existing['ids'])
                    logger.info(f"Deleted {len(existing['ids'])} existing chunks for {relative_path}")
            except Exception as e:
                logger.warning(f"Error deleting existing chunks: {e}")
            
            # Prepare documents for upsert
            chunk_ids = []
            chunk_texts = []
            chunk_metadatas = []
            
            last_updated = datetime.now().isoformat()
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{relative_path}__chunk_{i}"
                chunk_ids.append(chunk_id)
                chunk_texts.append(chunk)
                chunk_metadatas.append({
                    "file_path": relative_path,
                    "heading": heading,
                    "last_updated": last_updated,
                    "chunk_index": i
                })
            
            # Upsert to ChromaDB
            self.collection.upsert(
                ids=chunk_ids,
                documents=chunk_texts,
                metadatas=chunk_metadatas
            )
            
            # Update processed files tracking
            self.processed_files[str(file_path)] = file_path.stat().st_mtime
            
            logger.info(f"Updated in Database: {relative_path} ({len(chunks)} chunks)")
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}", exc_info=True)
    
    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation events."""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        if file_path.suffix == '.md':
            logger.info(f"File created: {file_path}")
            self._process_file(file_path)
    
    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events."""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        if file_path.suffix == '.md':
            logger.info(f"File modified: {file_path}")
            # Add small delay to avoid processing during file write
            time.sleep(0.5)
            self._process_file(file_path)


def main():
    """
    Main entry point for the ObSynapse Vault Watcher.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ObSynapse Vault Watcher - Monitor Obsidian vault and sync to ChromaDB"
    )
    parser.add_argument(
        '--vault-path',
        type=str,
        required=True,
        help='Path to the Obsidian vault directory'
    )
    parser.add_argument(
        '--chroma-db-path',
        type=str,
        default='./chroma_db',
        help='Path to ChromaDB database directory (default: ./chroma_db)'
    )
    parser.add_argument(
        '--collection-name',
        type=str,
        default='obsynapse_notes',
        help='ChromaDB collection name (default: obsynapse_notes)'
    )
    
    args = parser.parse_args()
    
    # Validate vault path
    vault_path = Path(args.vault_path)
    if not vault_path.exists() or not vault_path.is_dir():
        logger.error(f"Vault path does not exist or is not a directory: {vault_path}")
        return
    
    # Initialize ChromaDB client
    chroma_db_path = Path(args.chroma_db_path)
    chroma_db_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Initializing ChromaDB at: {chroma_db_path}")
    chroma_client = chromadb.PersistentClient(
        path=str(chroma_db_path),
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Initialize VaultWatcher
    watcher = VaultWatcher(
        vault_path=str(vault_path),
        chroma_client=chroma_client,
        collection_name=args.collection_name
    )
    
    # Initialize and start observer
    observer = Observer()
    observer.schedule(watcher, str(vault_path), recursive=True)
    observer.start()
    
    logger.info(f"Watching vault: {vault_path}")
    logger.info("Press Ctrl+C to stop...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping observer...")
        observer.stop()
    
    observer.join()
    logger.info("Observer stopped.")


if __name__ == "__main__":
    main()

