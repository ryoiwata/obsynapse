"""
Markdown-Aware Hierarchical Chunking Pipeline.

This module implements a two-step chunking approach:
1. MarkdownHeaderTextSplitter to split by headers (#, ##, ###)
2. RecursiveCharacterTextSplitter on resulting splits with
   code block protection
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize logger
logger = logging.getLogger(__name__)


class MarkdownChunker:
    """
    Hierarchical chunker for markdown documents that preserves structure
    and protects code blocks from being split.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        min_chunk_size: int = 100,
    ):
        """
        Initialize the markdown chunker.

        Args:
            chunk_size: Maximum size of chunks in characters
            chunk_overlap: Overlap between chunks in characters
            min_chunk_size: Minimum chunk size to keep (filters "thin" chunks)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        # Step A: MarkdownHeaderTextSplitter
        # Headers to split on: #, ##, ###
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )

        # Step B: RecursiveCharacterTextSplitter
        # Separators in order of preference
        separators = ["\n\n", "\n", " ", ""]
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            # Custom function to protect code blocks
            length_function=len,
            is_separator_regex=False,
        )

    def _protect_code_blocks(self, text: str) -> Tuple[str, List[str]]:
        """
        Protect code blocks from being split by replacing with placeholders.

        Args:
            text: Markdown text that may contain code blocks

        Returns:
            Tuple of (protected_text, code_blocks) where:
            - protected_text: Text with code blocks replaced by placeholders
            - code_blocks: List of original code blocks
        """
        code_block_pattern = re.compile(
            r"```[\s\S]*?```", re.MULTILINE
        )
        code_blocks = []
        placeholder_template = "CODE_BLOCK_PLACEHOLDER_{}"

        def replace_with_placeholder(match):
            code_blocks.append(match.group(0))
            return placeholder_template.format(len(code_blocks) - 1)

        protected_text = code_block_pattern.sub(
            replace_with_placeholder, text
        )
        return protected_text, code_blocks

    def _restore_code_blocks(
        self, text: str, code_blocks: List[str]
    ) -> str:
        """
        Restore code blocks from placeholders.

        Args:
            text: Text with placeholders
            code_blocks: List of original code blocks

        Returns:
            Text with code blocks restored
        """
        for i, code_block in enumerate(code_blocks):
            placeholder = f"CODE_BLOCK_PLACEHOLDER_{i}"
            text = text.replace(placeholder, code_block)
        return text

    def chunk_markdown(
        self,
        markdown_text: str,
        book_title: Optional[str] = None,
        chapter_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Chunk markdown text using hierarchical approach.

        Args:
            markdown_text: The markdown content to chunk
            book_title: Optional book title for metadata
            chapter_name: Optional chapter name for metadata

        Returns:
            List of chunk dictionaries with metadata
        """
        logger.debug("Starting chunking process for markdown text")
        logger.debug(f"Text length: {len(markdown_text)} characters")

        # Step A: Split by headers
        header_splits = self.header_splitter.split_text(markdown_text)
        logger.debug(f"Split into {len(header_splits)} header sections")

        all_chunks = []
        chunk_index = 0

        for header_split in header_splits:
            # Extract metadata from header split
            metadata = header_split.metadata.copy()
            content = header_split.page_content

            # Add book and chapter metadata if provided
            if book_title:
                metadata["book_title"] = book_title
            if chapter_name:
                metadata["chapter_name"] = chapter_name

            # Protect code blocks before recursive splitting
            protected_content, code_blocks = self._protect_code_blocks(
                content
            )

            # Step B: Recursive split on the header-level content
            recursive_splits = self.recursive_splitter.split_text(
                protected_content
            )

            for split in recursive_splits:
                # Restore code blocks
                restored_content = self._restore_code_blocks(
                    split, code_blocks
                )

                # Filter out "thin" chunks
                if len(restored_content.strip()) < self.min_chunk_size:
                    logger.warning(
                        f"Skipping chunk below minimum size "
                        f"({len(restored_content.strip())} < "
                        f"{self.min_chunk_size} characters)"
                    )
                    continue

                # Extract section name from metadata
                section_name = None
                if "Header 1" in metadata:
                    section_name = metadata["Header 1"]
                elif "Header 2" in metadata:
                    section_name = metadata["Header 2"]
                elif "Header 3" in metadata:
                    section_name = metadata["Header 3"]

                chunk = {
                    "content": restored_content.strip(),
                    "book_title": metadata.get("book_title", ""),
                    "chapter_name": metadata.get("chapter_name", ""),
                    "section_name": section_name or "",
                    "chunk_index": chunk_index,
                    "metadata": metadata,
                }

                all_chunks.append(chunk)
                chunk_index += 1

        logger.info(f"Generated {len(all_chunks)} chunks from markdown text")
        return all_chunks

    def chunk_file(
        self,
        file_path: Path,
        book_title: Optional[str] = None,
        chapter_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Chunk a markdown file.

        Args:
            file_path: Path to the markdown file
            book_title: Optional book title for metadata
            chapter_name: Optional chapter name
                (can be extracted from filename)

        Returns:
            List of chunk dictionaries
        """
        file_path = Path(file_path)
        if not file_path.exists():
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info(f"Processing file: {file_path}")
        logger.debug(f"File path: {file_path.absolute()}")

        with open(file_path, "r", encoding="utf-8") as f:
            markdown_text = f.read()

        logger.debug(f"Read {len(markdown_text)} characters from file")

        # If chapter_name not provided, try to extract from filename
        if not chapter_name:
            chapter_name = file_path.stem
            logger.debug(f"Using filename as chapter_name: {chapter_name}")

        return self.chunk_markdown(
            markdown_text, book_title=book_title, chapter_name=chapter_name
        )


def setup_logging(log_file: str = "ingestion.log") -> None:
    """
    Configure logging to output to both console and file.

    Args:
        log_file: Path to the log file (default: ingestion.log)
    """
    # Clear any existing handlers
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    # Console handler - INFO level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler - DEBUG level
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logger.debug(
        f"Logging configured: console (INFO), file (DEBUG): {log_file}"
    )


def main() -> int:
    """
    Command-line interface for the markdown chunker.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        description="Chunk markdown files using hierarchical approach",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python chunker.py input.md --book-title "AI Engineering"
  python chunker.py input.md --chunk-size 2000 --overlap 200
  python chunker.py input.md --min-size 150
        """,
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Path to the markdown file to chunk",
    )

    parser.add_argument(
        "--book-title",
        type=str,
        default=None,
        help="Title of the book (optional)",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum size of chunks in characters (default: 1000)",
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=150,
        help="Overlap between chunks in characters (default: 150)",
    )

    parser.add_argument(
        "--min-size",
        type=int,
        default=100,
        help="Minimum chunk size to keep, filters small chunks (default: 100)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help=(
            "Output JSON file path "
            "(default: input filename with .json extension)"
        ),
    )

    parser.add_argument(
        "--log-file",
        type=str,
        default="ingestion.log",
        help="Path to log file (default: ingestion.log)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_file=args.log_file)

    try:
        # Validate input file
        if not args.input.exists():
            logger.error(f"Input file not found: {args.input}")
            return 1

        # Initialize chunker with provided parameters
        logger.info(
            f"Initializing chunker: chunk_size={args.chunk_size}, "
            f"overlap={args.overlap}, min_size={args.min_size}"
        )
        chunker = MarkdownChunker(
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap,
            min_chunk_size=args.min_size,
        )

        # Chunk the file
        logger.info(f"Starting chunking process for: {args.input}")
        chunks = chunker.chunk_file(
            args.input, book_title=args.book_title
        )

        # Output results
        logger.info(f"Successfully generated {len(chunks)} chunks")
        if chunks:
            logger.debug("First chunk preview:")
            logger.debug(f"  Content: {chunks[0]['content'][:200]}...")
            logger.debug(f"  Metadata: {chunks[0]['metadata']}")

        # Determine output file path
        if args.output is None:
            # Default to input filename with .json extension
            output_path = args.input.with_suffix(".json")
        else:
            output_path = Path(args.output)

        # Save chunks to JSON file
        try:
            logger.info(f"Saving {len(chunks)} chunks to: {output_path}")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
            logger.info(f"Chunks saved successfully to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save chunks to {output_path}: {e}")
            return 1

        return 0

    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error during chunking: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
