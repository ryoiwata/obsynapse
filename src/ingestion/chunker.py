"""
Markdown-Aware Hierarchical Chunking Pipeline.

This module implements a two-step chunking approach:
1. MarkdownHeaderTextSplitter to split by headers (#, ##, ###)
2. RecursiveCharacterTextSplitter on resulting splits with code block protection
"""

import re
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


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

    def _protect_code_blocks(self, text: str) -> str:
        """
        Protect code blocks from being split by replacing them with placeholders.

        Args:
            text: Markdown text that may contain code blocks

        Returns:
            Text with code blocks replaced by placeholders
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
        # Step A: Split by headers
        header_splits = self.header_splitter.split_text(markdown_text)

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
            chapter_name: Optional chapter name (can be extracted from filename)

        Returns:
            List of chunk dictionaries
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            markdown_text = f.read()

        # If chapter_name not provided, try to extract from filename
        if not chapter_name:
            chapter_name = file_path.stem

        return self.chunk_markdown(
            markdown_text, book_title=book_title, chapter_name=chapter_name
        )


def main():
    """Example usage of the chunker."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python chunker.py <markdown_file> [book_title]")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    book_title = sys.argv[2] if len(sys.argv) > 2 else None

    chunker = MarkdownChunker(chunk_size=1000, chunk_overlap=150)
    chunks = chunker.chunk_file(file_path, book_title=book_title)

    print(f"Generated {len(chunks)} chunks from {file_path}")
    print("\nFirst chunk example:")
    if chunks:
        print(f"Content: {chunks[0]['content'][:200]}...")
        print(f"Metadata: {chunks[0]['metadata']}")


if __name__ == "__main__":
    main()


