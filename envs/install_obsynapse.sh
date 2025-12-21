#!/bin/bash
# obsynapse Environment Setup Script
# This script creates a conda environment with all dependencies needed for the obsynapse project

# Deactivate any currently active conda environment
conda deactivate

# Create conda environment with Python 3.12 (compatible with DBT)
# Using local path ./obsynapse instead of global environment
conda create -p ./obsynapse python=3.12 --yes

# Activate the newly created environment
conda activate ./obsynapse

# watchdog: File system event monitoring library
# Used by VaultWatcher to detect when .md files are created or modified in the Obsidian vault
conda install conda-forge::watchdog --yes

# python-frontmatter: YAML frontmatter parser for markdown files
# Used to extract metadata (tags, title, etc.) from Obsidian notes to filter for #study tagged files
pip install python-frontmatter

# chromadb: Local vector database for storing embeddings
# Used to store and query chunked note embeddings for RAG integration and flashcard generation
conda install conda-forge::chromadb --yes

# langchain-text-splitters: Text chunking utilities
# Used to split markdown notes into 500-800 character chunks with 10% overlap for embedding
conda install conda-forge::langchain-text-splitters --yes