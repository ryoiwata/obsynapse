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

# fastapi: Modern, fast web framework for building APIs
# Used for the Review Interface API to serve flashcards and manage the spaced repetition system
conda install conda-forge::fastapi --yes

# inngest: Event-driven workflow orchestration platform
# Used to coordinate multi-step LLM workflows (Extractor Agent, Flashcard Architect, Pedagogical Critic)
pip install inngest

# python-dotenv: Environment variable management
# Used to load API keys, database paths, and other configuration from .env files
conda install conda-forge::python-dotenv

# qdrant-client: Vector database client for Qdrant
# Alternative or additional vector store option for storing note embeddings and RAG operations
conda install conda-forge::qdrant-client

# uvicorn: ASGI web server implementation
# Used to run the FastAPI Review Interface API server in production and development
conda install conda-forge::uvicorn --yes

# openai: Official OpenAI Python SDK
# Used to interact with OpenAI's LLM APIs (GPT-4o, GPT-4o-mini) for the Agentic Forge flashcard generation
conda install conda-forge::openai --yes

# streamlit: Interactive web application framework
# Alternative UI option for the Review Interface to display flashcards and manage spaced repetition reviews
conda install conda-forge::streamlit --yes

# uv: Fast Python package installer and resolver (written in Rust)
# High-performance alternative to pip for faster package installation and dependency resolution
conda install conda-forge::uv --yes

# markdown-it-py: Markdown parser that converts markdown to AST tokens
# Used by the structure extraction module to parse Obsidian markdown into sections, blocks, and callouts
conda install conda-forge::markdown-it-py --yes

# pydantic: Data validation library using Python type annotations
# Used for creating structured data models (DocumentStructure, Section, Block) for the markdown IR
conda install conda-forge::pydantic --yes


conda install conda-forge::pyyaml --yes

pip install --upgrade pymupdf