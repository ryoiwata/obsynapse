# 🧠 ObSynapse

> **Bridging the gap between static notes and long-term mastery.**

ObSynapse is an **Agentic AI pipeline** that monitors your Obsidian vault, extracts high-signal knowledge using Large Language Models (LLMs), and transforms it into a dynamic Spaced Repetition System (SRS). Instead of manually creating flashcards, ObSynapse acts as a cognitive partner that decides *what* you need to remember and *when* you need to see it.

---

## 🚀 The Vision

Most second-brain tools are "write-only"—information goes in but is rarely retrieved. **ObSynapse** closes the loop. By using an agent-based architecture, the system doesn't just copy text; it understands context, synthesizes concepts, and handles the "boring" work of card maintenance.

---

## 🛠️ System Architecture

The project is divided into four modular services:

### 1. The Vault Observer (Ingestion)

* **Function:** A Python-based file watcher that monitors `.md` files.
* **Tech:** `Watchdog`, `PyYAML` (for frontmatter parsing).
* **Logic:** Detects new notes or those tagged with `#study`. It generates a diff to ensure only new content is processed, saving tokens and compute.

### 2. The Agentic Forge (AI Processing)

This is the "brain" of the project. Rather than a single prompt, it uses a **Generator-Critic loop**:

* **Extractor Agent:** Analyzes the Markdown and identifies "Atomic Concepts."
* **Flashcard Architect:** Formats these concepts into Q&A, Cloze deletions, or Concept Maps.
* **Pedagogical Critic:** A secondary agent that reviews cards for clarity, difficulty, and hallucinations. If a card is too "wordy," it is sent back for simplification.

### 3. The Synapse Engine (SRS Logic)

* **Function:** Manages the "forgetting curve."
* **Algorithm:** An implementation of the **SM-2 (SuperMemo)** algorithm.
* **Database:** SQLite via SQLAlchemy to track card health, intervals, and ease factors.

### 4. The Review Interface

* **Function:** A lightweight CLI or Web Dashboard (FastAPI/Streamlit) where the user reviews cards.
* **Syncing:** Future plans include an "Anki-Sync" module to push generated cards directly into the user’s existing Anki decks.

---

## 🧠 AI Engineering Highlights

* **RAG Integration:** Uses Vector embeddings to find related notes, ensuring flashcards have "Contextual Hints" if the user gets stuck.
* **Agentic Self-Correction:** Demonstrates the ability to build multi-step LLM workflows that minimize hallucination.
* **Token Optimization:** Intelligent filtering to ensure the LLM only processes high-value information.

---

## 🗺️ Roadmap

* [ ] **Phase 1:** Local file watcher and basic GPT-4o-mini card generation.
* [ ] **Phase 2:** Implement the Critic Agent to improve card quality.
* [ ] **Phase 3:** Build the SQLite-based Spaced Repetition scheduler.
* [ ] **Phase 4:** Develop a "Review Mode" UI.
* [ ] **Phase 5:** Plugin integration for Obsidian (Optional).

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **AI Orchestration:** LangGraph / LangChain
* **LLM:** OpenAI GPT-4o / Claude 3.5 Sonnet
* **Database:** SQLite & ChromaDB (Vector)
* **API:** FastAPI

---

### How would you like to proceed?

I can help you write the **Python script for the Vault Observer** (to start "listening" to your Obsidian files), or we can draft the **System Prompt** for your "Flashcard Architect" agent!