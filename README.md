# RAG-based Educational Guidance Assistant

A simple Retrieval-Augmented Generation (RAG) chatbot supporting Ugandan students seeking computer science studies in Germany. Developed as part of a term paper research project at Furtwangen University of Applied Sciences.

## Purpose

Addresses the information gap Ugandan students face when navigating German higher education requirements for computer science programs. Provides personalized guidance on admissions, language requirements, and application procedures.

## Features

- **RAG Architecture**: Contextual responses using retrieval-augmented generation
- **Local LLM**: Runs offline using Ollama with Qwen3-1.7B model
- **Curated Knowledge Base**: Built from official DAAD and university sources
- **Interactive CLI**: Simple command-line interface
- **Evaluation Framework**: Built-in testing with performance metrics

## Tech Stack

- **LangChain**: RAG pipeline framework
- **Chroma**: Vector database for document storage
- **HuggingFace Embeddings**: `all-MiniLM-L6-v2` for semantic search
- **Ollama**: Local LLM runtime
- **PyPDF**: Document processing


## Structure

```
├── app.py              # Main chatbot interface
├── rag_system.py       # Core RAG implementation
├── evaluate.py         # Testing framework
├── requirements.txt    # Dependencies
└── qa_corpus.pdf  # Knowledge base
```

## License

MIT License

## Acknowledgments

- **Supervisor**: Prof. Dr. Simon Albrecht
- **Institution**: Furtwangen University of Applied Sciences
- **Open Source**: LangChain, HuggingFace, Ollama teams
- **Development Support**: ChatGPT and Claude


