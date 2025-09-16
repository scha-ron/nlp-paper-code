# RAG-based Educational Guidance Assistant

A simple Retrieval-Augmented Generation (RAG) chatbot supporting Ugandan students seeking computer science studies in Germany. Developed as part of a term paper research project at Furtwangen University of Applied Sciences.

## 🎯 Purpose

Addresses the information gap Ugandan students face when navigating German higher education requirements for computer science programs. Provides personalized guidance on admissions, language requirements, and application procedures.

## ✨ Features

- **RAG Architecture**: Contextual responses using retrieval-augmented generation
- **Local LLM**: Runs offline using Ollama with Qwen3-1.7B model
- **Curated Knowledge Base**: Built from official DAAD and university sources
- **Interactive CLI**: Simple command-line interface
- **Evaluation Framework**: Built-in testing with performance metrics

## 🛠 Tech Stack

- **LangChain**: RAG pipeline framework
- **Chroma**: Vector database for document storage
- **HuggingFace Embeddings**: `all-MiniLM-L6-v2` for semantic search
- **Ollama**: Local LLM runtime
- **PyPDF**: Document processing

## 🚀 Quick Start

1. **Prerequisites**
   ```bash
   # Install Ollama and pull model
   ollama pull qwen3:1.7b
   ```

2. **Setup**
   ```bash
   git clone https://github.com/scha-ron/nlp-paper-code.git
   cd nlp-paper-code
   pip install -r requirements.txt
   ```

3. **Run**
   ```bash
   python app.py  # Interactive chatbot
   python evaluate.py  # Run evaluation tests
   ```

## 📁 Structure

```
├── app.py              # Main chatbot interface
├── rag_system.py       # Core RAG implementation
├── evaluate.py         # Testing framework
├── requirements.txt    # Dependencies
└── data/qa_corpus.pdf  # Knowledge base
```

## 🔬 Research Results

- **Reliability**: 100% response rate
- **Response Time**: 33.56s average
- **Performance**: Above-average clarity, moderate accuracy
- **Limitation**: Small dataset (4 QA pairs) affects comprehensiveness

## 🔧 Configuration

- **Chunk Size**: 500 characters
- **Model**: Qwen3-1.7B via Ollama
- **Embeddings**: all-MiniLM-L6-v2
- **Knowledge Base**: PDF in `data/` directory

## 📄 License

MIT License

## 🙏 Acknowledgments

- **Supervisor**: Prof. Dr. Simon Albrecht
- **Institution**: Furtwangen University of Applied Sciences
- **Open Source**: LangChain, HuggingFace, Ollama teams
- **Development Support**: ChatGPT and Claude


