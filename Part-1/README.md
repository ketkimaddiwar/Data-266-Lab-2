# Multimodal RAG Chatbot System

A Retrieval-Augmented Generation (RAG) pipeline for medical document question answering using hybrid retrieval, cross-encoder reranking, and OpenAI GPT-4o.

## Overview

This system processes PDF documents, extracts text and figures, and answers questions using:
- Hybrid retrieval (BM25 + semantic search)
- Cross-encoder reranking for improved relevance
- Conversation-aware query rewriting
- Automatic figure/table citation

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Required Python packages (see requirements.txt)

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd MultiModal-Chatbot-System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

1. Ensure input files are in place:
   - `data/WHO_document.pdf` - Source PDF document
   - `data/lab2_questions.csv` - Questions CSV file

2. Run the pipeline:
```bash
python rag_pipeline.py
```

3. Results will be saved to:
   - `results/submission.csv` - Generated answers with figure references

## Configuration

Key configuration parameters can be modified in `rag_pipeline.py`:

- `CHUNK_SIZE`: Text chunk size in words (default: 400)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 150)
- `INITIAL_RETRIEVAL_K`: Initial retrieval candidates (default: 50)
- `RERANK_TOP_K`: Candidates after reranking (default: 15)
- `FINAL_TOP_K`: Final chunks for generation (default: 8)
- `TOP_K_FIGURES`: Number of figures to retrieve (default: 3)
- `TEMPERATURE`: Generation temperature (default: 0.1)
- `MAX_TOKENS`: Maximum tokens per response (default: 800)

## Output Format

The submission CSV contains:
- `id`: Question ID
- `conversation_id`: Conversation identifier
- `question_id`: Question number within conversation
- `answer`: Generated answer text
- `figure_references`: Comma-separated figure/table numbers or '0'

## Architecture

The pipeline consists of:
1. PDFProcessor: Extracts text, figures, and tables
2. SmartChunker: Creates overlapping text chunks
3. HybridRetriever: Combines BM25 and semantic search
4. QueryRewriter: Rewrites queries with conversation context
5. FigureManager: Retrieves relevant figures/tables
6. ResponseGenerator: Generates answers with citations
7. RAGPipeline: Orchestrates the complete pipeline

## Requirements

See `requirements.txt` for complete list. Key dependencies:
- openai
- sentence-transformers
- chromadb
- pypdf
- rank-bm25
- pandas
- numpy

## License

See repository license file.

