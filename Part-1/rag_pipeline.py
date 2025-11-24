"""
Advanced Multimodal RAG Pipeline for Medical Document Question Answering.

This module implements a Retrieval-Augmented Generation (RAG) pipeline that processes
medical documents (PDFs) and answers questions using hybrid retrieval (BM25 + semantic
search), cross-encoder reranking, and OpenAI GPT-4o for response generation.

Architecture:
    The pipeline consists of seven main components:
    1. PDFProcessor: Extracts text, figures, and tables from PDF documents
    2. SmartChunker: Creates overlapping text chunks preserving sentence boundaries
    3. HybridRetriever: Combines BM25 keyword search with dense semantic embeddings
    4. QueryRewriter: Uses GPT-4o to rewrite queries with conversation context
    5. FigureManager: Retrieves relevant figures/tables using semantic similarity
    6. ResponseGenerator: Generates answers with automatic figure citation
    7. RAGPipeline: Orchestrates the end-to-end pipeline

Key Features:
    - Hybrid retrieval using Reciprocal Rank Fusion (RRF)
    - Cross-encoder reranking for improved relevance
    - Conversation-aware query rewriting
    - Automatic figure/table citation extraction
    - Page-level location tracking for all extracted content

Usage:
    Set OPENAI_API_KEY environment variable, then run:
        python rag_pipeline.py

The pipeline will process questions from data/lab2_questions.csv and generate
submission.csv in the results/ directory.

Author: MultiModal Chatbot System
Version: 1.0
"""

# Standard library imports
import logging
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import chromadb
import numpy as np
import pandas as pd
from chromadb.config import Settings
from openai import OpenAI
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm.auto import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Enable unbuffered output for better logging in some environments
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

# Constants
RECIPROCAL_RANK_FUSION_K: int = 60  # RRF constant for combining retrieval results
FIGURE_CONTEXT_WINDOW: int = 300  # Characters before/after figure for context
TABLE_CONTEXT_WINDOW_BEFORE: int = 300  # Characters before table
TABLE_CONTEXT_WINDOW_AFTER: int = 500  # Characters after table
CONVERSATION_HISTORY_LIMIT: int = 3  # Maximum conversation turns to keep
QUERY_REWRITE_HISTORY_TURNS: int = 2  # Number of previous turns for query rewriting
QUERY_REWRITE_TEMPERATURE: float = 0.3  # Temperature for query rewriting
QUERY_REWRITE_MAX_TOKENS: int = 150  # Max tokens for query rewriting


@dataclass
class Config:
    """
    Configuration settings for the RAG pipeline.
    
    Attributes:
        PDF_PATH: Path to the input PDF document
        QUESTIONS_CSV: Path to the CSV file containing questions
        OUTPUT_CSV: Path where the submission CSV will be saved
        CHROMA_PERSIST_DIR: Directory for ChromaDB persistence
        OPENAI_API_KEY: OpenAI API key from environment variable
        EMBEDDING_MODEL: Sentence transformer model for embeddings
        RERANK_MODEL: Cross-encoder model for reranking
        LLM_MODEL: OpenAI model for generation and query rewriting
        CHUNK_SIZE: Target chunk size in words
        CHUNK_OVERLAP: Overlap size in words between chunks
        INITIAL_RETRIEVAL_K: Number of initial candidates from hybrid retrieval
        RERANK_TOP_K: Number of candidates after reranking
        FINAL_TOP_K: Final number of chunks used for generation
        TOP_K_FIGURES: Number of top figures to retrieve
        MAX_TOKENS: Maximum tokens for response generation
        TEMPERATURE: Temperature for response generation (lower = more factual)
        REQUEST_DELAY: Delay in seconds between API requests (rate limiting)
    """
    # File paths
    PDF_PATH: str = "./data/WHO_document.pdf"
    QUESTIONS_CSV: str = "./data/lab2_questions.csv"
    OUTPUT_CSV: str = "./results/submission.csv"
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    
    # API configuration
    OPENAI_API_KEY: Optional[str] = os.getenv('OPENAI_API_KEY')
    
    # Model configuration
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    LLM_MODEL: str = "gpt-4o"
    
    # Chunking configuration
    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 150
    
    # Retrieval configuration
    INITIAL_RETRIEVAL_K: int = 50
    RERANK_TOP_K: int = 15
    FINAL_TOP_K: int = 8
    TOP_K_FIGURES: int = 3
    
    # Generation configuration
    MAX_TOKENS: int = 800
    TEMPERATURE: float = 0.1
    
    # Rate limiting
    REQUEST_DELAY: int = 12


# Initialize configuration
config = Config()

# Validate configuration
if not config.OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY environment variable not set!\n"
        "Get one at: https://platform.openai.com/api-keys\n"
        "Set it as OPENAI_API_KEY environment variable"
    )

# Initialize OpenAI client
openai_client = OpenAI(api_key=config.OPENAI_API_KEY)


class PDFProcessor:
    """
    Extracts text, figures, and tables from PDF documents with page-level tracking.
    
    This class processes PDF files to extract:
    - Text content with page number tracking
    - Figure captions and surrounding context
    - Table captions and surrounding context
    - Mapping of figures/tables to their page numbers
    
    Attributes:
        pdf_path: Path to the PDF file
        text_content: Full extracted text content
        page_contents: List of dictionaries containing page number and text
        figures: List of extracted figures with metadata
        tables: List of extracted tables with metadata
        figure_page_map: Dictionary mapping figure/table numbers to page numbers
    """
    
    def __init__(self, pdf_path: str) -> None:
        """
        Initialize the PDF processor.
        
        Args:
            pdf_path: Path to the PDF file to process
        """
        self.pdf_path = pdf_path
        self.text_content: str = ""
        self.page_contents: List[Dict[str, Any]] = []
        self.figures: List[Dict[str, Any]] = []
        self.tables: List[Dict[str, Any]] = []
        self.figure_page_map: Dict[str, int] = {}
    
    def extract_text_with_pages(self) -> List[Dict[str, Any]]:
        """
        Extract text content while maintaining page information.
        
        Returns:
            List of dictionaries, each containing:
                - 'page_num': Page number (1-indexed)
                - 'text': Text content from that page
                
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: For other PDF reading errors
        """
        logger.info("Extracting text with page tracking...")
        
        try:
            reader = PdfReader(self.pdf_path)
            logger.info(f"Found {len(reader.pages)} pages")
            
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text:
                    self.page_contents.append({
                        'page_num': page_num,
                        'text': text
                    })
            
            self.text_content = "\n\n".join([p['text'] for p in self.page_contents])
            logger.info(f"Extracted {len(self.text_content)} characters")
            
        except FileNotFoundError:
            logger.error(f"PDF file not found: {self.pdf_path}")
            raise
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise
            
        return self.page_contents
    
    def extract_figures_and_tables(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extract figures and tables with page location tracking.
        
        Uses regex patterns to identify figure and table captions, then extracts
        surrounding context for better semantic understanding.
        
        Returns:
            Tuple containing:
                - List of figure dictionaries with keys: number, caption, context, page, type
                - List of table dictionaries with keys: number, caption, context, page, type
                
        Note:
            Figure context window: 300 chars before/after
            Table context window: 300 chars before, 500 chars after
        """
        logger.info("Extracting figures and tables with location tracking...")
        
        for page_data in self.page_contents:
            page_num = page_data['page_num']
            text = page_data['text']
            
            # Extract figures using regex pattern
            # Pattern matches: "Figure X.Y: Caption text" with optional multi-line captions
            figure_pattern = r'Figure\s+(\d+\.?\d*)[:\s.]+([^\n]+(?:\n(?!Figure|Table)[^\n]+){0,3})'
            for match in re.finditer(figure_pattern, text, re.IGNORECASE):
                figure_number = match.group(1)
                caption = match.group(2).strip()
                
                # Extract context window around figure
                start_position = max(0, match.start() - FIGURE_CONTEXT_WINDOW)
                end_position = min(len(text), match.end() + FIGURE_CONTEXT_WINDOW)
                context = text[start_position:end_position]
                
                self.figures.append({
                    'number': figure_number,
                    'caption': caption,
                    'context': context,
                    'page': page_num,
                    'type': 'figure'
                })
                self.figure_page_map[figure_number] = page_num
            
            # Extract tables using similar pattern
            table_pattern = r'Table\s+(\d+\.?\d*)[:\s.]+([^\n]+(?:\n(?!Figure|Table)[^\n]+){0,3})'
            for match in re.finditer(table_pattern, text, re.IGNORECASE):
                table_number = match.group(1)
                caption = match.group(2).strip()
                
                # Tables get more context after (for data descriptions)
                start_position = max(0, match.start() - TABLE_CONTEXT_WINDOW_BEFORE)
                end_position = min(len(text), match.end() + TABLE_CONTEXT_WINDOW_AFTER)
                context = text[start_position:end_position]
                
                self.tables.append({
                    'number': table_number,
                    'caption': caption,
                    'context': context,
                    'page': page_num,
                    'type': 'table'
                })
                self.figure_page_map[table_number] = page_num
        
        logger.info(f"Extracted {len(self.figures)} figures, {len(self.tables)} tables")
        return self.figures, self.tables


class SmartChunker:
    """
    Creates intelligent text chunks with sentence boundary preservation.
    
    This chunker splits text into overlapping chunks while:
    - Preserving sentence boundaries (never splits mid-sentence)
    - Maintaining page information
    - Creating overlapping chunks for better context continuity
    
    Attributes:
        chunk_size: Target chunk size in words
        overlap: Overlap size in words between consecutive chunks
    """
    
    def __init__(self, chunk_size: int = 400, overlap: int = 150) -> None:
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target chunk size in words
            overlap: Overlap size in words between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, page_contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create overlapping chunks while preserving page information.
        
        Algorithm:
            1. Split text into sentences using punctuation markers
            2. Build chunks by adding sentences until size limit
            3. When limit reached, create overlap by keeping last N words
            4. Continue with overlap as starting point for next chunk
            
        Args:
            page_contents: List of dictionaries with 'page_num' and 'text' keys
            
        Returns:
            List of chunk dictionaries with keys:
                - 'id': Unique chunk identifier
                - 'text': Chunk text content
                - 'page': Page number where chunk originates
                - 'size': Word count of chunk
        """
        logger.info("Creating intelligent chunks...")
        
        chunks: List[Dict[str, Any]] = []
        chunk_id = 0
        
        for page_data in page_contents:
            page_num = page_data['page_num']
            text = page_data['text']
            
            # Normalize whitespace and split into sentences
            text = re.sub(r'\s+', ' ', text).strip()
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            current_chunk: List[str] = []
            current_size = 0
            
            for sentence in sentences:
                words = sentence.split()
                sentence_size = len(words)
                
                # If adding this sentence would exceed chunk size, finalize current chunk
                if current_size + sentence_size > self.chunk_size and current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append({
                        'id': chunk_id,
                        'text': chunk_text,
                        'page': page_num,
                        'size': current_size
                    })
                    chunk_id += 1
                    
                    # Create overlap: keep last N words up to overlap size
                    overlap_sentences = []
                    overlap_size = 0
                    for sentence_item in reversed(current_chunk):
                        sentence_words = sentence_item.split()
                        if overlap_size + len(sentence_words) <= self.overlap:
                            overlap_sentences.insert(0, sentence_item)
                            overlap_size += len(sentence_words)
                        else:
                            break
                    
                    current_chunk = overlap_sentences
                    current_size = overlap_size
                
                current_chunk.append(sentence)
                current_size += sentence_size
            
            # Add final chunk if remaining content
            if current_chunk:
                chunks.append({
                    'id': chunk_id,
                    'text': ' '.join(current_chunk),
                    'page': page_num,
                    'size': current_size
                })
                chunk_id += 1
        
        avg_size = sum(c['size'] for c in chunks) / len(chunks) if chunks else 0
        logger.info(f"Created {len(chunks)} chunks (avg: {avg_size:.0f} words)")
        return chunks


class HybridRetriever:
    """
    Advanced hybrid retrieval combining BM25 keyword search and dense semantic embeddings.
    
    This retriever uses:
    - BM25: Keyword-based sparse retrieval for exact term matching
    - Dense embeddings: Semantic similarity using sentence transformers
    - Reciprocal Rank Fusion (RRF): Combines both retrieval methods
    - Cross-encoder reranking: Final relevance scoring
    
    Attributes:
        embedder: Sentence transformer model for dense embeddings
        reranker: Cross-encoder model for reranking
        chunks: List of indexed text chunks
        chunk_embeddings: Precomputed embeddings for all chunks
        bm25: BM25 index for keyword search
        tokenized_chunks: Tokenized chunks for BM25
        client: ChromaDB client instance
        collection: ChromaDB collection storing chunk embeddings
    """
    
    def __init__(self, embedding_model: str, rerank_model: str) -> None:
        """
        Initialize the hybrid retriever.
        
        Args:
            embedding_model: Name/path of sentence transformer model
            rerank_model: Name/path of cross-encoder reranking model
        """
        logger.info("Initializing Hybrid Retriever...")
        self.embedder = SentenceTransformer(embedding_model)
        self.reranker = CrossEncoder(rerank_model)
        
        self.chunks: List[Dict[str, Any]] = []
        self.chunk_embeddings: Optional[np.ndarray] = None
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_chunks: List[List[str]] = []
        
        # Initialize ChromaDB client
        self.client = chromadb.Client(Settings(
            persist_directory=config.CHROMA_PERSIST_DIR,
            anonymized_telemetry=False
        ))
        
        # Clear existing collection if present
        try:
            self.client.delete_collection("chunks")
        except Exception:
            # Collection doesn't exist, which is fine
            pass
        
        self.collection = self.client.create_collection(
            name="chunks",
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info("Hybrid Retriever initialized")
    
    def index_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Build BM25 and dense indexes for retrieval.
        
        This method:
        1. Generates dense embeddings for all chunks
        2. Stores embeddings in ChromaDB with HNSW indexing
        3. Builds BM25 index for keyword-based retrieval
        
        Args:
            chunks: List of chunk dictionaries to index
        """
        logger.info("Indexing chunks...")
        
        self.chunks = chunks
        chunk_texts = [chunk['text'] for chunk in chunks]
        
        # Generate dense embeddings
        logger.info("  - Generating embeddings...")
        self.chunk_embeddings = self.embedder.encode(
            chunk_texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Store in ChromaDB
        logger.info("  - Storing in ChromaDB...")
        chunk_ids = [f"chunk_{chunk['id']}" for chunk in chunks]
        metadatas = [{'chunk_id': chunk['id'], 'page': chunk['page']} for chunk in chunks]
        embeddings_list = [emb.tolist() for emb in self.chunk_embeddings]
        
        self.collection.add(
            ids=chunk_ids,
            documents=chunk_texts,
            embeddings=embeddings_list,
            metadatas=metadatas
        )
        
        # Build BM25 index
        logger.info("  - Building BM25 index...")
        self.tokenized_chunks = [text.lower().split() for text in chunk_texts]
        self.bm25 = BM25Okapi(self.tokenized_chunks)
        
        logger.info(f"Indexed {len(chunks)} chunks")
    
    def retrieve_hybrid(
        self, 
        query: str, 
        top_k: int = 50
    ) -> List[Tuple[str, int, float]]:
        """
        Combine BM25 and dense retrieval using Reciprocal Rank Fusion (RRF).
        
        RRF Algorithm:
            - Each retrieval method ranks documents
            - RRF score = sum(1 / (k + rank)) for each method
            - Documents appearing in both methods get higher scores
            - Final ranking by combined RRF score
            
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of tuples: (document_text, chunk_id, rrf_score)
            Sorted by RRF score (descending)
        """
        # Dense retrieval using semantic embeddings
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)[0]
        dense_results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        dense_documents = dense_results['documents'][0]
        
        # BM25 keyword-based retrieval
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        
        # Reciprocal Rank Fusion (RRF) to combine results
        rrf_scores: Dict[str, float] = {}
        document_to_index: Dict[str, int] = {}
        
        # Add dense retrieval scores
        for rank, document_text in enumerate(dense_documents):
            rrf_scores[document_text] = 1 / (RECIPROCAL_RANK_FUSION_K + rank + 1)
            # Find corresponding chunk index
            for chunk_index, chunk in enumerate(self.chunks):
                if chunk['text'] == document_text:
                    document_to_index[document_text] = chunk_index
                    break
        
        # Add BM25 scores (combining with dense scores)
        for rank, chunk_index in enumerate(bm25_top_indices):
            document_text = self.chunks[chunk_index]['text']
            rrf_scores[document_text] = rrf_scores.get(document_text, 0) + 1 / (
                RECIPROCAL_RANK_FUSION_K + rank + 1
            )
            document_to_index[document_text] = chunk_index
        
        # Sort by RRF score and return top_k
        sorted_documents = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        results: List[Tuple[str, int, float]] = []
        for document_text, score in sorted_documents[:top_k]:
            chunk_index = document_to_index[document_text]
            results.append((document_text, self.chunks[chunk_index]['id'], score))
        
        return results
    
    def rerank(
        self, 
        query: str, 
        candidates: List[Tuple[str, int, float]], 
        top_k: int = 15
    ) -> List[Tuple[str, int, float]]:
        """
        Rerank candidates using cross-encoder for better relevance scoring.
        
        Cross-encoders provide more accurate relevance scores by jointly encoding
        query-document pairs, but are slower than bi-encoders.
        
        Args:
            query: Search query string
            candidates: List of (document_text, chunk_id, score) tuples
            top_k: Number of top reranked results to return
            
        Returns:
            List of reranked tuples: (document_text, chunk_id, rerank_score)
            Sorted by rerank score (descending)
        """
        if len(candidates) == 0:
            return []
        
        # Create query-document pairs for cross-encoder
        query_document_pairs = [[query, document_text] for document_text, _, _ in candidates]
        rerank_scores = self.reranker.predict(query_document_pairs)
        
        # Combine candidates with rerank scores
        reranked: List[Tuple[str, int, float]] = []
        for (document_text, chunk_id, _), score in zip(candidates, rerank_scores):
            reranked.append((document_text, chunk_id, float(score)))
        
        # Sort by rerank score and return top_k
        reranked.sort(key=lambda x: x[2], reverse=True)
        return reranked[:top_k]


class QueryRewriter:
    """
    Rewrites queries using OpenAI GPT to incorporate conversation context.
    
    This class makes queries standalone by incorporating necessary context from
    previous conversation turns, improving retrieval quality for follow-up questions.
    
    Attributes:
        model_name: OpenAI model name for query rewriting
    """
    
    def __init__(self, model_name: str) -> None:
        """
        Initialize the query rewriter.
        
        Args:
            model_name: OpenAI model name (e.g., 'gpt-4o')
        """
        self.model_name = model_name
    
    def rewrite_query(
        self, 
        query: str, 
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        Rewrite query to be standalone and context-aware.
        
        If no conversation history exists, returns the original query.
        Otherwise, uses GPT to incorporate context from recent turns.
        
        Args:
            query: Original user query
            conversation_history: List of conversation turns, each with 'user' and 'assistant' keys
            
        Returns:
            Rewritten query string that incorporates conversation context
            
        Note:
            Uses last 2 conversation turns for context to avoid token limits
        """
        if not conversation_history:
            return query
        
        # Format recent conversation history
        history_string = "\n".join([
            f"User: {turn['user']}\nAssistant: {turn['assistant']}"
            for turn in conversation_history[-QUERY_REWRITE_HISTORY_TURNS:]
        ])
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a query rewriter. Rewrite user queries to be standalone "
                    "by incorporating necessary context from conversation history. "
                    "Keep it concise (1-2 sentences)."
                )
            },
            {
                "role": "user",
                "content": f"""Conversation history:
{history_string}

Current query: "{query}"

Rewrite this query to be standalone and include necessary context:"""
            }
        ]
        
        try:
            response = openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=QUERY_REWRITE_TEMPERATURE,
                max_tokens=QUERY_REWRITE_MAX_TOKENS
            )
            
            rewritten_query = response.choices[0].message.content.strip()
            time.sleep(config.REQUEST_DELAY)
            return rewritten_query
            
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}, using original query")
            return query


class FigureManager:
    """
    Manages figure and table retrieval and citation using semantic similarity.
    
    This class indexes figures and tables with their captions and context,
    then retrieves relevant visual elements for a given query using cosine
    similarity between query and figure embeddings.
    
    Attributes:
        visual_elements: Combined list of figures and tables
        embedder: Sentence transformer model for embeddings
        embeddings: Precomputed embeddings for all visual elements
    """
    
    def __init__(
        self, 
        figures: List[Dict[str, Any]], 
        tables: List[Dict[str, Any]], 
        embedder: SentenceTransformer
    ) -> None:
        """
        Initialize the figure manager.
        
        Args:
            figures: List of figure dictionaries
            tables: List of table dictionaries
            embedder: Sentence transformer model for generating embeddings
        """
        self.visual_elements = figures + tables
        self.embedder = embedder
        
        if self.visual_elements:
            # Create embedding text: "Figure X: Caption Context"
            embedding_texts = [
                f"{element['type']} {element['number']}: {element['caption']} {element['context']}"
                for element in self.visual_elements
            ]
            self.embeddings = embedder.encode(embedding_texts, convert_to_numpy=True)
        else:
            self.embeddings = np.array([])
    
    def retrieve_figures(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant figures using semantic similarity.
        
        Uses cosine similarity between query embedding and figure embeddings
        to find the most relevant visual elements.
        
        Args:
            query: Search query string
            top_k: Number of top figures to return
            
        Returns:
            List of figure/table dictionaries with added 'similarity' score
            Sorted by similarity (descending)
        """
        if len(self.visual_elements) == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)[0]
        
        # Calculate cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top_k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Build results with similarity scores
        results: List[Dict[str, Any]] = []
        for index in top_indices:
            element = self.visual_elements[index].copy()
            element['similarity'] = float(similarities[index])
            results.append(element)
        
        return results


class ResponseGenerator:
    """
    Generates responses with explicit figure citations using OpenAI GPT-4o.
    
    This class formats context chunks and figures, then uses GPT-4o to generate
    answers that automatically cite relevant figures/tables when appropriate.
    
    Attributes:
        model_name: OpenAI model name for generation
    """
    
    def __init__(self, model_name: str) -> None:
        """
        Initialize the response generator.
        
        Args:
            model_name: OpenAI model name (e.g., 'gpt-4o')
        """
        self.model_name = model_name
    
    def generate(
        self,
        query: str,
        context_chunks: List[str],
        figures: List[Dict[str, Any]],
        conversation_history: List[Dict[str, str]]
    ) -> Tuple[str, List[str]]:
        """
        Generate response with automatic figure citation extraction.
        
        Args:
            query: User's question
            context_chunks: List of retrieved text chunks
            figures: List of retrieved figure/table dictionaries
            conversation_history: Previous conversation turns
            
        Returns:
            Tuple of:
                - Generated answer string
                - List of cited figure/table numbers (e.g., ['1.4', '2.1'])
        """
        # Format context chunks with numbering
        context_text = "\n\n".join([
            f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks)
        ])
        
        # Format figures/tables
        figure_text = ""
        if figures:
            figure_text = "\n\nRelevant Figures/Tables:\n" + "\n".join([
                f"- {figure['type'].capitalize()} {figure['number']}: {figure['caption']}"
                for figure in figures
            ])
        
        # Build message history
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful medical information assistant. Answer questions "
                    "using the provided context. Be conversational, precise, and cite "
                    "specific figures/tables when relevant.\n\n"
                    "CRITICAL: When referencing figures or tables, use the exact format "
                    '"Figure X" or "Table X" (e.g., "As shown in Figure 2.1" or '
                    '"Table 1.5 indicates").\n\n'
                    "Keep answers concise (2-4 sentences) and factually grounded in the context."
                )
            }
        ]
        
        # Add conversation history (last 2 turns)
        for turn in conversation_history[-QUERY_REWRITE_HISTORY_TURNS:]:
            messages.append({"role": "user", "content": turn['user']})
            messages.append({"role": "assistant", "content": turn['assistant']})
        
        # Add current query with context
        user_message = f"""Context:
{context_text}{figure_text}

Question: {query}"""
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Extract cited figures from answer text
            cited_figures = self._extract_figure_numbers(answer)
            
            # If no figures cited but figures available, add top one
            if not cited_figures and figures:
                top_figure = figures[0]
                cited_figures = [top_figure['number']]
            
            time.sleep(config.REQUEST_DELAY)
            return answer, cited_figures
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return "I apologize, I couldn't generate a proper response.", []
    
    def _extract_figure_numbers(self, text: str) -> List[str]:
        """
        Extract figure/table numbers from text using regex.
        
        Args:
            text: Text to search for figure/table references
            
        Returns:
            List of unique figure/table numbers found (e.g., ['1.4', '2.1'])
        """
        pattern = r'(?:Figure|Table)\s+(\d+\.?\d*)'
        matches = re.findall(pattern, text, re.IGNORECASE)
        return list(set(matches))


class RAGPipeline:
    """
    End-to-end RAG pipeline orchestrating all components.
    
    This class coordinates the complete pipeline:
    1. PDF processing and extraction
    2. Text chunking
    3. Index building
    4. Query processing with retrieval and generation
    5. Batch processing of questions from CSV
    
    Attributes:
        pdf_processor: PDF extraction component
        chunker: Text chunking component
        retriever: Hybrid retrieval component
        query_rewriter: Query rewriting component
        generator: Response generation component
        figure_manager: Figure retrieval component
        conversation_history: Current conversation history
        initialized: Whether pipeline has been initialized
        current_conv_id: Current conversation ID being processed
    """
    
    def __init__(self) -> None:
        """Initialize the RAG pipeline with all components."""
        logger.info("Initializing RAG Pipeline with OpenAI...")
        
        self.pdf_processor = PDFProcessor(config.PDF_PATH)
        self.chunker = SmartChunker(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        self.retriever = HybridRetriever(config.EMBEDDING_MODEL, config.RERANK_MODEL)
        self.query_rewriter = QueryRewriter(config.LLM_MODEL)
        self.generator = ResponseGenerator(config.LLM_MODEL)
        
        self.figure_manager: Optional[FigureManager] = None
        self.conversation_history: List[Dict[str, str]] = []
        self.initialized = False
        self.current_conv_id: Optional[int] = None
        
        logger.info("Pipeline initialized")
    
    def initialize(self) -> None:
        """
        Process document and build all indexes.
        
        This method:
        1. Extracts text, figures, and tables from PDF
        2. Chunks the text
        3. Builds retrieval indexes (BM25 and dense)
        4. Initializes figure manager
        
        Raises:
            RuntimeError: If initialization fails
        """
        logger.info("\n" + "="*80)
        logger.info("INITIALIZING PIPELINE")
        logger.info("="*80)
        
        page_contents = self.pdf_processor.extract_text_with_pages()
        figures, tables = self.pdf_processor.extract_figures_and_tables()
        
        chunks = self.chunker.chunk_text(page_contents)
        
        self.retriever.index_chunks(chunks)
        
        self.figure_manager = FigureManager(
            figures,
            tables,
            self.retriever.embedder
        )
        
        self.initialized = True
        logger.info("\nPipeline ready!")
    
    def process_query(
        self,
        query: str,
        conversation_id: Optional[int] = None
    ) -> Tuple[str, List[str]]:
        """
        Process a single query through the complete pipeline.
        
        Pipeline steps:
        1. Rewrite query with conversation context
        2. Hybrid retrieval (BM25 + dense)
        3. Cross-encoder reranking
        4. Figure retrieval
        5. Response generation with figure citation
        
        Args:
            query: User's question
            conversation_id: Optional conversation ID for history management
            
        Returns:
            Tuple of (answer_text, list_of_cited_figures)
            
        Raises:
            RuntimeError: If pipeline not initialized
        """
        if not self.initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        # Manage conversation history: clear when conversation changes
        if conversation_id is not None:
            if self.current_conv_id is None:
                self.current_conv_id = conversation_id
            elif self.current_conv_id != conversation_id:
                self.conversation_history = []
                self.current_conv_id = conversation_id
        
        # Step 1: Rewrite query with conversation context
        rewritten_query = self.query_rewriter.rewrite_query(
            query,
            self.conversation_history
        )
        logger.debug(f"Rewritten query: {rewritten_query[:100]}...")
        
        # Step 2: Hybrid retrieval
        candidates = self.retriever.retrieve_hybrid(
            rewritten_query,
            top_k=config.INITIAL_RETRIEVAL_K
        )
        
        # Step 3: Rerank candidates
        reranked = self.retriever.rerank(
            rewritten_query,
            candidates,
            top_k=config.RERANK_TOP_K
        )
        
        # Step 4: Select final chunks
        top_chunks = [document_text for document_text, _, _ in reranked[:config.FINAL_TOP_K]]
        
        # Step 5: Retrieve relevant figures
        figures = self.figure_manager.retrieve_figures(
            rewritten_query,
            top_k=config.TOP_K_FIGURES
        )
        
        # Step 6: Generate response
        answer, cited_figures = self.generator.generate(
            query,
            top_chunks,
            figures,
            self.conversation_history
        )
        
        # Update conversation history
        self.conversation_history.append({
            'user': query,
            'assistant': answer
        })
        
        # Keep only recent history (limit memory usage)
        if len(self.conversation_history) > CONVERSATION_HISTORY_LIMIT:
            self.conversation_history = self.conversation_history[-CONVERSATION_HISTORY_LIMIT:]
        
        return answer, cited_figures
    
    def process_csv(
        self, 
        questions_csv: str, 
        output_csv: str
    ) -> pd.DataFrame:
        """
        Process all questions from CSV and generate submission file.
        
        Questions are grouped by conversation_id to maintain conversation context.
        Each question is processed sequentially within its conversation.
        
        Args:
            questions_csv: Path to input CSV file with questions
            output_csv: Path where submission CSV will be saved
            
        Returns:
            DataFrame containing all results
            
        Raises:
            FileNotFoundError: If questions CSV doesn't exist
            Exception: For other processing errors
        """
        logger.info("\n" + "="*80)
        logger.info("PROCESSING QUESTIONS")
        logger.info("="*80)
        
        try:
            df = pd.read_csv(questions_csv)
        except FileNotFoundError:
            logger.error(f"Questions CSV not found: {questions_csv}")
            raise
        except Exception as e:
            logger.error(f"Error reading questions CSV: {e}")
            raise
        
        logger.info(f"Loaded {len(df)} questions")
        
        results: List[Dict[str, Any]] = []
        
        # Process questions grouped by conversation
        for conv_id, group in tqdm(df.groupby('conversation_id'), desc="Conversations"):
            self.conversation_history = []
            
            for _, row in group.iterrows():
                try:
                    answer, figures = self.process_query(row['question'], conv_id)
                    
                    # Format figure references as comma-separated string or '0'
                    figure_references = ','.join(figures) if figures else '0'
                    
                    results.append({
                        'id': row['id'],
                        'conversation_id': conv_id,
                        'question_id': row['question_id'],
                        'answer': answer,
                        'figure_references': figure_references
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing question {row['question_id']}: {e}")
                    results.append({
                        'id': row['id'],
                        'conversation_id': conv_id,
                        'question_id': row['question_id'],
                        'answer': "Error processing question",
                        'figure_references': '0'
                    })
        
        # Create output DataFrame and save
        df_output = pd.DataFrame(results)
        df_output = df_output[['id', 'conversation_id', 'question_id', 'answer', 'figure_references']]
        
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_output.to_csv(output_csv, index=False)
        
        logger.info(f"\nSubmission saved to {output_csv}")
        return df_output


def main() -> Tuple[RAGPipeline, pd.DataFrame]:
    """
    Main entry point for the RAG pipeline.
    
    Initializes and runs the complete pipeline:
    1. Validates configuration
    2. Initializes pipeline
    3. Processes all questions
    4. Generates submission CSV
    
    Returns:
        Tuple of (pipeline_instance, results_dataframe)
        
    Raises:
        ValueError: If OpenAI API key not set
        RuntimeError: If pipeline initialization fails
    """
    logger.info("\n" + "="*70)
    logger.info("MULTIMODAL RAG CHATBOT SYSTEM")
    logger.info("="*70 + "\n")
    
    # Verify API key
    if not config.OPENAI_API_KEY:
        raise ValueError(
            "Please set your OpenAI API key!\n"
            "Get one at: https://platform.openai.com/api-keys\n"
            "Set it as OPENAI_API_KEY environment variable"
        )
    
    # Initialize pipeline
    pipeline = RAGPipeline()
    pipeline.initialize()
    
    # Process questions
    df_submission = pipeline.process_csv(
        config.QUESTIONS_CSV,
        config.OUTPUT_CSV
    )
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Total questions: {len(df_submission)}")
    logger.info(f"\nFirst 5 results:")
    logger.info(f"\n{df_submission.head()}")
    
    return pipeline, df_submission


if __name__ == "__main__":
    pipeline, results = main()
