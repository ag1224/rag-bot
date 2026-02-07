"""Configuration for the RAG pipeline."""

import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

# ChromaDB Configuration (open-source, self-hosted)
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "youtube_transcripts"

# Data Configuration
DATA_PATH = "./data/youtube_rag_data.csv"

# Retrieval Configuration
TOP_K = 5
HYBRID_ALPHA = 0.5  # 0 = pure BM25, 1 = pure dense

# Chunking Configuration
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
