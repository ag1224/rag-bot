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
TOP_K = 10
HYBRID_ALPHA = 0.5  # 0 = pure BM25, 1 = pure dense

# Guardrails for domain/evidence checks
ENABLE_DOMAIN_GUARD = True
ENABLE_EVIDENCE_GUARD = True

# If the top fused score is below this, treat retrieval as weak evidence.
MIN_TOP_SCORE = 0.25

# If average score across returned nodes is below this, treat retrieval as weak.
MIN_AVG_SCORE = 0.15

# Cache Configuration
CACHE_TTL_SECONDS = 300
