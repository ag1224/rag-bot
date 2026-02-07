"""Data ingestion module for YouTube transcripts."""

import pandas as pd
from tqdm import tqdm
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
import config


def load_youtube_data(file_path: str = config.DATA_PATH) -> list[Document]:
    """Load YouTube transcript data from CSV and convert to LlamaIndex Documents."""
    df = pd.read_csv(file_path)
    documents = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading documents"):
        # Create metadata dict
        metadata = {
            "id": str(row.get("id", "")),
            "title": str(row.get("title", "")),
            "url": str(row.get("url", "")),
            "channel_id": str(row.get("channel_id", "")),
            "published": str(row.get("published", "")),
            "start_time": float(row.get("start", 0)),
            "end_time": float(row.get("end", 0)),
        }

        # Get text content
        text = str(row.get("text", ""))
        if not text or text == "nan":
            continue

        # Create document with metadata
        doc = Document(
            text=text,
            metadata=metadata,
            excluded_llm_metadata_keys=["id", "channel_id"],
            excluded_embed_metadata_keys=["id", "channel_id", "start_time", "end_time"],
        )
        documents.append(doc)

    print(f"Loaded {len(documents)} documents")
    return documents


def chunk_documents(documents: list[Document]) -> list:
    """Split documents into smaller chunks for better retrieval."""
    splitter = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    nodes = splitter.get_nodes_from_documents(documents, show_progress=True)
    print(f"Created {len(nodes)} chunks from {len(documents)} documents")
    return nodes


def prepare_data() -> list:
    """Load and prepare data for indexing."""
    documents = load_youtube_data()
    nodes = chunk_documents(documents)
    print(f"First node: {nodes[0]}")
    return nodes


if __name__ == "__main__":
    nodes = prepare_data()
    print(f"Prepared {len(nodes)} nodes for indexing")
