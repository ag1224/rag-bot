"""Data ingestion module for YouTube transcripts."""

import pandas as pd
from tqdm import tqdm
from llama_index.core.schema import TextNode
import config


def load_youtube_data(file_path: str = config.DATA_PATH) -> list[TextNode]:
    """Load YouTube transcript data from CSV and convert to TextNodes.
    
    Note: YouTube transcripts are already segmented with accurate start/end times.
    We skip chunking to preserve timestamp accuracy for each segment.
    """
    df = pd.read_csv(file_path)
    nodes = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading transcripts"):
        # Get text content
        text = str(row.get("text", ""))
        if not text or text == "nan":
            continue
        
        # Create metadata dict with accurate timestamps
        metadata = {
            "id": str(row.get("id", "")),
            "title": str(row.get("title", "")),
            "url": str(row.get("url", "")),
            "channel_id": str(row.get("channel_id", "")),
            "published": str(row.get("published", "")),
            "start_time": float(row.get("start", 0)),
            "end_time": float(row.get("end", 0)),
        }

        # Create TextNode directly (no splitting needed - preserves timestamps)
        node = TextNode(
            text=text,
            metadata=metadata,
            excluded_llm_metadata_keys=["id", "channel_id"],
            excluded_embed_metadata_keys=["id", "channel_id", "start_time", "end_time"],
        )
        nodes.append(node)

    print(f"Loaded {len(nodes)} transcript segments")
    return nodes


def prepare_data() -> list[TextNode]:
    """Load and prepare data for indexing."""
    nodes = load_youtube_data()
    if nodes:
        print_node_details(nodes[0])
    return nodes


def print_node_details(node) -> None:
    """Print relevant properties of a node."""
    print("--- Node Details ---")
    print(f"Node ID: {node.node_id}")
    print(f"Text (preview): {node.get_content()[:200]}")
    print(f"Metadata: {node.metadata}")
    print(f"Relationships: {node.relationships}")
    print(f"Start char idx: {node.start_char_idx}")
    print(f"End char idx: {node.end_char_idx}")


if __name__ == "__main__":
    nodes = prepare_data()
    print(f"Prepared {len(nodes)} nodes for indexing")
