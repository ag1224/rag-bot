"""Main entry point for the YouTube RAG Chatbot."""

import argparse
import sys
import config
from rag_pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(
        description="YouTube RAG Chatbot - Query YouTube transcripts with AI"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the vector index from scratch",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="Single query mode - ask one question and exit",
    )

    args = parser.parse_args()

    if not config.OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not set.")
        print("Create a .env file with your API key:")
        print("  OPENAI_API_KEY=your_key_here")
        sys.exit(1)

    print("Initializing RAG pipeline...")
    pipeline = RAGPipeline(rebuild_index=args.rebuild)

    if args.query:
        result = pipeline.query(args.query)
        print(f"\nAnswer: {result['answer']}")
        print(f"\nSources ({len(result['sources'])}):")
        for i, source in enumerate(result["sources"], 1):
            print(f"  {i}. {source['title']} ({source['url']})")
    else:
        pipeline.chat()


if __name__ == "__main__":
    main()
