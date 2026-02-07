"""RAG Pipeline with hybrid search using LlamaIndex and ChromaDB."""

import os
from typing import Optional

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.prompts import PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.retrievers.bm25 import BM25Retriever
import chromadb

import config
from ingest import prepare_data


class HybridRetriever(BaseRetriever):
    """Custom hybrid retriever combining dense (vector) and sparse (BM25) search."""

    def __init__(
        self,
        vector_retriever: BaseRetriever,
        nodes: list,
        alpha: float = config.HYBRID_ALPHA,
        top_k: int = config.TOP_K,
    ):
        super().__init__()
        self.vector_retriever = vector_retriever
        self.alpha = alpha
        self.top_k = top_k

        # Build BM25 retriever
        self.bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=top_k * 2,
        )
        self.node_map = {node.node_id: node for node in nodes}

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Retrieve using hybrid search (dense + sparse)."""
        query = query_bundle.query_str

        # Dense retrieval (vector search)
        dense_results = self.vector_retriever.retrieve(query)
        dense_scores = {r.node.node_id: r.score for r in dense_results}
        print(
            f"Dense scores length: {len(dense_scores)}, 1st dense score: {list(dense_scores.items())[0]}"
        )

        # Sparse retrieval (BM25)
        bm25_results = self.bm25_retriever.retrieve(query)
        sparse_scores = {r.node.node_id: r.score for r in bm25_results}
        print(
            f"Sparse scores length: {len(sparse_scores)}, 1st sparse score: {list(sparse_scores.items())[0]}"
        )

        # Combine scores: alpha * dense + (1 - alpha) * sparse
        all_node_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
        combined_scores = {}
        print(f"Length of all node ids: {len(all_node_ids)}")

        for node_id in all_node_ids:
            dense_score = dense_scores.get(node_id, 0)
            sparse_score = sparse_scores.get(node_id, 0)
            combined_scores[node_id] = (
                self.alpha * dense_score + (1 - self.alpha) * sparse_score
            )

        # Sort and get top-k
        sorted_nodes = sorted(
            combined_scores.items(), key=lambda x: x[1], reverse=True
        )[: self.top_k]

        # Build results
        results = []
        for node_id, score in sorted_nodes:
            if node_id in self.node_map:
                node = self.node_map[node_id]
                results.append(NodeWithScore(node=node, score=score))
        print(f"Results length: {len(results)}, 1st result: {results[0]}")
        return results


class RAGPipeline:
    """Main RAG pipeline with hybrid search using ChromaDB."""

    def __init__(self, rebuild_index: bool = False):
        self._setup_llm()
        self.index: Optional[VectorStoreIndex] = None
        self.hybrid_retriever: Optional[HybridRetriever] = None
        self.nodes: list = []

        if rebuild_index or not self._index_exists():
            self._build_index()
        else:
            self._load_index()

    def _setup_llm(self):
        """Configure LLM and embedding models."""
        Settings.embed_model = OpenAIEmbedding(
            model=config.EMBEDDING_MODEL,
            api_key=config.OPENAI_API_KEY,
        )
        Settings.llm = OpenAI(
            model=config.LLM_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=0.1,
        )

    def _index_exists(self) -> bool:
        """Check if index already exists."""
        return os.path.exists(config.CHROMA_PATH)

    def _build_index(self):
        """Build vector index from documents."""
        print("Building index...")

        # Prepare data
        self.nodes = prepare_data()

        # Initialize ChromaDB client with persistent storage
        chroma_client = chromadb.PersistentClient(path=config.CHROMA_PATH)

        # Delete existing collection if rebuilding
        try:
            chroma_client.delete_collection(config.COLLECTION_NAME)
        except chromadb.errors.NotFoundError:
            pass

        # Create collection
        chroma_collection = chroma_client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        # Create vector store
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        # Build index
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.index = VectorStoreIndex(
            nodes=self.nodes,
            storage_context=storage_context,
            show_progress=True,
        )

        # Setup hybrid retriever
        self._setup_hybrid_retriever()
        print("Index built successfully!")

    def _load_index(self):
        """Load existing index."""
        print("Loading existing index...")

        chroma_client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        chroma_collection = chroma_client.get_collection(config.COLLECTION_NAME)

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        self.index = VectorStoreIndex.from_vector_store(vector_store)

        # Reload nodes for BM25
        self.nodes = prepare_data()
        self._setup_hybrid_retriever()
        print("Index loaded successfully!")

    def _setup_hybrid_retriever(self):
        """Setup hybrid retriever with dense and sparse search."""
        vector_retriever = self.index.as_retriever(similarity_top_k=config.TOP_K * 2)
        self.hybrid_retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            nodes=self.nodes,
            alpha=config.HYBRID_ALPHA,
            top_k=config.TOP_K,
        )

    def _get_qa_prompt(self) -> PromptTemplate:
        """Get the QA prompt template with context."""
        template = """You are a helpful AI assistant specializing in explaining technical concepts from YouTube video transcripts. 

                Use the following context from video transcripts to answer the question. Include relevant video titles and timestamps when helpful.

                If the answer is not in the context, say so clearly. Do not make up information.

                Context:
                {context}

                Question: {query}

                Answer: """
        return PromptTemplate(template)

    def _format_context(self, nodes: list[NodeWithScore]) -> str:
        """Format retrieved nodes into context string."""
        context_parts = []
        for i, node in enumerate(nodes, 1):
            metadata = node.node.metadata
            title = metadata.get("title", "Unknown")
            url = metadata.get("url", "")
            start = metadata.get("start_time", 0)

            minutes = int(start // 60)
            seconds = int(start % 60)
            timestamp = f"{minutes}:{seconds:02d}"

            context_parts.append(
                f'[Source {i}] Video: "{title}" (at {timestamp})\n'
                f"URL: {url}&t={int(start)}s\n"
                f"Content: {node.node.get_content()}\n"
            )

        return "\n---\n".join(context_parts)

    def query(self, question: str) -> dict:
        """Query the RAG pipeline."""
        retrieved_nodes = self.hybrid_retriever.retrieve(question)
        context = self._format_context(retrieved_nodes)

        prompt = self._get_qa_prompt()
        formatted_prompt = prompt.format(context=context, query=question)
        response = Settings.llm.complete(formatted_prompt)

        return {
            "answer": str(response),
            "sources": [
                {
                    "title": n.node.metadata.get("title", ""),
                    "url": n.node.metadata.get("url", ""),
                    "start_time": n.node.metadata.get("start_time", 0),
                    "score": n.score,
                    "text_preview": n.node.get_content()[:200] + "...",
                }
                for n in retrieved_nodes
            ],
        }

    def chat(self):
        """Interactive chat mode."""
        print("\n" + "=" * 60)
        print("YouTube RAG Chatbot - Ask questions about the transcripts!")
        print("Type 'quit' or 'exit' to end the conversation")
        print("Type 'sources' to see detailed source info for last response")
        print("=" * 60 + "\n")

        last_result = None

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["quit", "exit"]:
                    print("Goodbye!")
                    break

                if user_input.lower() == "sources" and last_result:
                    print("\n📚 Sources from last response:")
                    for i, source in enumerate(last_result["sources"], 1):
                        print(f"\n{i}. {source['title']}")
                        print(f"   URL: {source['url']}&t={int(source['start_time'])}s")
                        print(f"   Relevance: {source['score']:.3f}")
                    continue

                print("\n🤔 Thinking...")
                result = self.query(user_input)
                last_result = result

                print(f"\n🤖 Assistant: {result['answer']}")
                print(f"\n📎 ({len(result['sources'])} sources used)")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break


if __name__ == "__main__":
    pipeline = RAGPipeline(rebuild_index=False)
    pipeline.chat()
