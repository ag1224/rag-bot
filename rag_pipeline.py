"""RAG Pipeline with hybrid search using LlamaIndex and ChromaDB."""

import os
from typing import Optional

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
from llama_index.core.schema import NodeWithScore
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
        # self.node_map = {node.node_id: node for node in nodes}
        self.fusion_retriever = QueryFusionRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            similarity_top_k=top_k,
            retriever_weights=[
                self.alpha,
                1 - self.alpha,
            ],  # weight for vector and bm25 retrievers
            num_queries=1,  # set this to 1 to disable query generation
            mode="relative_score",  # use relative score to combine scores
            use_async=True,
            verbose=True,
        )

    def _retrieve(self, query: str) -> list[NodeWithScore]:
        """Retrieve using hybrid search (dense + sparse)."""
        return self.fusion_retriever.retrieve(query)


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

                Use the following context from video transcripts to answer the question.
                Respond in Markdown.
                Include source citations in the answer body like [Source 1], [Source 2] when relevant.

                If the answer is not in the context, say so clearly. Do not make up information.

                Context:
                {context}

                Question: {query}

                Answer: """
        return PromptTemplate(template)

    def _is_query_in_domain(self, question: str) -> bool:
        """Check whether the query is in scope for this transcript corpus."""
        if not config.ENABLE_DOMAIN_GUARD:
            return True

        guard_prompt = f"""You are a strict classifier.
                        Task: Decide if the user question is in-domain for a YouTube technical transcript QA system.
                        In-domain includes: AI, ML, LLMs, transformers, NLP, Python coding, model training, embeddings, vector databases, RAG, APIs, and software engineering topics likely discussed in technical videos.
                        Out-of-domain includes: medicine, law, finance advice, personal life, general trivia unrelated to technical video transcript content.

                        Return ONLY one token: YES or NO.

                        Question: {question}
                        Label:"""
        label = str(Settings.llm.complete(guard_prompt)).strip().upper()
        return label.startswith("YES")

    # def _has_sufficient_evidence(self, nodes: list[NodeWithScore]) -> bool:
    #     """Check if retrieved evidence is strong enough to answer confidently."""
    #     if not config.ENABLE_EVIDENCE_GUARD:
    #         return True
    #     if not nodes:
    #         return False

    #     scores = [float(n.score) for n in nodes if n.score is not None]
    #     if not scores:
    #         return False

    #     top_score = max(scores)
    #     avg_score = sum(scores) / len(scores)
    #     return top_score >= config.MIN_TOP_SCORE and avg_score >= config.MIN_AVG_SCORE

    def _guardrail_response(self, message: str) -> dict:
        """Consistent markdown response when guardrails block answering."""
        return {
            "answer": f"{message}\n\n## Sources\n- No sources used.",
            "sources": [],
        }

    def _build_context_and_sources(self, nodes: list[NodeWithScore]) -> tuple[str, str]:
        """Build LLM context and markdown source links in one pass."""
        if not nodes:
            return "", ""

        context_parts = []
        source_lines = ["## Sources"]

        for i, node in enumerate(nodes, 1):
            metadata = node.node.metadata
            title = metadata.get("title", "Unknown")
            base_url = metadata.get("url", "")
            start = int(metadata.get("start_time", 0))
            minutes = start // 60
            seconds = start % 60
            timestamp = f"{minutes}:{seconds:02d}"
            source_url = f"{base_url}&t={start}s" if base_url else ""

            context_parts.append(
                f'[Source {i}] Video: "{title}" (at {timestamp})\n'
                f"URL: {source_url}\n"
                f"Content: {node.node.get_content()}\n"
            )

            if source_url:
                source_lines.append(
                    f"- **[Source {i}]** [{title} ({timestamp})]({source_url})"
                )
            else:
                source_lines.append(f"- **[Source {i}]** {title} ({timestamp})")

        return "\n---\n".join(context_parts), "\n".join(source_lines)

    def query(self, question: str) -> dict:
        """Query the RAG pipeline."""
        if not self._is_query_in_domain(question):
            return self._guardrail_response(
                "I can only answer questions related to the technical YouTube transcripts in the dataset. "
            )

        retrieved_nodes = self.hybrid_retriever.retrieve(question)
        context, sources_markdown = self._build_context_and_sources(retrieved_nodes)

        prompt = self._get_qa_prompt()
        formatted_prompt = prompt.format(context=context, query=question)
        response = Settings.llm.complete(formatted_prompt)
        answer_markdown = str(response).strip()
        if sources_markdown:
            answer_markdown = f"{answer_markdown}\n\n{sources_markdown}"

        return {
            "answer": answer_markdown,
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
