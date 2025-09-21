# src/retrieval.py
"""
Retrieval-Augmented Generation (RAG) implementation with local LM Studio.

Features:
- Local embedding model via LM Studio (OpenAI-compatible API)
- Local LLM via LM Studio (OpenAI-compatible API)
- Enhanced metadata handling for legal documents
- Comprehensive error handling and logging
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from . import config
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools

# Import caching
from .caching import get_cache, cached_query
from .query_processor import get_query_processor

# Import monitoring
from .monitoring import get_monitor, timing_decorator, query_performance_tracker

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_rag_chain():
    """
    Creates a Retrieval-Augmented Generation (RAG) chain using local LM Studio models.
    """
    print("--- Creating RAG Chain ---")

    print(f"Initializing LLM: {config.LLM_MODEL}")

    # Initialize LLM using local LM Studio (OpenAI-compatible)
    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        openai_api_base="http://127.0.0.1:1234/v1",
        openai_api_key="lm-studio",
        temperature=0.1,
        max_tokens=2048
    )

    print(f"Initializing embedding model: {config.EMBEDDING_MODEL}")

    # Initialize embeddings using local LM Studio (OpenAI-compatible)
    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_base="http://127.0.0.1:1234/v1",
        openai_api_key="lm-studio",
        check_embedding_ctx_length=False
    )

    print(f"Loading vector store from: {config.CHROMA_PATH}")
    vectorstore = Chroma(
        persist_directory=config.CHROMA_PATH,
        embedding_function=embeddings
        # Removed collection_metadata for now to avoid HNSW parameter issues
        # collection_metadata=config.CHROMA_CONFIG
    )

    # Configure retriever with optimized settings (excluding fetch_k for Chroma compatibility)
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": config.PERFORMANCE_CONFIG.get("query_top_k", 10),
            # Note: fetch_k and lambda_mult are not supported by Chroma
            # "fetch_k": config.PERFORMANCE_CONFIG.get("query_fetch_k", 50),
            # "lambda_mult": config.PERFORMANCE_CONFIG.get("query_lambda_mult", 0.5)
        }
    )

    # Initialize query processor for enhanced retrieval
    query_processor = get_query_processor()

    # Enhanced retriever with query processing
    @timing_decorator("enhanced_retriever")
    def enhanced_retriever(query):
        """Enhanced retriever with query expansion and multi-query retrieval"""
        try:
            # Use advanced query processing
            processed_queries = query_processor.process_query(query)

            # Get documents for all processed queries
            all_docs = []
            seen_content = set()

            for processed_query in processed_queries:
                docs = retriever.invoke(processed_query)
                for doc in docs:
                    # Avoid duplicates based on content
                    content_hash = hash(doc.page_content)
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        all_docs.append(doc)

            # Limit to top k documents
            max_docs = config.PERFORMANCE_CONFIG.get("query_top_k", 10)
            all_docs = all_docs[:max_docs]

            # Re-rank documents if we have multiple queries
            if len(processed_queries) > 1:
                all_docs = query_processor.rerank_documents(query, all_docs)

            return all_docs

        except Exception as e:
            logger.warning(f"Query processing failed, falling back to standard retrieval: {e}")
            return retriever.invoke(query)  # Fallback to standard retrieval
    thread_pool = ThreadPoolExecutor(max_workers=config.PERFORMANCE_CONFIG.get("max_workers", 4))

    # Async wrapper for synchronous operations
    def run_in_thread(func, *args, **kwargs):
        """Run a synchronous function in a thread pool"""
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(thread_pool, functools.partial(func, *args, **kwargs))

    # Async enhanced retriever
    @timing_decorator("async_enhanced_retriever")
    async def async_enhanced_retriever(query):
        """Async version of enhanced retriever with concurrent query processing"""
        try:
            # Process query expansion asynchronously
            processed_queries = await run_in_thread(query_processor.process_query, query)

            # Concurrent retrieval for all processed queries
            retrieval_tasks = [
                run_in_thread(retriever.invoke, processed_query)
                for processed_query in processed_queries
            ]

            # Wait for all retrieval tasks to complete
            retrieval_results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)

            # Combine results and handle exceptions
            all_docs = []
            seen_content = set()

            for result in retrieval_results:
                if isinstance(result, Exception):
                    logger.warning(f"Retrieval task failed: {result}")
                    continue

                for doc in result:
                    # Avoid duplicates based on content
                    content_hash = hash(doc.page_content)
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        all_docs.append(doc)

            # Limit to top k documents
            max_docs = config.PERFORMANCE_CONFIG.get("query_top_k", 10)
            all_docs = all_docs[:max_docs]

            # Re-rank documents if we have multiple queries
            if len(processed_queries) > 1:
                all_docs = await run_in_thread(query_processor.rerank_documents, query, all_docs)

            return all_docs

        except Exception as e:
            logger.warning(f"Async query processing failed, falling back to sync: {e}")
            return await run_in_thread(enhanced_retriever, query)

    print("Async enhanced retriever initialized.")

    # Format documents with enhanced metadata for better context
    def format_docs_with_metadata(docs):
        """Format documents with enhanced metadata for better context"""
        if docs is None:
            return "No documents found for this query."

        formatted_docs = []

        for doc in docs:
            metadata = doc.metadata

            # Create rich context with legal metadata
            context_parts = [doc.page_content]

            # Add legal context if available
            if metadata.get('sections'):
                context_parts.append(f"Sections mentioned: {', '.join(metadata['sections'])}")
            if metadata.get('court_mentions'):
                context_parts.append(f"Courts mentioned: {', '.join(metadata['court_mentions'])}")
            if metadata.get('legal_terms'):
                context_parts.append(f"Legal terms: {', '.join(metadata['legal_terms'][:5])}")

            # Add quality information
            if 'quality_score' in metadata:
                quality = metadata['quality_score']
                quality_label = "High" if quality >= 0.8 else "Medium" if quality >= 0.6 else "Low"
                context_parts.append(f"Content Quality: {quality_label} ({quality:.2f})")

            formatted_docs.append("\n".join(context_parts))

        return "\n\n---\n\n".join(formatted_docs)

    # Create prompt template for RAG with enhanced context
    template = """You are a legal research assistant specializing in Indian law. Use the following pieces of context to answer the question at the end.

Each piece of context includes the original content plus relevant legal metadata (sections, courts, legal terms, and quality indicators) to help you provide accurate and well-supported answers.

Context:
{context}

Question: {question}

Instructions:
1. Use the legal metadata to understand the context and relevance of each document section
2. Cite specific sections, articles, or legal references when relevant
3. If the context contains quality indicators, consider the reliability of the information
4. Provide comprehensive but concise answers based on the legal context provided
5. If you don't know the answer based on the provided context, say so clearly

Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    # Create the main RAG chain using LCEL with enhanced retriever
    def create_context(question):
        """Create context using enhanced retriever"""
        return format_docs_with_metadata(enhanced_retriever(question))

    rag_chain = (
        RunnablePassthrough.assign(context=lambda x: create_context(x))
        | prompt
        | llm
        | StrOutputParser()
    )

    # Create a wrapper that returns both answer and context with enhanced metadata
    @query_performance_tracker("rag_sync")
    def rag_chain_with_context(question):
        # Get relevant documents using enhanced retriever
        docs = enhanced_retriever(question)
        context = format_docs_with_metadata(docs)

        # Get the answer
        answer = rag_chain.invoke(question)

        return {
            "answer": answer,
            "context": docs,
            "metadata": {
                "num_docs": len(docs),
                "total_quality_score": sum(doc.metadata.get('quality_score', 0) for doc in docs),
                "avg_quality_score": sum(doc.metadata.get('quality_score', 0) for doc in docs) / max(1, len(docs)),
                "query_processed": True
            }
        }

    # Async RAG chain
    @query_performance_tracker("rag_async")
    async def async_rag_chain(question):
        """Async version of the RAG chain"""
        try:
            # Get documents asynchronously
            docs = await async_enhanced_retriever(question)
            context = await run_in_thread(format_docs_with_metadata, docs)

            # Create the prompt and get answer
            formatted_prompt = prompt.format(context=context, question=question)
            answer = await run_in_thread(llm.invoke, [formatted_prompt])

            return answer.content if hasattr(answer, 'content') else str(answer)

        except Exception as e:
            logger.error(f"Async RAG chain failed: {e}")
            # Fallback to sync version
            return rag_chain.invoke(question)

    # Async wrapper that returns both answer and context
    @query_performance_tracker("rag_async_context")
    async def async_rag_chain_with_context(question):
        """Async version that returns answer, context, and metadata"""
        try:
            # Get relevant documents using async enhanced retriever
            docs = await async_enhanced_retriever(question)
            context = await run_in_thread(format_docs_with_metadata, docs)

            # Get the answer asynchronously
            answer = await async_rag_chain(question)

            return {
                "answer": answer,
                "context": docs,
                "metadata": {
                    "num_docs": len(docs),
                    "total_quality_score": sum(doc.metadata.get('quality_score', 0) for doc in docs),
                    "avg_quality_score": sum(doc.metadata.get('quality_score', 0) for doc in docs) / max(1, len(docs)),
                    "query_processed": True,
                    "async_processed": True
                }
            }
        except Exception as e:
            logger.error(f"Async processing failed, falling back to sync: {e}")
            return rag_chain_with_context(question)

    # Wrap async version with caching
    async def cached_async_rag_chain(question):
        """Cached async RAG chain"""
        cache_key = f"async_rag:{hash(question)}"

        # Try to get from cache first
        cached_result = await run_in_thread(get_cache().get, cache_key)
        if cached_result:
            logger.info("Returning cached async result")
            return cached_result

        # Process the query
        result = await async_rag_chain_with_context(question)

        # Cache the result
        await run_in_thread(get_cache().set, cache_key, result, ttl=3600)  # 1 hour TTL

        return result

    # Wrap with caching
    cached_rag_chain = cached_query(rag_chain_with_context)

    print("--- Async RAG Chain Created Successfully with Caching ---")
    return cached_async_rag_chain, cached_rag_chain