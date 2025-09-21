# src/query_processor.py

import logging
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from . import config
from .caching import get_cache, cached_query

logger = logging.getLogger(__name__)

class AdvancedQueryProcessor:
    """Advanced query processing with expansion and multi-query techniques"""

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(
            model=config.LLM_MODEL,
            openai_api_base="http://127.0.0.1:1234/v1",
            openai_api_key="lm-studio",
            **config.MODEL_CONFIG
        )
        self.cache = get_cache()

    @cached_query
    def expand_query(self, query: str, num_expansions: int = 3) -> List[str]:
        """
        Expand a query with synonyms and related legal terms

        Args:
            query: Original query string
            num_expansions: Number of expanded queries to generate

        Returns:
            List of expanded query strings
        """
        expansion_prompt = f"""
        You are a legal research assistant. Given the query: "{query}"

        Generate {num_expansions} expanded versions of this query that include:
        1. Legal synonyms and related terms
        2. Broader legal concepts
        3. Specific legal procedures or doctrines
        4. Related case law or statutory references

        Each expanded query should be more comprehensive than the original but still focused on the core legal issue.

        Return only the expanded queries, one per line, without numbering or additional text.
        """

        try:
            response = self.llm.invoke(expansion_prompt)
            expansions = [line.strip() for line in response.content.split('\n') if line.strip()]
            expansions = expansions[:num_expansions]  # Limit to requested number

            # Add original query at the beginning
            return [query] + expansions

        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return [query]

    def generate_multi_queries(self, query: str, num_variations: int = 5) -> List[str]:
        """
        Generate multiple query variations for better retrieval

        Args:
            query: Original query
            num_variations: Number of variations to generate

        Returns:
            List of query variations
        """
        variation_prompt = f"""
        You are a legal research assistant. Given the query: "{query}"

        Generate {num_variations} different phrasings of this query that would help retrieve relevant legal information.
        Each variation should:
        1. Use different legal terminology
        2. Focus on different aspects of the legal issue
        3. Include relevant legal procedures or remedies
        4. Consider different jurisdictional approaches

        Return only the query variations, one per line, without numbering or additional text.
        """

        try:
            response = self.llm.invoke(variation_prompt)
            variations = [line.strip() for line in response.content.split('\n') if line.strip()]
            variations = variations[:num_variations]

            return variations

        except Exception as e:
            logger.warning(f"Multi-query generation failed: {e}")
            return [query]

    def rerank_documents(self, query: str, documents: List[Document], top_k: int = 10) -> List[Document]:
        """
        Re-rank documents based on relevance to the query

        Args:
            query: Original query
            documents: List of documents to rerank
            top_k: Number of top documents to return

        Returns:
            Re-ranked list of documents
        """
        if len(documents) <= top_k:
            return documents

        # Simple relevance scoring based on keyword matching
        query_terms = set(query.lower().split())
        legal_terms = {'court', 'case', 'section', 'article', 'act', 'law', 'judgment', 'appeal', 'petition'}

        scored_docs = []
        for doc in documents:
            content = doc.page_content.lower()
            score = 0

            # Exact query term matches
            for term in query_terms:
                if term in content:
                    score += 2

            # Legal term matches
            for term in legal_terms:
                if term in content:
                    score += 1

            # Length penalty (prefer more substantial content)
            word_count = len(content.split())
            if 100 <= word_count <= 1000:
                score += 1
            elif word_count < 50:
                score -= 1

            scored_docs.append((score, doc))

        # Sort by score (descending) and return top_k
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]

    def hybrid_retrieval(self, retriever_func, query: str, **kwargs) -> Dict[str, Any]:
        """
        Perform hybrid retrieval with query expansion and re-ranking

        Args:
            retriever_func: Function that performs the actual retrieval
            query: Original query
            **kwargs: Additional arguments for retriever

        Returns:
            Dictionary with retrieval results and metadata
        """
        logger.info(f"Performing hybrid retrieval for query: {query}")

        # Generate expanded queries
        expanded_queries = self.expand_query(query, num_expansions=2)

        all_docs = []
        query_results = {}

        # Retrieve documents for each query variation
        for i, expanded_query in enumerate(expanded_queries):
            try:
                logger.debug(f"Retrieving for variation {i+1}: {expanded_query}")
                results = retriever_func(expanded_query, **kwargs)

                if isinstance(results, dict) and 'context' in results:
                    docs = results['context']
                elif isinstance(results, list):
                    docs = results
                else:
                    docs = []

                all_docs.extend(docs)
                query_results[f"query_{i+1}"] = {
                    'query': expanded_query,
                    'num_docs': len(docs)
                }

            except Exception as e:
                logger.warning(f"Retrieval failed for query variation {i+1}: {e}")
                query_results[f"query_{i+1}"] = {
                    'query': expanded_query,
                    'error': str(e)
                }

        # Remove duplicates while preserving order
        seen_ids = set()
        unique_docs = []
        for doc in all_docs:
            doc_id = getattr(doc, 'id', None) or getattr(doc, 'metadata', {}).get('source', str(id(doc)))
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append(doc)

        # Re-rank documents
        reranked_docs = self.rerank_documents(query, unique_docs, top_k=kwargs.get('top_k', 10))

        result = {
            'answer': None,  # Will be filled by RAG chain
            'context': reranked_docs,
            'metadata': {
                'original_query': query,
                'num_expanded_queries': len(expanded_queries),
                'total_docs_retrieved': len(all_docs),
                'unique_docs_after_deduplication': len(unique_docs),
                'final_docs_after_reranking': len(reranked_docs),
                'query_results': query_results
            }
        }

        logger.info(f"Hybrid retrieval completed: {len(reranked_docs)} final documents")
        return result

    def process_query(self, query: str) -> List[str]:
        """
        Process a query and return multiple variations for retrieval

        Args:
            query: Original query string

        Returns:
            List of query variations including original and expanded versions
        """
        try:
            # Start with the original query
            queries = [query]

            # Add expanded queries
            expanded = self.expand_query(query, num_expansions=2)
            queries.extend(expanded)

            # Add multi-query variations
            multi_queries = self.generate_multi_queries(query, num_variations=3)
            queries.extend(multi_queries)

            # Remove duplicates while preserving order
            seen = set()
            unique_queries = []
            for q in queries:
                if q not in seen:
                    seen.add(q)
                    unique_queries.append(q)

            return unique_queries[:5]  # Limit to 5 queries for performance

        except Exception as e:
            logger.warning(f"Query processing failed: {e}")
            return [query]  # Return original query as fallback

# Global instance
_query_processor = None

def get_query_processor(llm: Optional[ChatOpenAI] = None) -> AdvancedQueryProcessor:
    """Get global query processor instance"""
    global _query_processor
    if _query_processor is None:
        _query_processor = AdvancedQueryProcessor(llm)
    return _query_processor