# src/config.py
import os

# Get the absolute path of the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Data and Chroma paths
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
CHROMA_PATH = os.path.join(PROJECT_ROOT, "chroma")

# LM Studio models (local)
EMBEDDING_MODEL = "text-embedding-embeddinggemma-300m-qat"
LLM_MODEL = "local-model"  # LM Studio will use whatever model is loaded

# Performance configurations
PERFORMANCE_CONFIG = {
    "batch_size": 50,
    "max_workers": 4,
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "cache_ttl": 3600,  # 1 hour cache
    "max_memory_gb": 8,
    "embedding_batch_size": 100,
    "query_timeout": 30,  # seconds
    "max_concurrent_queries": 5,
    "lazy_loading_threshold": 20,  # Use lazy loading for > 20 documents
    "lazy_loading_size_mb": 50,  # Use lazy loading for > 50MB content
    "lazy_batch_size": 5,  # Process 5 documents at a time in lazy mode
    "query_top_k": 10,  # Number of documents to retrieve
    "query_fetch_k": 50,  # Number of documents to fetch before filtering
    "query_lambda_mult": 0.5  # Lambda multiplier for hybrid search
}

# Chroma HNSW index optimization settings
CHROMA_CONFIG = {
    "hnsw:space": "cosine",  # Better for semantic similarity
    "hnsw:ef_construction": 200,  # Higher quality index
    "hnsw:ef_search": 100,  # Faster search
    "hnsw:max_neighbors": 32,  # Better connectivity
    "hnsw:num_threads": 4,  # Parallel index building
    "hnsw:batch_size": 100,  # Batch processing size
    "hnsw:sync_threshold": 1000  # Sync frequency
}

# Caching configurations
CACHE_CONFIG = {
    "redis_url": "redis://localhost:6379",
    "ttl_seconds": 3600,  # 1 hour cache
    "max_cache_size": 1000,
    "cache_dir": os.path.join(PROJECT_ROOT, "cache")
}

# Model optimizations
MODEL_CONFIG = {
    "temperature": 0.1,
    "max_tokens": 2048,
    "top_p": 0.9,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
}

# Graph cache path
GRAPH_CACHE_PATH = os.path.join(CACHE_CONFIG["cache_dir"], "knowledge_graph.pkl")

# Create cache directory if it doesn't exist
os.makedirs(CACHE_CONFIG["cache_dir"], exist_ok=True)