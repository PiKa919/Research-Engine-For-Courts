# src/ingest.py

import os
import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src import config
from src.enhanced_document_processor import create_enhanced_ingestion_pipeline_lazy
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_batch_parallel(batch: List[Document], batch_num: int, total_batches: int, vectorstore: Chroma) -> Dict[str, Any]:
    """Process a single batch of documents with error handling and timing"""
    start_time = time.time()

    try:
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

        # Add documents to vector store
        vectorstore.add_documents(batch)

        processing_time = time.time() - start_time
        logger.info(f"✓ Batch {batch_num} completed successfully in {processing_time:.2f}s")

        return {
            'batch_num': batch_num,
            'status': 'success',
            'chunks_processed': len(batch),
            'processing_time': processing_time
        }

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"✗ Error processing batch {batch_num}: {e}")

        return {
            'batch_num': batch_num,
            'status': 'error',
            'error': str(e),
            'chunks_processed': 0,
            'processing_time': processing_time
        }

def ingest_data_parallel():
    """
    Enhanced parallel data ingestion with optimized batch processing
    """
    print("=== Enhanced Parallel Data Ingestion Pipeline ===")

    if not os.path.exists(config.DATA_PATH):
        os.makedirs(config.DATA_PATH)
        print(f"Created directory: {config.DATA_PATH}. Awaiting PDF files...")
        return

    print(f"Loading documents from: {config.DATA_PATH}")

    # Initialize embeddings with optimized settings
    print(f"Initializing embedding model: {config.EMBEDDING_MODEL}")
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_base="http://127.0.0.1:1234/v1",
        openai_api_key="lm-studio",
        check_embedding_ctx_length=False,
        **config.EMBEDDING_CONFIG
    )

    # Check document count and size to decide on processing strategy
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    temp_loader = PyPDFDirectoryLoader(config.DATA_PATH)
    temp_docs = temp_loader.load()
    doc_count = len(temp_docs)

    # Calculate total content size
    total_content_size = sum(len(doc.page_content) for doc in temp_docs)

    # Use lazy loading for large collections
    use_lazy_loading = (
        doc_count > config.PERFORMANCE_CONFIG.get("lazy_loading_threshold", 20) or
        total_content_size > config.PERFORMANCE_CONFIG.get("lazy_loading_size_mb", 50) * 1024 * 1024
    )

    if use_lazy_loading:
        logger.info(f"Using lazy loading for {doc_count} documents ({total_content_size / (1024*1024):.1f} MB)")
        processed_chunks, stats = create_enhanced_ingestion_pipeline_lazy(
            config.DATA_PATH,
            config,
            embeddings,
            batch_size=config.PERFORMANCE_CONFIG.get("lazy_batch_size", 5)
        )
    else:
        logger.info(f"Using standard processing for {doc_count} documents")
        processed_chunks, stats = create_enhanced_ingestion_pipeline_lazy(config.DATA_PATH, config, embeddings)

    if not processed_chunks:
        print("No chunks to process. Exiting.")
        return

    print(f"Creating optimized vector store at: {config.CHROMA_PATH}")

    # Initialize Chroma with optimized configuration
    vectorstore = Chroma(
        persist_directory=config.CHROMA_PATH,
        embedding_function=embeddings
        # Removed empty collection_metadata to use Chroma defaults
    )

    # Use optimized batch size from config
    batch_size = config.PERFORMANCE_CONFIG["batch_size"]
    max_workers = config.PERFORMANCE_CONFIG["max_workers"]

    total_batches = (len(processed_chunks) + batch_size - 1) // batch_size
    print(f"Adding {len(processed_chunks)} chunks to vector store in {total_batches} batches...")
    print(f"Using {max_workers} parallel workers for processing")

    # Process batches in parallel
    successful_batches = 0
    failed_batches = 0
    total_processing_time = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batches for parallel processing
        future_to_batch = {}
        for i in range(0, len(processed_chunks), batch_size):
            batch = processed_chunks[i:i + batch_size]
            batch_num = i // batch_size + 1
            future = executor.submit(process_batch_parallel, batch, batch_num, total_batches, vectorstore)
            future_to_batch[future] = batch_num

        # Process completed batches with progress tracking
        with tqdm(total=total_batches, desc="Processing batches") as pbar:
            for future in as_completed(future_to_batch):
                batch_num = future_to_batch[future]
                try:
                    result = future.result()
                    if result['status'] == 'success':
                        successful_batches += 1
                    else:
                        failed_batches += 1

                    total_processing_time += result['processing_time']
                    pbar.update(1)

                except Exception as e:
                    logger.error(f"Batch {batch_num} generated an exception: {e}")
                    failed_batches += 1
                    pbar.update(1)

    # Print final statistics
    avg_batch_time = total_processing_time / max(1, successful_batches)
    print("\n=== Ingestion Summary ===")
    print(f"Total chunks processed: {len(processed_chunks)}")
    print(f"Successful batches: {successful_batches}")
    print(f"Failed batches: {failed_batches}")
    print(f"Average batch processing time: {avg_batch_time:.2f}s")
    print(f"Total processing time: {total_processing_time:.2f}s")
    print(f"Chunks per second: {len(processed_chunks) / max(1, total_processing_time):.2f}")

    if stats:
        print("\nDocument Analysis:")
        print(f"- Total documents: {stats.get('analysis', {}).get('total_documents', 0)}")
        print(f"- Average document length: {stats.get('analysis', {}).get('avg_doc_length', 0):.0f} characters")
        print(f"- Legal documents detected: {stats.get('analysis', {}).get('is_legal_document', False)}")
        print(f"- Final chunk count: {stats.get('final_chunk_count', 0)}")

    print("\n=== Enhanced Parallel Ingestion Pipeline Completed ===")

def ingest_data():
    """Legacy function for backward compatibility"""
    return ingest_data_parallel()

if __name__ == "__main__":
    ingest_data_parallel()