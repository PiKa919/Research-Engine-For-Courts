# src/enhanced_document_processor.py

"""
Enhanced Document Processing for Legal RAG System
Following LangChain best practices for document processing and text splitting
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

try:
    from langchain_experimental.text_splitter import SemanticChunker
    SEMANTIC_CHUNKER_AVAILABLE = True
except ImportError:
    SEMANTIC_CHUNKER_AVAILABLE = False
    print("Warning: langchain_experimental not available. Using standard chunking only.")

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document as LCDocument

# from .retrieval import RateLimitedEmbeddings  # Removed to avoid circular import
from . import config

logger = logging.getLogger(__name__)

class DocumentStructureAnalyzer:
    """Analyzes document structure to determine optimal processing strategy"""

    def __init__(self):
        self.legal_patterns = {
            'section_pattern': re.compile(r'^(?:Section|Article|Chapter|Clause)\s+(\d+|[IVXLCDM]+)', re.IGNORECASE | re.MULTILINE),
            'court_pattern': re.compile(r'\b(?:Supreme Court|High Court|Constitutional Court|District Court)\b', re.IGNORECASE),
            'legal_terms': re.compile(r'\b(?:petitioner|respondent|appellant|defendant|plaintiff|judgment|verdict|precedent)\b', re.IGNORECASE),
            'citation_pattern': re.compile(r'\[\d+\]|\(\d{4}\)', re.IGNORECASE),
            'header_pattern': re.compile(r'^#{1,6}\s|^[A-Z][^.!?]*:$|^SECTION|^CHAPTER|^ARTICLE', re.MULTILINE)
        }

    def analyze_documents(self, documents: List[LCDocument]) -> Dict[str, Any]:
        """Analyze document structure and content characteristics"""
        analysis = {
            'total_documents': len(documents),
            'avg_doc_length': 0,
            'is_legal_document': False,
            'has_structure': False,
            'has_citations': False,
            'document_types': set(),
            'language': 'english',  # Default assumption
            'complexity_score': 0
        }

        total_length = 0
        legal_indicators = 0
        structural_indicators = 0

        for doc in documents:
            content = doc.page_content
            content_length = len(content)
            total_length += content_length

            # Check for legal document indicators
            for pattern_name, pattern in self.legal_patterns.items():
                if pattern.search(content):
                    legal_indicators += 1
                    if pattern_name == 'court_pattern':
                        analysis['is_legal_document'] = True
                    elif pattern_name == 'citation_pattern':
                        analysis['has_citations'] = True
                    elif pattern_name == 'header_pattern':
                        analysis['has_structure'] = True

            # Check for structural elements
            if re.search(r'\n\s*\d+\.\s|\n\s*[a-z]\)\s|\n\s*\([a-z]\)\s', content):
                structural_indicators += 1

            # Determine document type from filename
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                source = doc.metadata['source']
                if 'constitution' in source.lower():
                    analysis['document_types'].add('constitution')
                elif 'code' in source.lower():
                    analysis['document_types'].add('code')
                elif 'act' in source.lower():
                    analysis['document_types'].add('act')
                elif 'rules' in source.lower():
                    analysis['document_types'].add('rules')

        # Calculate averages and scores
        analysis['avg_doc_length'] = total_length / max(1, len(documents))
        analysis['complexity_score'] = min(1.0, (legal_indicators + structural_indicators) / (len(documents) * 2))

        # Final legal document determination
        if legal_indicators >= len(documents) * 0.3:  # 30% of documents have legal indicators
            analysis['is_legal_document'] = True

        logger.info(f"Document analysis completed: {analysis}")
        return analysis

class AdaptiveTextSplitter:
    """Adaptive text splitter that adjusts strategy based on document analysis"""

    def __init__(self, analysis: Dict[str, Any]):
        self.analysis = analysis
        self._configure_splitter()

    def _configure_splitter(self):
        """Configure the text splitter based on document analysis"""

        if self.analysis['is_legal_document']:
            # Legal documents need careful section preservation
            separators = [
                "\nSection ", "\nArticle ", "\nChapter ", "\nClause ",
                "\n\n", "\n", " ", ""
            ]
            chunk_size = 1500  # Larger chunks for legal context
            chunk_overlap = 300  # More overlap for legal continuity

        elif self.analysis['has_structure']:
            # Structured documents with headers
            separators = [
                "\n# ", "\n## ", "\n### ", "\n#### ",
                "\n\n", "\n", " ", ""
            ]
            chunk_size = 1200
            chunk_overlap = 250

        elif self.analysis['complexity_score'] > 0.5:
            # Complex documents
            separators = ["\n\n", "\n", ". ", " ", ""]
            chunk_size = 1000
            chunk_overlap = 200

        else:
            # Simple documents
            separators = ["\n\n", "\n", " ", ""]
            chunk_size = 800
            chunk_overlap = 150

        # Adjust for document length
        avg_length = self.analysis['avg_doc_length']
        if avg_length > 50000:  # Very long documents
            chunk_size = int(chunk_size * 1.2)
            chunk_overlap = int(chunk_overlap * 1.3)
        elif avg_length < 5000:  # Short documents
            chunk_size = int(chunk_size * 0.8)
            chunk_overlap = int(chunk_overlap * 0.7)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
            keep_separator=True,
            add_start_index=True
        )

        logger.info(f"Configured adaptive splitter: chunk_size={chunk_size}, overlap={chunk_overlap}")

    def split_documents(self, documents: List[LCDocument]) -> List[LCDocument]:
        """Split documents using the adaptive strategy"""
        return self.splitter.split_documents(documents)

class SemanticTextSplitter:
    """Advanced semantic text splitter for legal documents"""

    def __init__(self, embeddings, analysis: Dict[str, Any]):
        self.embeddings = embeddings
        self.analysis = analysis
        self._configure_semantic_splitter()

    def _configure_semantic_splitter(self):
        """Configure semantic chunker based on document analysis"""

        # Adjust breakpoint threshold based on document complexity
        if self.analysis['complexity_score'] > 0.7:
            # High complexity - more conservative splitting
            breakpoint_threshold = 95
        elif self.analysis['complexity_score'] > 0.4:
            # Medium complexity
            breakpoint_threshold = 90
        else:
            # Low complexity - more aggressive splitting
            breakpoint_threshold = 85

        # Use percentile-based threshold for legal documents
        if SEMANTIC_CHUNKER_AVAILABLE:
            try:
                self.semantic_splitter = SemanticChunker(
                    embeddings=self.embeddings,
                    breakpoint_threshold_type="percentile",
                    breakpoint_threshold_amount=breakpoint_threshold,
                    buffer_size=1,  # Minimal buffer for legal precision
                    min_chunk_size=200  # Minimum chunk size for legal context
                )
                logger.info(f"Configured semantic splitter: threshold={breakpoint_threshold}%")
            except Exception as e:
                logger.warning(f"Could not initialize semantic chunker: {e}")
                self.semantic_splitter = None
        else:
            self.semantic_splitter = None
            logger.info("Semantic chunker not available, using recursive splitter only")

    def split_documents(self, documents: List[LCDocument]) -> List[LCDocument]:
        """Split documents using semantic chunking"""
        if self.semantic_splitter is not None:
            return self.semantic_splitter.split_documents(documents)
        else:
            # Fallback to recursive splitting
            logger.warning("Semantic splitter not available, falling back to recursive splitting")
            fallback_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                keep_separator=True
            )
            return fallback_splitter.split_documents(documents)

class HybridTextSplitter:
    """Hybrid splitter that combines adaptive and semantic splitting"""

    def __init__(self, embeddings, analysis: Dict[str, Any]):
        self.embeddings = embeddings
        self.analysis = analysis
        self.adaptive_splitter = AdaptiveTextSplitter(analysis)
        if SEMANTIC_CHUNKER_AVAILABLE:
            try:
                self.semantic_splitter = SemanticTextSplitter(embeddings, analysis)
            except Exception as e:
                logger.warning(f"Could not initialize semantic text splitter: {e}")
                self.semantic_splitter = None
        else:
            self.semantic_splitter = None

    def split_documents(self, documents: List[LCDocument]) -> List[LCDocument]:
        """Split documents using hybrid approach"""

        # For very large documents, use semantic splitting
        if self.analysis['avg_doc_length'] > 100000 and self.semantic_splitter is not None:
            logger.info("Using semantic splitting for large documents")
            return self.semantic_splitter.split_documents(documents)

        # For legal documents with high complexity, use semantic splitting
        elif (self.analysis['is_legal_document'] and 
              self.analysis['complexity_score'] > 0.6 and 
              self.semantic_splitter is not None):
            logger.info("Using semantic splitting for complex legal documents")
            return self.semantic_splitter.split_documents(documents)

        # For other cases, use adaptive splitting
        else:
            logger.info("Using adaptive splitting for standard documents")
            return self.adaptive_splitter.split_documents(documents)

class EnhancedMetadataExtractor:
    """Enhanced metadata extraction for legal documents"""

    def __init__(self):
        self.extraction_patterns = {
            'sections': re.compile(r'Section\s+(\d+|[IVXLCDM]+)', re.IGNORECASE),
            'articles': re.compile(r'Article\s+(\d+|[IVXLCDM]+)', re.IGNORECASE),
            'chapters': re.compile(r'Chapter\s+(\d+|[IVXLCDM]+)', re.IGNORECASE),
            'clauses': re.compile(r'Clause\s+(\d+|[IVXLCDM]+)', re.IGNORECASE),
            'court_mentions': re.compile(r'\b(?:Supreme Court|High Court|Constitutional Court|District Court)\b', re.IGNORECASE),
            'legal_terms': re.compile(r'\b(?:petitioner|respondent|appellant|defendant|plaintiff|judgment|verdict|precedent|constitution|amendment)\b', re.IGNORECASE),
            'citations': re.compile(r'\[(\d+)\]|(\(\d{4}\))', re.IGNORECASE),
            'dates': re.compile(r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b', re.IGNORECASE)
        }

    def extract_metadata(self, documents: List[LCDocument]) -> List[LCDocument]:
        """Extract enhanced metadata from documents"""

        enhanced_docs = []

        for i, doc in enumerate(documents):
            content = doc.page_content
            metadata = doc.metadata.copy()

            # Basic metadata
            metadata.update({
                'chunk_id': i,
                'word_count': len(content.split()),
                'char_count': len(content),
                'has_citations': bool(self.extraction_patterns['citations'].search(content)),
                'has_dates': bool(self.extraction_patterns['dates'].search(content))
            })

            # Extract legal elements
            legal_elements = self._extract_legal_elements(content)
            metadata.update(legal_elements)

            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(content)
            metadata.update(quality_metrics)

            # Create enhanced document
            enhanced_doc = LCDocument(
                page_content=content,
                metadata=metadata
            )
            enhanced_docs.append(enhanced_doc)

        logger.info(f"Enhanced metadata extraction completed for {len(enhanced_docs)} documents")
        return enhanced_docs

    def _extract_legal_elements(self, content: str) -> Dict[str, Any]:
        """Extract legal-specific elements from content"""

        elements = {
            'sections': [],
            'articles': [],
            'chapters': [],
            'clauses': [],
            'court_mentions': [],
            'legal_terms': [],
            'citation_numbers': []
        }

        # Extract sections, articles, chapters, clauses
        for key, pattern in [('sections', 'sections'), ('articles', 'articles'),
                           ('chapters', 'chapters'), ('clauses', 'clauses')]:
            matches = self.extraction_patterns[pattern].findall(content)
            elements[key] = ', '.join(list(set(matches))) if matches else ''  # Convert to comma-separated string

        # Extract court mentions
        court_matches = self.extraction_patterns['court_mentions'].findall(content)
        elements['court_mentions'] = ', '.join(list(set(court_matches))) if court_matches else ''

        # Extract legal terms
        legal_term_matches = self.extraction_patterns['legal_terms'].findall(content)
        elements['legal_terms'] = ', '.join(list(set(legal_term_matches))) if legal_term_matches else ''

        # Extract citation numbers
        citation_matches = self.extraction_patterns['citations'].findall(content)
        citation_numbers = []
        for match in citation_matches:
            # Handle tuple results from regex with capturing groups
            if isinstance(match, tuple):
                # Take the first non-empty group
                citation = next((group for group in match if group), '')
                if citation:
                    citation_numbers.append(citation)
            elif match:
                citation_numbers.append(match)
        elements['citation_numbers'] = ', '.join(citation_numbers) if citation_numbers else ''

        return elements

    def _calculate_quality_metrics(self, content: str) -> Dict[str, Any]:
        """Calculate quality metrics for the chunk"""

        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]

        metrics = {
            'sentence_count': len(sentences),
            'avg_sentence_length': sum(len(s.split()) for s in sentences) / max(1, len(sentences)),
            'has_complete_sentences': len([s for s in sentences if len(s.split()) > 3]) > 0,
            'readability_score': self._calculate_readability(content)
        }

        return metrics

    def _calculate_readability(self, content: str) -> float:
        """Simple readability calculation"""
        words = content.split()
        sentences = re.split(r'[.!?]+', content)
        sentences = [s for s in sentences if s.strip()]

        if not words or not sentences:
            return 0.0

        avg_words_per_sentence = len(words) / len(sentences)
        avg_syllables_per_word = sum(self._count_syllables(word) for word in words) / len(words)

        # Simplified readability score (lower is easier to read)
        readability = 0.39 * avg_words_per_sentence + 11.8 * avg_syllables_per_word - 15.59
        return max(0.0, min(100.0, readability))

    def _count_syllables(self, word: str) -> int:
        """Simple syllable counting"""
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        if word[0] in vowels:
            count += 1
        for i in range(1, len(word)):
            if word[i] in vowels and word[i - 1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        return max(1, count)

class ChunkQualityValidator:
    """Validates chunk quality and filters low-quality chunks"""

    def __init__(self, min_quality_score: float = 0.6):
        self.min_quality_score = min_quality_score

    def validate_chunks(self, chunks: List[LCDocument]) -> Tuple[List[LCDocument], Dict[str, Any]]:
        """Validate chunks and return quality statistics"""

        validated_chunks = []
        quality_stats = {
            'total_chunks': len(chunks),
            'validated_chunks': 0,
            'rejected_chunks': 0,
            'avg_quality_score': 0.0,
            'quality_distribution': {'high': 0, 'medium': 0, 'low': 0}
        }

        total_quality = 0.0

        for chunk in chunks:
            quality_score = self._calculate_quality_score(chunk)

            if quality_score >= self.min_quality_score:
                # Add quality score to metadata
                chunk.metadata['quality_score'] = quality_score
                validated_chunks.append(chunk)
                quality_stats['validated_chunks'] += 1

                # Categorize quality
                if quality_score >= 0.8:
                    quality_stats['quality_distribution']['high'] += 1
                elif quality_score >= 0.6:
                    quality_stats['quality_distribution']['medium'] += 1
                else:
                    quality_stats['quality_distribution']['low'] += 1
            else:
                quality_stats['rejected_chunks'] += 1

            total_quality += quality_score

        quality_stats['avg_quality_score'] = total_quality / max(1, len(chunks))

        logger.info(f"Chunk validation completed: {quality_stats}")
        return validated_chunks, quality_stats

    def _calculate_quality_score(self, chunk: LCDocument) -> float:
        """Calculate comprehensive quality score for a chunk"""

        content = chunk.page_content
        metadata = chunk.metadata

        score = 0.0
        max_score = 0.0

        # Length factor (30% weight)
        word_count = metadata.get('word_count', len(content.split()))
        if 50 <= word_count <= 2000:
            score += 0.3
        elif word_count < 50:
            score += 0.1
        max_score += 0.3

        # Sentence completeness factor (25% weight)
        if metadata.get('has_complete_sentences', False):
            score += 0.25
        max_score += 0.25

        # Sentence length factor (15% weight)
        avg_sentence_length = metadata.get('avg_sentence_length', 15)
        if 10 <= avg_sentence_length <= 50:
            score += 0.15
        elif 5 <= avg_sentence_length <= 60:
            score += 0.1
        max_score += 0.15

        # Legal content factor (20% weight)
        legal_indicators = (
            len(metadata.get('sections', [])) +
            len(metadata.get('articles', [])) +
            len(metadata.get('chapters', [])) +
            len(metadata.get('clauses', [])) +
            len(metadata.get('court_mentions', [])) +
            len(metadata.get('legal_terms', []))
        )
        if legal_indicators > 0:
            score += min(0.2, legal_indicators * 0.05)
        max_score += 0.2

        # Citation factor (10% weight)
        if metadata.get('has_citations', False):
            score += 0.1
        max_score += 0.1

        return score / max_score if max_score > 0 else 0.0

class EnhancedDocumentProcessor:
    """Main processor that orchestrates all document processing steps"""

    def __init__(self, embeddings=None):
        self.embeddings = embeddings
        self.analyzer = DocumentStructureAnalyzer()
        self.metadata_extractor = EnhancedMetadataExtractor()
        self.quality_validator = ChunkQualityValidator()

    def process_documents(self, documents: List[LCDocument]) -> Tuple[List[LCDocument], Dict[str, Any]]:
        """Complete document processing pipeline"""

        logger.info("Starting enhanced document processing pipeline")

        # Step 1: Analyze document structure
        analysis = self.analyzer.analyze_documents(documents)

        # Step 2: Create hybrid text splitter (adaptive + semantic)
        if self.embeddings:
            splitter = HybridTextSplitter(self.embeddings, analysis)
        else:
            # Fallback to adaptive splitter if no embeddings provided
            splitter = AdaptiveTextSplitter(analysis)

        # Step 3: Split documents
        chunks = splitter.split_documents(documents)
        logger.info(f"Documents split into {len(chunks)} chunks")

        # Step 4: Extract enhanced metadata
        enhanced_chunks = self.metadata_extractor.extract_metadata(chunks)

        # Step 5: Validate chunk quality
        validated_chunks, quality_stats = self.quality_validator.validate_chunks(enhanced_chunks)

        # Combine statistics
        processing_stats = {
            'analysis': analysis,
            'quality_stats': quality_stats,
            'final_chunk_count': len(validated_chunks),
            'processing_steps': [
                'structure_analysis',
                'hybrid_splitting',
                'metadata_extraction',
                'quality_validation'
            ]
        }

        logger.info(f"Document processing completed: {processing_stats}")
        return validated_chunks, processing_stats

def create_enhanced_ingestion_pipeline_lazy(data_path: str, config_obj, embeddings: Optional[Any] = None, batch_size: int = 10) -> Tuple[List[LCDocument], Dict[str, Any]]:
    """Create and run the enhanced ingestion pipeline with lazy loading"""

    logger.info("=== Starting Enhanced Lazy Ingestion Pipeline ===")

    # Use lazy loading for large document collections
    loader = PyPDFDirectoryLoader(data_path)

    # Get total document count for progress tracking
    all_docs = list(loader.load())
    total_docs = len(all_docs)

    if total_docs == 0:
        logger.warning(f"No documents found in {data_path}")
        return [], {}

    logger.info(f"Found {total_docs} documents, processing in batches of {batch_size}")

    # Process documents in batches to manage memory
    all_processed_chunks = []
    total_stats = {
        'total_documents': total_docs,
        'batches_processed': 0,
        'total_chunks': 0,
        'processing_times': []
    }

    for i in range(0, total_docs, batch_size):
        batch_docs = all_docs[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_docs + batch_size - 1) // batch_size

        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_docs)} documents)")

        # Process this batch
        processor = EnhancedDocumentProcessor(embeddings=embeddings)
        batch_chunks, batch_stats = processor.process_documents(batch_docs)

        all_processed_chunks.extend(batch_chunks)

        # Update statistics
        total_stats['batches_processed'] += 1
        total_stats['total_chunks'] += len(batch_chunks)
        if 'processing_time' in batch_stats:
            total_stats['processing_times'].append(batch_stats['processing_time'])

        logger.info(f"Batch {batch_num} completed: {len(batch_chunks)} chunks generated")

    # Calculate final statistics
    final_stats = {
        'total_documents': total_docs,
        'total_chunks': len(all_processed_chunks),
        'batches_processed': total_stats['batches_processed'],
        'avg_chunks_per_doc': len(all_processed_chunks) / max(1, total_docs),
        'avg_batch_time': sum(total_stats['processing_times']) / max(1, len(total_stats['processing_times'])) if total_stats['processing_times'] else 0,
        'processing_method': 'lazy_batch_processing'
    }

    logger.info("=== Enhanced Lazy Ingestion Pipeline Completed ===")
    logger.info(f"Processed {total_docs} documents into {len(all_processed_chunks)} chunks")

    return all_processed_chunks, final_stats