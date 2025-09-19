# src/document_comparator.py

import logging
from typing import List, Dict, Optional, Tuple
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
import difflib
import os

logger = logging.getLogger(__name__)

class ComparisonResult(BaseModel):
    """Result of document comparison"""
    similarities: List[str] = Field(description="Key similarities between documents")
    differences: List[str] = Field(description="Key differences between documents")
    legal_implications: List[str] = Field(description="Legal implications of differences")
    precedential_conflicts: List[str] = Field(description="Potential precedential conflicts")
    recommendation: str = Field(description="Recommendation for handling differences")

class DocumentComparator:
    """Compare multiple legal documents for similarities and differences"""

    def __init__(self, api_key: Optional[str] = None):
        # No API key required for local LM Studio
        self.llm = ChatOpenAI(
            model="local-model",  # LM Studio model name (will be overridden by LM Studio)
            openai_api_base="http://127.0.0.1:1234/v1",  # LM Studio local endpoint
            openai_api_key="lm-studio",  # Dummy key for local LM Studio
            temperature=0.1,  # Lower temperature for consistent legal analysis
            max_tokens=2048
        )

        self.parser = PydanticOutputParser(pydantic_object=ComparisonResult)

    def compare_documents(self, docs: List[Document]) -> ComparisonResult:
        """Compare multiple documents and identify similarities/differences"""

        if len(docs) < 2:
            return ComparisonResult(
                similarities=[],
                differences=[],
                legal_implications=[],
                precedential_conflicts=[],
                recommendation="At least 2 documents required for comparison"
            )

        # Prepare document content for comparison
        doc_contents = []
        for i, doc in enumerate(docs):
            content = doc.page_content[:1500]  # Limit content for analysis
            doc_contents.append(f"Document {i+1}:\n{content}")

        combined_content = "\n\n".join(doc_contents)

        prompt_template = """
        You are a legal expert comparing multiple court documents. Analyze the following documents and provide a detailed comparison:

        Documents to compare:
        {documents}

        Please provide:
        1. Similarities: Key similarities in facts, legal issues, or holdings
        2. Differences: Important differences in approach, reasoning, or outcomes
        3. Legal Implications: What the differences mean legally
        4. Precedential Conflicts: Any conflicts with established precedents
        5. Recommendation: How to reconcile or address the differences

        Focus on legal analysis rather than textual similarity.
        """

        prompt = ChatPromptTemplate.from_template(prompt_template)

        try:
            formatted_prompt = prompt.format(documents=combined_content)
            response = self.llm.invoke(formatted_prompt)
            comparison = self.parser.parse(response.content)

            return comparison

        except Exception as e:
            logger.error(f"Error comparing documents: {e}")
            return ComparisonResult(
                similarities=[],
                differences=[],
                legal_implications=[],
                precedential_conflicts=[],
                recommendation="Comparison failed due to error"
            )

    def generate_text_diff(self, doc1: Document, doc2: Document) -> str:
        """Generate a text diff between two documents"""

        text1 = doc1.page_content
        text2 = doc2.page_content

        # Create unified diff
        diff = list(difflib.unified_diff(
            text1.splitlines(keepends=True),
            text2.splitlines(keepends=True),
            fromfile=f"{doc1.metadata.get('source', 'Doc1')}",
            tofile=f"{doc2.metadata.get('source', 'Doc2')}",
            lineterm=''
        ))

        return ''.join(diff)

    def find_common_legal_terms(self, docs: List[Document]) -> Dict[str, List[str]]:
        """Find common legal terms and phrases across documents"""

        legal_terms = {
            "constitutional": ["fundamental rights", "article", "constitution", "supreme court"],
            "criminal": ["ipc", "crpc", "accused", "prosecution", "bail", "sentence"],
            "civil": ["contract", "tort", "damages", "injunction", "specific performance"],
            "evidence": ["witness", "testimony", "documentary evidence", "oral evidence"],
            "procedure": ["jurisdiction", "limitation", "res judicata", "maintainability"]
        }

        results = {}

        for category, terms in legal_terms.items():
            common_terms = []
            for term in terms:
                found_in_docs = []
                for i, doc in enumerate(docs):
                    if term.lower() in doc.page_content.lower():
                        found_in_docs.append(f"Doc {i+1}")

                if len(found_in_docs) > 1:  # Found in multiple documents
                    common_terms.append(f"'{term}' found in: {', '.join(found_in_docs)}")

            if common_terms:
                results[category] = common_terms

        return results

class CaseTimelineBuilder:
    """Build timeline of related cases and proceedings"""

    def __init__(self, api_key: Optional[str] = None):
        # No API key required for local LM Studio
        self.llm = ChatOpenAI(
            model="local-model",  # LM Studio model name
            openai_api_base="http://127.0.0.1:1234/v1",  # LM Studio local endpoint
            openai_api_key="lm-studio",  # Dummy key for local LM Studio
            temperature=0.1,  # Lower temperature for consistent legal analysis
            max_tokens=2048
        )

    def build_timeline(self, documents: List[Document]) -> List[Dict[str, str]]:
        """Build a chronological timeline from case documents"""

        # Extract dates and events from each document
        timeline_events = []

        for doc in documents:
            content = doc.page_content

            # Extract dates (simple pattern matching)
            date_patterns = [
                r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',
                r'\b\d{4}-\d{2}-\d{2}\b'
            ]

            dates_found = []
            for pattern in date_patterns:
                matches = re.findall(pattern, content)
                dates_found.extend(matches)

            # Extract key events (simplified)
            event_keywords = [
                "filed", "petition", "appeal", "hearing", "judgment", "order",
                "dismissed", "allowed", "granted", "denied", "decided"
            ]

            events_found = []
            for keyword in event_keywords:
                if keyword.lower() in content.lower():
                    events_found.append(keyword.title())

            if dates_found and events_found:
                timeline_events.append({
                    "date": dates_found[0],  # Take first date found
                    "event": f"{', '.join(events_found)} - {doc.metadata.get('source', 'Unknown')}",
                    "document": doc.metadata.get('source', 'Unknown')
                })

        # Sort by date (simplified sorting)
        try:
            timeline_events.sort(key=lambda x: x["date"])
        except:
            pass  # If date parsing fails, keep original order

        return timeline_events

    def generate_timeline_report(self, timeline_events: List[Dict[str, str]]) -> str:
        """Generate a formatted timeline report"""

        report = "# 📅 **CASE TIMELINE**\n\n"

        if not timeline_events:
            report += "No timeline events could be extracted from the documents.\n"
            return report

        for event in timeline_events:
            report += f"**{event['date']}**: {event['event']}\n\n"

        return report