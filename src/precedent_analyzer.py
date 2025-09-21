# src/precedent_analyzer.py

import logging
from typing import List, Dict, Optional, Tuple
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import numpy as np
import os

logger = logging.getLogger(__name__)

class PrecedentAnalysis(BaseModel):
    """Analysis of legal precedents"""
    similar_cases: List[Dict[str, str]] = Field(description="List of similar cases with relevance scores")
    key_principles: List[str] = Field(description="Key legal principles from precedents")
    distinguishing_factors: List[str] = Field(description="Factors that distinguish this case")
    precedential_value: str = Field(description="Assessment of precedential value")
    research_suggestions: List[str] = Field(description="Suggestions for further research")

class PrecedentAnalyzer:
    """Analyze legal precedents and find similar cases"""

    def __init__(self, vectorstore_path: str, api_key: Optional[str] = None):
        # No API key required for local LM Studio
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-embeddinggemma-300m-qat",  # LM Studio model name
            openai_api_base="http://127.0.0.1:1234/v1",  # LM Studio local endpoint
            openai_api_key="lm-studio",  # Dummy key for local LM Studio
            check_embedding_ctx_length=False
        )

        self.llm = ChatOpenAI(
            model="local-model",  # LM Studio model name (will be overridden by LM Studio)
            openai_api_base="http://127.0.0.1:1234/v1",  # LM Studio local endpoint
            openai_api_key="lm-studio",  # Dummy key for local LM Studio
            temperature=0.1,  # Lower temperature for consistent legal analysis
            max_tokens=2048
        )

        self.vectorstore = Chroma(
            persist_directory=vectorstore_path,
            embedding_function=self.embeddings
        )

        self.parser = PydanticOutputParser(pydantic_object=PrecedentAnalysis)

    def find_similar_cases(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Find similar cases using semantic search"""
        try:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
            docs = retriever.invoke(query)

            # Get similarity scores (if available)
            results = []
            for doc in docs:
                # Chroma doesn't return scores directly, so we'll use a default relevance
                score = 0.8  # Placeholder - in production, you'd get actual similarity scores
                results.append((doc, score))

            return results

        except Exception as e:
            logger.error(f"Error finding similar cases: {e}")
            return []

    def analyze_precedents(self, current_case: Document, similar_cases: List[Document]) -> PrecedentAnalysis:
        """Analyze precedents and their relevance to current case"""

        # Prepare context from similar cases
        precedents_context = ""
        for i, case in enumerate(similar_cases[:3], 1):  # Limit to top 3 for analysis
            precedents_context += f"\nPrecedent {i}:\n{case.page_content[:1000]}...\n"

        prompt_template = """
        You are a legal expert analyzing precedents for a current case. Based on the current case and similar precedents provided, perform a comprehensive precedent analysis.

        Current Case:
        {current_case}

        Similar Precedents:
        {precedents}

        Please provide:
        1. Similar Cases: List the most relevant precedents with brief explanations of similarity
        2. Key Principles: Extract key legal principles from these precedents
        3. Distinguishing Factors: Identify factors that might distinguish the current case
        4. Precedential Value: Assess the precedential value and binding nature
        5. Research Suggestions: Suggest further research directions

        Format your response as a structured analysis.
        """

        prompt = ChatPromptTemplate.from_template(prompt_template)

        try:
            formatted_prompt = prompt.format(
                current_case=current_case.page_content[:2000],
                precedents=precedents_context
            )

            response = self.llm.invoke(formatted_prompt)
            analysis = self.parser.parse(response.content)

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing precedents: {e}")
            return PrecedentAnalysis(
                similar_cases=[],
                key_principles=[],
                distinguishing_factors=[],
                precedential_value="Analysis failed",
                research_suggestions=[]
            )

    def generate_precedent_report(self, current_case: Document, analysis: PrecedentAnalysis) -> str:
        """Generate a formatted precedent analysis report"""

        report = f"""
# ⚖️ **PRECEDENT ANALYSIS REPORT**

## **Current Case Summary**
{current_case.page_content[:500]}...

## **Similar Cases Found**
"""

        for i, case in enumerate(analysis.similar_cases, 1):
            report += f"""
### **{i}. {case.get('case_name', 'Similar Case')}**
**Relevance:** {case.get('relevance', 'High')}
**Key Similarity:** {case.get('similarity_reason', 'Similar legal issues')}
"""

        report += f"""
## **Key Legal Principles**
"""
        for principle in analysis.key_principles:
            report += f"• {principle}\n"

        report += f"""
## **Distinguishing Factors**
"""
        for factor in analysis.distinguishing_factors:
            report += f"• {factor}\n"

        report += f"""
## **Precedential Value Assessment**
{analysis.precedential_value}

## **Research Suggestions**
"""
        for suggestion in analysis.research_suggestions:
            report += f"• {suggestion}\n"

        return report

class LegalIssueExtractor:
    """Extract key legal issues from case documents"""

    def __init__(self, api_key: Optional[str] = None):
        # No API key required for local LM Studio
        self.llm = ChatOpenAI(
            model="local-model",  # LM Studio model name
            openai_api_base="http://127.0.0.1:1234/v1",  # LM Studio local endpoint
            openai_api_key="lm-studio",  # Dummy key for local LM Studio
            temperature=0.1,  # Lower temperature for consistent legal analysis
            max_tokens=2048
        )

    def extract_legal_issues(self, document: Document) -> Dict[str, List[str]]:
        """Extract key legal issues, holdings, and ratios from a case"""

        prompt_template = """
        Analyze the following court document and extract:

        1. Legal Issues/Questions: The key legal questions before the court
        2. Holdings: The court's decisions on each issue
        3. Ratio Decidendi: The binding principles of law established
        4. Key Evidence: Important facts or evidence considered
        5. Legal Reasoning: The court's reasoning process

        Document:
        {content}

        Provide a structured extraction focusing on the most important elements.
        """

        prompt = ChatPromptTemplate.from_template(prompt_template)

        try:
            formatted_prompt = prompt.format(content=document.page_content[:3000])
            response = self.llm.invoke(formatted_prompt)

            # Parse the response (simplified parsing)
            content = response.content

            return {
                "legal_issues": self._extract_section(content, "Legal Issues", "Holdings"),
                "holdings": self._extract_section(content, "Holdings", "Ratio"),
                "ratio_decidendi": self._extract_section(content, "Ratio", "Key Evidence"),
                "key_evidence": self._extract_section(content, "Key Evidence", "Legal Reasoning"),
                "legal_reasoning": self._extract_section(content, "Legal Reasoning", "")
            }

        except Exception as e:
            logger.error(f"Error extracting legal issues: {e}")
            return {
                "legal_issues": [],
                "holdings": [],
                "ratio_decidendi": [],
                "key_evidence": [],
                "legal_reasoning": []
            }

    def _extract_section(self, content: str, start_marker: str, end_marker: str) -> List[str]:
        """Extract a section from the LLM response"""
        try:
            start = content.find(start_marker)
            if start == -1:
                return []

            start = start + len(start_marker)
            if end_marker:
                end = content.find(end_marker, start)
                if end == -1:
                    end = len(content)
            else:
                end = len(content)

            section = content[start:end].strip()

            # Split into bullet points or numbered items
            items = []
            for line in section.split('\n'):
                line = line.strip()
                if line and (line.startswith('•') or line.startswith('-') or line[0].isdigit()):
                    items.append(line.lstrip('•-0123456789. '))

            return items

        except Exception:
            return []