# src/case_brief_generator.py

import re
import logging
from typing import Dict, List, Optional, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class CaseBrief(BaseModel):
    """Structured case brief model"""
    case_name: str = Field(description="Full case name with citation")
    court: str = Field(description="Court that decided the case")
    judges: List[str] = Field(description="Names of the judges")
    date: str = Field(description="Date of judgment")
    appellant: str = Field(description="Name of appellant/petitioner")
    respondent: str = Field(description="Name of respondent")
    counsel_appellant: List[str] = Field(description="Counsel for appellant")
    counsel_respondent: List[str] = Field(description="Counsel for respondent")
    facts: str = Field(description="Brief statement of facts")
    legal_issues: List[str] = Field(description="Key legal issues/questions")
    arguments_appellant: str = Field(description="Arguments of appellant")
    arguments_respondent: str = Field(description="Arguments of respondent")
    holdings: List[str] = Field(description="Court's holdings/decisions")
    ratio_decidendi: str = Field(description="Ratio decidendi/principle of law")
    obiter_dicta: str = Field(description="Obiter dicta/observations")
    final_order: str = Field(description="Final order/disposition")

class CaseBriefGenerator:
    """Generate structured case briefs from legal documents"""

    def __init__(self, api_key: Optional[str] = None):
        # No API key required for local LM Studio
        self.llm = ChatOpenAI(
            model="local-model",  # LM Studio model name (will be overridden by LM Studio)
            openai_api_base="http://127.0.0.1:1234/v1",  # LM Studio local endpoint
            openai_api_key="lm-studio",  # Dummy key for local LM Studio
            temperature=0.1,  # Lower temperature for consistent legal analysis
            max_tokens=2048  # Reasonable token limit for case briefs
        )

        self.parser = PydanticOutputParser(pydantic_object=CaseBrief)

    def extract_case_info_regex(self, text: str) -> Dict[str, str]:
        """Extract basic case information using regex patterns"""
        info = {}

        # Case name patterns
        case_name_patterns = [
            r'(?:IN THE\s+)?(?:SUPREME COURT OF INDIA|HIGH COURT OF\s+\w+).*?(?:CIVIL|CRIMINAL).*?(?:APPEAL|WRIT|PETITION).*?(?:\n|\r)',
            r'(?:IN THE\s+)?(?:MATTER OF|CASE OF).*?(?:\n|\r)',
            r'([A-Z][A-Z\s,&]+(?:LIMITED|PVT\.?\s*LTD\.?|CORPORATION|CORPN\.?|COMPANY|CO\.?)?\s*(?:VS?\.?|VERSUS)\s*[A-Z][A-Z\s,&]+(?:LIMITED|PVT\.?\s*LTD\.?|CORPORATION|CORPN\.?|COMPANY|CO\.?)?)'
        ]

        for pattern in case_name_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                info['case_name'] = match.group(0).strip()
                break

        # Court patterns
        court_patterns = [
            r'SUPREME COURT OF INDIA',
            r'HIGH COURT OF\s+(\w+)',
            r'(?:AT\s+)?([A-Z\s]+COURT)'
        ]

        for pattern in court_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                info['court'] = match.group(0).strip()
                break

        # Date patterns
        date_patterns = [
            r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',
            r'\b\d{4}-\d{2}-\d{2}\b'
        ]

        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                info['date'] = match.group(0).strip()
                break

        return info

    def generate_case_brief(self, document: Document) -> CaseBrief:
        """Generate a structured case brief from a document"""

        text = document.page_content
        basic_info = self.extract_case_info_regex(text)

        # Create the prompt for case brief generation
        prompt_template = """
        You are a legal expert tasked with creating a comprehensive case brief from the provided court document.

        Extract and structure the following information from the document:

        1. Case Name: Full case citation and name
        2. Court: The court that decided the case
        3. Judges: Names of the judges who decided the case
        4. Date: Date of the judgment
        5. Parties: Appellant/Petitioner and Respondent
        6. Counsel: Lawyers appearing for each party
        7. Facts: Brief statement of relevant facts
        8. Legal Issues: Key legal questions/issues before the court
        9. Arguments: Main arguments of both sides
        10. Holdings: Court's decisions on each issue
        11. Ratio Decidendi: The principle of law established
        12. Obiter Dicta: Important observations not part of ratio
        13. Final Order: The final disposition/order

        Document content:
        {content}

        Basic extracted information:
        {basic_info}

        Generate a complete, accurate case brief in the specified JSON format.
        If certain information is not available in the document, use "Not specified" or provide the best inference.
        """

        prompt = ChatPromptTemplate.from_template(prompt_template)

        # Format the prompt
        formatted_prompt = prompt.format(
            content=text[:10000],  # Limit content length
            basic_info=str(basic_info)
        )

        try:
            # Generate the case brief
            response = self.llm.invoke(formatted_prompt)
            brief_data = self.parser.parse(response.content)

            logger.info(f"Successfully generated case brief for: {brief_data.case_name}")
            return brief_data

        except Exception as e:
            logger.error(f"Error generating case brief: {e}")
            # Return a basic brief with available information
            return CaseBrief(
                case_name=basic_info.get('case_name', 'Unknown Case'),
                court=basic_info.get('court', 'Not specified'),
                judges=[],
                date=basic_info.get('date', 'Not specified'),
                appellant="Not specified",
                respondent="Not specified",
                counsel_appellant=[],
                counsel_respondent=[],
                facts="Could not extract facts from document",
                legal_issues=[],
                arguments_appellant="Not specified",
                arguments_respondent="Not specified",
                holdings=[],
                ratio_decidendi="Not specified",
                obiter_dicta="Not specified",
                final_order="Not specified"
            )

    def format_brief_for_display(self, brief: CaseBrief) -> str:
        """Format the case brief for display in Streamlit"""
        formatted = f"""
# 📋 **CASE BRIEF**

## **Case Citation**
**{brief.case_name}**

## **Court & Bench**
**Court:** {brief.court}  
**Judges:** {', '.join(brief.judges) if brief.judges else 'Not specified'}  
**Date:** {brief.date}

## **Parties**
**Appellant/Petitioner:** {brief.appellant}  
**Respondent:** {brief.respondent}

## **Counsel**
**For Appellant:** {', '.join(brief.counsel_appellant) if brief.counsel_appellant else 'Not specified'}  
**For Respondent:** {', '.join(brief.counsel_respondent) if brief.counsel_respondent else 'Not specified'}

## **Statement of Facts**
{brief.facts}

## **Legal Issues**
"""
        if brief.legal_issues:
            for i, issue in enumerate(brief.legal_issues, 1):
                formatted += f"{i}. {issue}\n"
        else:
            formatted += "Not specified\n"

        formatted += f"""
## **Arguments**

### **Appellant's Arguments**
{brief.arguments_appellant}

### **Respondent's Arguments**
{brief.arguments_respondent}

## **Court's Holdings**
"""
        if brief.holdings:
            for i, holding in enumerate(brief.holdings, 1):
                formatted += f"{i}. {holding}\n"
        else:
            formatted += "Not specified\n"

        formatted += f"""
## **Ratio Decidendi**
{brief.ratio_decidendi}

## **Obiter Dicta**
{brief.obiter_dicta}

## **Final Order**
{brief.final_order}
"""
        return formatted

def batch_generate_briefs(documents: List[Document], api_key: Optional[str] = None) -> List[Tuple[Document, CaseBrief]]:
    """Generate case briefs for multiple documents"""
    generator = CaseBriefGenerator(api_key)
    results = []

    for doc in documents:
        try:
            brief = generator.generate_case_brief(doc)
            results.append((doc, brief))
        except Exception as e:
            logger.error(f"Failed to generate brief for document: {e}")
            continue

    return results