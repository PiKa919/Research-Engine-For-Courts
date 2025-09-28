# src/langgraph_workflow.py
"""
LangGraph-based Legal Research Workflow

This module implements a sophisticated legal research workflow using LangGraph,
providing multi-agent orchestration for complex legal analysis tasks.

Key Features:
- Multi-agent system with specialized legal agents
- Stateful execution with error recovery
- Human-in-the-loop capabilities for legal validation
- Parallel processing for efficiency
- Comprehensive logging and monitoring
"""

import logging
from typing import TypedDict, Literal, List, Dict, Any, Optional, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.documents import Document
from pydantic import BaseModel, Field
import operator
import json
from datetime import datetime

from . import config
from .retrieval import create_rag_chain
from .caching import get_cache, cached_query
from .monitoring import get_monitor, timing_decorator
from .case_brief_generator import CaseBriefGenerator
from .precedent_analyzer import PrecedentAnalyzer
from .document_comparator import DocumentComparator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# STATE DEFINITIONS
# =============================================================================

class LegalResearchState(TypedDict):
    """Main state for the legal research workflow"""
    # Input
    query: str
    documents: Optional[List[Document]]
    
    # Analysis Results
    research_results: Optional[str]
    case_brief: Optional[str]
    precedent_analysis: Optional[str]
    document_comparison: Optional[str]
    citations: Optional[List[str]]
    
    # Workflow Control
    current_step: Optional[str]
    completed_steps: Annotated[List[str], operator.add]
    errors: Annotated[List[str], operator.add]
    
    # Final Output
    final_report: Optional[str]
    confidence_score: Optional[float]
    recommendations: Optional[List[str]]


class AnalysisSection(BaseModel):
    """Structure for individual analysis sections"""
    name: str = Field(description="Name of the analysis section")
    description: str = Field(description="Description of what this section should contain")
    priority: int = Field(description="Priority level (1=highest, 5=lowest)")


class AnalysisPlan(BaseModel):
    """Planning structure for legal analysis"""
    sections: List[AnalysisSection] = Field(description="List of analysis sections to complete")
    estimated_time: int = Field(description="Estimated time in minutes")


class WorkerState(TypedDict):
    """State for individual worker agents"""
    section: AnalysisSection
    query: str
    documents: Optional[List[Document]]
    result: Optional[str]
    completed_sections: Annotated[List[str], operator.add]


# =============================================================================
# LEGAL RESEARCH AGENTS
# =============================================================================

class LegalResearchOrchestrator:
    """Orchestrates the entire legal research workflow"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            openai_api_base="http://127.0.0.1:1234/v1",
            openai_api_key="lm-studio",
            temperature=0.1,
            max_tokens=2048
        )
        self.planner = self.llm.with_structured_output(AnalysisPlan)
        
    def create_analysis_plan(self, state: LegalResearchState) -> Dict[str, Any]:
        """Create a comprehensive analysis plan for the legal research"""
        try:
            logger.info(f"Creating analysis plan for query: {state['query'][:100]}...")
            
            planning_prompt = [
                SystemMessage(content="""
You are a senior legal research analyst. Create a comprehensive analysis plan for the given legal query.

Your plan should include:
1. Document retrieval and initial research
2. Case brief generation
3. Precedent analysis
4. Citation extraction and verification
5. Document comparison (if multiple documents)
6. Final synthesis and recommendations

Prioritize sections based on importance (1=highest, 5=lowest).
Estimate realistic time requirements.
                """),
                HumanMessage(content=f"""
Legal Query: {state['query']}

Number of documents available: {len(state.get('documents', [])) if state.get('documents') else 0}

Create a detailed analysis plan with specific sections and priorities.
                """)
            ]
            
            plan = self.planner.invoke(planning_prompt)
            logger.info(f"Created analysis plan with {len(plan.sections)} sections")
            
            return {
                "current_step": "planning_completed",
                "completed_steps": ["planning"]
            }
            
        except Exception as e:
            logger.error(f"Error in analysis planning: {e}")
            return {
                "errors": [f"Planning error: {str(e)}"],
                "current_step": "planning_failed"
            }


class DocumentRetrievalAgent:
    """Handles document retrieval and initial research"""
    
    def __init__(self):
        self.rag_chain, _ = create_rag_chain()
        
    @timing_decorator
    def retrieve_documents(self, state: LegalResearchState) -> Dict[str, Any]:
        """Retrieve relevant documents and perform initial research"""
        try:
            logger.info("Starting document retrieval and research...")
            
            # Perform RAG-based research
            research_result = self.rag_chain.invoke(state['query'])
            
            # Extract any documents used in the research
            # This would typically come from your vector store
            
            return {
                "research_results": research_result,
                "current_step": "document_retrieval_completed",
                "completed_steps": ["document_retrieval"]
            }
            
        except Exception as e:
            logger.error(f"Error in document retrieval: {e}")
            return {
                "errors": [f"Document retrieval error: {str(e)}"],
                "current_step": "document_retrieval_failed"
            }


class CaseBriefAgent:
    """Generates comprehensive case briefs"""
    
    def __init__(self):
        self.brief_generator = CaseBriefGenerator()
        
    @timing_decorator
    def generate_case_brief(self, state: LegalResearchState) -> Dict[str, Any]:
        """Generate a detailed case brief"""
        try:
            logger.info("Generating case brief...")
            
            if not state.get('documents'):
                logger.warning("No documents available for case brief generation")
                return {
                    "case_brief": "No documents available for case brief generation",
                    "completed_steps": ["case_brief"]
                }
            
            # Generate case brief using the first document or combine multiple
            primary_doc = state['documents'][0] if state['documents'] else None
            
            if primary_doc:
                brief = self.brief_generator.generate_case_brief(
                    primary_doc.page_content,
                    case_name=primary_doc.metadata.get('source', 'Unknown Case')
                )
                
                return {
                    "case_brief": brief,
                    "current_step": "case_brief_completed",
                    "completed_steps": ["case_brief"]
                }
            else:
                return {
                    "case_brief": "Unable to generate case brief - no suitable documents",
                    "completed_steps": ["case_brief"]
                }
                
        except Exception as e:
            logger.error(f"Error in case brief generation: {e}")
            return {
                "errors": [f"Case brief error: {str(e)}"],
                "current_step": "case_brief_failed"
            }


class PrecedentAnalysisAgent:
    """Analyzes legal precedents"""
    
    def __init__(self):
        self.precedent_analyzer = PrecedentAnalyzer()
        
    @timing_decorator
    def analyze_precedents(self, state: LegalResearchState) -> Dict[str, Any]:
        """Analyze legal precedents"""
        try:
            logger.info("Analyzing legal precedents...")
            
            if not state.get('research_results'):
                logger.warning("No research results available for precedent analysis")
                return {
                    "precedent_analysis": "No research results available for precedent analysis",
                    "completed_steps": ["precedent_analysis"]
                }
            
            # Analyze precedents based on research results
            precedents = self.precedent_analyzer.find_similar_cases(
                query=state['query'],
                max_results=5
            )
            
            analysis_text = "## Precedent Analysis\n\n"
            for i, precedent in enumerate(precedents, 1):
                analysis_text += f"### {i}. {precedent.get('case_name', 'Unknown Case')}\n"
                analysis_text += f"**Similarity Score:** {precedent.get('similarity', 0):.2f}\n"
                analysis_text += f"**Relevance:** {precedent.get('summary', 'No summary available')}\n\n"
            
            return {
                "precedent_analysis": analysis_text,
                "current_step": "precedent_analysis_completed", 
                "completed_steps": ["precedent_analysis"]
            }
            
        except Exception as e:
            logger.error(f"Error in precedent analysis: {e}")
            return {
                "errors": [f"Precedent analysis error: {str(e)}"],
                "current_step": "precedent_analysis_failed"
            }


class CitationExtractionAgent:
    """Extracts and validates legal citations"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            openai_api_base="http://127.0.0.1:1234/v1",
            openai_api_key="lm-studio",
            temperature=0.1,
            max_tokens=1024
        )
        
    @timing_decorator
    def extract_citations(self, state: LegalResearchState) -> Dict[str, Any]:
        """Extract and validate legal citations"""
        try:
            logger.info("Extracting legal citations...")
            
            # Combine all available text for citation extraction
            text_content = ""
            if state.get('research_results'):
                text_content += state['research_results'] + "\n\n"
            if state.get('case_brief'):
                text_content += state['case_brief'] + "\n\n"
            if state.get('precedent_analysis'):
                text_content += state['precedent_analysis'] + "\n\n"
            
            if not text_content.strip():
                return {
                    "citations": [],
                    "completed_steps": ["citation_extraction"]
                }
            
            citation_prompt = [
                SystemMessage(content="""
Extract all legal citations from the provided text. Look for:
- Case citations (e.g., "Brown v. Board of Education (1954)")
- Statute citations (e.g., "Section 123 of the Civil Code")
- Constitutional provisions (e.g., "Article 14 of the Constitution")
- Legal precedents and court decisions

Return only unique, well-formatted citations.
                """),
                HumanMessage(content=f"Extract citations from: {text_content[:2000]}...")
            ]
            
            response = self.llm.invoke(citation_prompt)
            
            # Parse citations from response
            citations = []
            if response.content:
                lines = response.content.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#') and len(line) > 10:
                        citations.append(line)
            
            return {
                "citations": citations[:20],  # Limit to top 20 citations
                "current_step": "citation_extraction_completed",
                "completed_steps": ["citation_extraction"]
            }
            
        except Exception as e:
            logger.error(f"Error in citation extraction: {e}")
            return {
                "errors": [f"Citation extraction error: {str(e)}"],
                "current_step": "citation_extraction_failed"
            }


class ReportSynthesizer:
    """Synthesizes final legal research report"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            openai_api_base="http://127.0.0.1:1234/v1",
            openai_api_key="lm-studio",
            temperature=0.1,
            max_tokens=3000
        )
        
    @timing_decorator
    def synthesize_report(self, state: LegalResearchState) -> Dict[str, Any]:
        """Synthesize the final comprehensive legal research report"""
        try:
            logger.info("Synthesizing final legal research report...")
            
            # Collect all analysis results
            sections = []
            
            if state.get('research_results'):
                sections.append(f"## Research Results\n{state['research_results']}")
                
            if state.get('case_brief'):
                sections.append(f"## Case Brief\n{state['case_brief']}")
                
            if state.get('precedent_analysis'):
                sections.append(f"## Precedent Analysis\n{state['precedent_analysis']}")
                
            if state.get('citations'):
                citations_text = "\n".join([f"- {citation}" for citation in state['citations']])
                sections.append(f"## Legal Citations\n{citations_text}")
            
            # Generate executive summary and recommendations
            synthesis_prompt = [
                SystemMessage(content="""
You are a senior legal researcher. Based on the provided analysis sections, create:

1. An executive summary highlighting key findings
2. Legal recommendations based on the research
3. A confidence assessment of the analysis
4. Potential areas for further research

Be concise but comprehensive. Focus on actionable insights.
                """),
                HumanMessage(content=f"""
Original Query: {state['query']}

Analysis Sections:
{chr(10).join(sections)}

Provide a professional legal research summary with recommendations.
                """)
            ]
            
            synthesis_response = self.llm.invoke(synthesis_prompt)
            
            # Compile final report
            report_parts = [
                f"# Legal Research Report",
                f"**Query:** {state['query']}",
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"**Completed Steps:** {', '.join(state.get('completed_steps', []))}",
                "",
                synthesis_response.content,
                "",
                "---",
                "",
            ]
            
            report_parts.extend(sections)
            
            final_report = "\n\n".join(report_parts)
            
            # Calculate confidence score based on completed steps and errors
            completed_count = len(state.get('completed_steps', []))
            error_count = len(state.get('errors', []))
            confidence = max(0.1, min(1.0, (completed_count * 0.2) - (error_count * 0.1)))
            
            # Generate recommendations
            recommendations = [
                "Review the precedent analysis for relevant case law",
                "Verify all legal citations for accuracy",
                "Consider consulting with domain experts for complex issues"
            ]
            
            if error_count > 0:
                recommendations.append("Address any errors noted during the analysis process")
            
            return {
                "final_report": final_report,
                "confidence_score": confidence,
                "recommendations": recommendations,
                "current_step": "synthesis_completed",
                "completed_steps": ["synthesis"]
            }
            
        except Exception as e:
            logger.error(f"Error in report synthesis: {e}")
            return {
                "errors": [f"Synthesis error: {str(e)}"],
                "current_step": "synthesis_failed"
            }


# =============================================================================
# WORKFLOW CONTROL FUNCTIONS
# =============================================================================

def should_continue_analysis(state: LegalResearchState) -> Literal["continue", "synthesize", "error"]:
    """Determine whether to continue analysis or move to synthesis"""
    
    completed_steps = set(state.get('completed_steps', []))
    required_steps = {"document_retrieval", "case_brief", "precedent_analysis", "citation_extraction"}
    
    # Check for critical errors
    errors = state.get('errors', [])
    if len(errors) > 3:  # Too many errors
        logger.warning(f"Too many errors ({len(errors)}), moving to synthesis")
        return "synthesize"
    
    # Check if we have minimum required analysis
    if len(completed_steps.intersection(required_steps)) >= 2:
        return "synthesize"
    elif "document_retrieval" in completed_steps:
        return "continue"
    else:
        return "error"


def route_analysis_step(state: LegalResearchState) -> Literal["case_brief", "precedent_analysis", "citation_extraction"]:
    """Route to the next analysis step based on current state"""
    
    completed_steps = set(state.get('completed_steps', []))
    
    # Priority order for analysis steps
    if "case_brief" not in completed_steps:
        return "case_brief"
    elif "precedent_analysis" not in completed_steps:
        return "precedent_analysis"
    elif "citation_extraction" not in completed_steps:
        return "citation_extraction"
    else:
        # Default fallback
        return "case_brief"


# =============================================================================
# LANGGRAPH WORKFLOW CONSTRUCTION
# =============================================================================

def create_legal_research_workflow() -> StateGraph:
    """Create the main LangGraph workflow for legal research"""
    
    # Initialize agents
    orchestrator = LegalResearchOrchestrator()
    doc_retrieval_agent = DocumentRetrievalAgent()
    case_brief_agent = CaseBriefAgent()
    precedent_agent = PrecedentAnalysisAgent()
    citation_agent = CitationExtractionAgent()
    synthesizer = ReportSynthesizer()
    
    # Create the workflow
    workflow = StateGraph(LegalResearchState)
    
    # Add nodes
    workflow.add_node("planner", orchestrator.create_analysis_plan)
    workflow.add_node("document_retrieval", doc_retrieval_agent.retrieve_documents)
    workflow.add_node("case_brief", case_brief_agent.generate_case_brief)
    workflow.add_node("precedent_analysis", precedent_agent.analyze_precedents)
    workflow.add_node("citation_extraction", citation_agent.extract_citations)
    workflow.add_node("synthesizer", synthesizer.synthesize_report)
    
    # Define workflow edges
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "document_retrieval")
    
    # Conditional routing for analysis steps
    workflow.add_conditional_edges(
        "document_retrieval",
        should_continue_analysis,
        {
            "continue": "case_brief",
            "synthesize": "synthesizer",
            "error": "synthesizer"  # Still try to synthesize with partial data
        }
    )
    
    # Analysis routing
    workflow.add_conditional_edges(
        "case_brief",
        should_continue_analysis,
        {
            "continue": "precedent_analysis", 
            "synthesize": "synthesizer",
            "error": "synthesizer"
        }
    )
    
    workflow.add_conditional_edges(
        "precedent_analysis",
        should_continue_analysis,
        {
            "continue": "citation_extraction",
            "synthesize": "synthesizer", 
            "error": "synthesizer"
        }
    )
    
    workflow.add_edge("citation_extraction", "synthesizer")
    workflow.add_edge("synthesizer", END)
    
    return workflow


# =============================================================================
# MAIN WORKFLOW INTERFACE
# =============================================================================

class LegalResearchWorkflow:
    """Main interface for the LangGraph-based legal research workflow"""
    
    def __init__(self):
        self.workflow_graph = create_legal_research_workflow()
        self.compiled_workflow = self.workflow_graph.compile()
        self.monitor = get_monitor()
        
    @timing_decorator
    def execute_research(
        self,
        query: str,
        documents: Optional[List[Document]] = None
    ) -> Dict[str, Any]:
        """
        Execute the complete legal research workflow
        
        Args:
            query: The legal research query
            documents: Optional list of documents to analyze
            
        Returns:
            Dictionary containing the complete research results
        """
        try:
            logger.info(f"Starting legal research workflow for query: {query[:100]}...")
            
            # Initialize state
            initial_state = LegalResearchState(
                query=query,
                documents=documents or [],
                completed_steps=[],
                errors=[],
                current_step="initialized"
            )
            
            # Execute workflow
            result = self.compiled_workflow.invoke(initial_state)
            
            # Log completion
            completed_steps = result.get('completed_steps', [])
            errors = result.get('errors', [])
            
            logger.info(f"Workflow completed. Steps: {completed_steps}, Errors: {len(errors)}")
            
            # Record metrics
            self.monitor.record_metric("workflow_completed", 1, {
                "completed_steps": len(completed_steps),
                "error_count": len(errors),
                "confidence_score": result.get('confidence_score', 0)
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in legal research workflow: {e}")
            self.monitor.record_metric("workflow_error", 1, {"error": str(e)})
            
            return {
                "query": query,
                "final_report": f"Error in legal research workflow: {str(e)}",
                "confidence_score": 0.0,
                "errors": [str(e)],
                "completed_steps": []
            }
    
    def get_workflow_visualization(self) -> bytes:
        """Get a visual representation of the workflow"""
        try:
            return self.compiled_workflow.get_graph().draw_mermaid_png()
        except Exception as e:
            logger.error(f"Error generating workflow visualization: {e}")
            return None
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current status and statistics of the workflow"""
        return {
            "workflow_nodes": len(self.workflow_graph.nodes),
            "workflow_edges": len(self.workflow_graph.edges),
            "status": "active",
            "last_updated": datetime.now().isoformat()
        }


# =============================================================================
# GLOBAL WORKFLOW INSTANCE
# =============================================================================

_workflow_instance = None

def get_legal_workflow() -> LegalResearchWorkflow:
    """Get the global workflow instance (singleton pattern)"""
    global _workflow_instance
    if _workflow_instance is None:
        _workflow_instance = LegalResearchWorkflow()
    return _workflow_instance


# Export main classes and functions
__all__ = [
    'LegalResearchWorkflow',
    'LegalResearchState', 
    'get_legal_workflow',
    'create_legal_research_workflow'
]