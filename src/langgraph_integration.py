# src/langgraph_integration.py
"""
LangGraph Integration Module

This module provides seamless integration of LangGraph workflows
with the existing Legal Research Engine components.
"""

import logging
import streamlit as st
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

from .langgraph_workflow import get_legal_workflow, LegalResearchWorkflow
from .retrieval import create_rag_chain
from .monitoring import get_monitor

logger = logging.getLogger(__name__)

class LangGraphRAGIntegration:
    """
    Integration layer that combines LangGraph workflows with existing RAG chain
    """
    
    def __init__(self):
        self.workflow = get_legal_workflow()
        self.rag_chain, self.sync_rag_chain = create_rag_chain()
        self.monitor = get_monitor()
        
    def enhanced_query_processing(
        self, 
        query: str, 
        use_langgraph: bool = True,
        documents: Optional[List[Document]] = None
    ) -> Dict[str, Any]:
        """
        Process query using either LangGraph workflow or traditional RAG
        
        Args:
            query: User query
            use_langgraph: Whether to use LangGraph workflow (True) or simple RAG (False)
            documents: Optional documents to analyze
            
        Returns:
            Dictionary with processing results
        """
        try:
            if use_langgraph:
                logger.info("Processing query with LangGraph workflow")
                result = self.workflow.execute_research(query, documents)
                
                return {
                    "method": "langgraph",
                    "query": query,
                    "response": result.get("final_report", "No response generated"),
                    "confidence_score": result.get("confidence_score", 0.0),
                    "completed_steps": result.get("completed_steps", []),
                    "errors": result.get("errors", []),
                    "citations": result.get("citations", []),
                    "recommendations": result.get("recommendations", []),
                    "case_brief": result.get("case_brief"),
                    "precedent_analysis": result.get("precedent_analysis")
                }
            else:
                logger.info("Processing query with traditional RAG")
                response = self.rag_chain.invoke(query)
                
                return {
                    "method": "traditional_rag",
                    "query": query,
                    "response": response,
                    "confidence_score": 0.7,  # Default confidence for RAG
                    "completed_steps": ["rag_retrieval"],
                    "errors": [],
                    "citations": [],
                    "recommendations": ["Consider using advanced workflow for deeper analysis"]
                }
                
        except Exception as e:
            logger.error(f"Error in query processing: {e}")
            return {
                "method": "error",
                "query": query,
                "response": f"Error processing query: {str(e)}",
                "confidence_score": 0.0,
                "completed_steps": [],
                "errors": [str(e)],
                "citations": [],
                "recommendations": ["Please try again or contact support"]
            }


def create_langgraph_streamlit_interface():
    """
    Create Streamlit interface components for LangGraph workflow
    """
    
    st.subheader("🔄 Advanced Legal Research Workflow (LangGraph)")
    
    # Workflow selection
    workflow_mode = st.selectbox(
        "Select Research Mode:",
        ["Traditional RAG (Fast)", "Advanced LangGraph Workflow (Comprehensive)"],
        help="Choose between fast RAG responses or comprehensive multi-agent analysis"
    )
    
    use_langgraph = "Advanced LangGraph" in workflow_mode
    
    # Advanced options for LangGraph
    if use_langgraph:
        st.info("🚀 Advanced mode uses multi-agent analysis including case briefs, precedent analysis, and citation extraction")
        
        with st.expander("Advanced Options"):
            include_precedent_analysis = st.checkbox("Include Precedent Analysis", value=True)
            include_case_brief = st.checkbox("Generate Case Brief", value=True)
            include_citation_extraction = st.checkbox("Extract Citations", value=True)
            
            confidence_threshold = st.slider(
                "Minimum Confidence Threshold",
                0.0, 1.0, 0.7,
                help="Minimum confidence score for recommendations"
            )
    
    return {
        "use_langgraph": use_langgraph,
        "include_precedent_analysis": use_langgraph and include_precedent_analysis if use_langgraph else False,
        "include_case_brief": use_langgraph and include_case_brief if use_langgraph else False,
        "include_citation_extraction": use_langgraph and include_citation_extraction if use_langgraph else False,
        "confidence_threshold": confidence_threshold if use_langgraph else 0.7
    }


def display_langgraph_results(result: Dict[str, Any]):
    """
    Display LangGraph workflow results in Streamlit
    """
    
    method = result.get("method", "unknown")
    
    # Main response
    st.subheader("📋 Research Results")
    
    if method == "langgraph":
        # Confidence indicator
        confidence = result.get("confidence_score", 0.0)
        confidence_color = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.4 else "🔴"
        st.metric(
            "Analysis Confidence", 
            f"{confidence:.2f}",
            delta=f"{confidence_color} {'High' if confidence > 0.7 else 'Medium' if confidence > 0.4 else 'Low'}"
        )
        
        # Completed steps
        completed_steps = result.get("completed_steps", [])
        if completed_steps:
            st.success(f"✅ Completed: {', '.join(completed_steps)}")
        
        # Errors if any
        errors = result.get("errors", [])
        if errors:
            with st.expander("⚠️ Warnings/Errors", expanded=False):
                for error in errors:
                    st.warning(error)
    
    # Main response
    response = result.get("response", "No response generated")
    st.markdown(response)
    
    # Additional sections for LangGraph
    if method == "langgraph":
        
        # Case Brief
        case_brief = result.get("case_brief")
        if case_brief and case_brief.strip() and "No documents available" not in case_brief:
            with st.expander("📄 Case Brief", expanded=False):
                st.markdown(case_brief)
        
        # Precedent Analysis  
        precedent_analysis = result.get("precedent_analysis")
        if precedent_analysis and precedent_analysis.strip() and "No research results available" not in precedent_analysis:
            with st.expander("⚖️ Precedent Analysis", expanded=False):
                st.markdown(precedent_analysis)
        
        # Citations
        citations = result.get("citations", [])
        if citations:
            with st.expander("📚 Legal Citations", expanded=False):
                for i, citation in enumerate(citations, 1):
                    st.write(f"{i}. {citation}")
        
        # Recommendations
        recommendations = result.get("recommendations", [])
        if recommendations:
            with st.expander("💡 Recommendations", expanded=False):
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"{i}. {rec}")


def create_workflow_monitoring_dashboard():
    """
    Create a monitoring dashboard for LangGraph workflows
    """
    st.subheader("📊 Workflow Monitoring Dashboard")
    
    try:
        workflow = get_legal_workflow()
        status = workflow.get_workflow_status()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Workflow Nodes", status.get("workflow_nodes", 0))
        
        with col2: 
            st.metric("Workflow Edges", status.get("workflow_edges", 0))
        
        with col3:
            st.metric("Status", status.get("status", "Unknown"))
        
        # Workflow Visualization
        if st.button("🔍 View Workflow Graph"):
            try:
                workflow_viz = workflow.get_workflow_visualization()
                if workflow_viz:
                    st.image(workflow_viz, caption="LangGraph Workflow Structure")
                else:
                    st.error("Unable to generate workflow visualization")
            except Exception as e:
                st.error(f"Error generating visualization: {e}")
        
        # Performance Metrics
        monitor = get_monitor()
        if hasattr(monitor, 'get_metrics_summary'):
            with st.expander("📈 Performance Metrics", expanded=False):
                try:
                    metrics = monitor.get_metrics_summary()
                    st.json(metrics)
                except Exception as e:
                    st.error(f"Error loading metrics: {e}")
                    
    except Exception as e:
        st.error(f"Error loading workflow status: {e}")


# Export main functions
__all__ = [
    'LangGraphRAGIntegration',
    'create_langgraph_streamlit_interface',
    'display_langgraph_results',
    'create_workflow_monitoring_dashboard'
]