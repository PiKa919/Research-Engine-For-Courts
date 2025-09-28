# example_langgraph_usage.py
"""
Example script demonstrating LangGraph usage in the Legal Research Engine

This script shows how to use the new LangGraph workflows for legal research.
Run this after your documents are ingested and LM Studio is running.
"""

import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.langgraph_workflow import get_legal_workflow
from src.langgraph_integration import LangGraphRAGIntegration
from langchain_community.document_loaders import PyPDFDirectoryLoader
from src.config import DATA_PATH

def load_sample_documents():
    """Load sample documents for testing"""
    try:
        if not os.path.exists(DATA_PATH):
            print(f"Data directory not found: {DATA_PATH}")
            print("Please add some PDF files to the data directory and run ingestion first.")
            return []
        
        loader = PyPDFDirectoryLoader(DATA_PATH)
        documents = loader.load()
        print(f"Loaded {len(documents)} documents from {DATA_PATH}")
        return documents
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []

def run_traditional_rag_example():
    """Example of traditional RAG processing"""
    print("\n" + "="*50)
    print("TRADITIONAL RAG EXAMPLE")
    print("="*50)
    
    integration = LangGraphRAGIntegration()
    
    query = "What are the provisions for commercial courts in India?"
    print(f"Query: {query}")
    
    result = integration.enhanced_query_processing(
        query=query,
        use_langgraph=False  # Use traditional RAG
    )
    
    print(f"\nMethod: {result['method']}")
    print(f"Response: {result['response'][:500]}...")
    print(f"Confidence: {result['confidence_score']}")

def run_langgraph_workflow_example():
    """Example of LangGraph workflow processing"""
    print("\n" + "="*50)
    print("LANGGRAPH WORKFLOW EXAMPLE")  
    print("="*50)
    
    integration = LangGraphRAGIntegration()
    documents = load_sample_documents()
    
    query = "What are the provisions for commercial courts in India?"
    print(f"Query: {query}")
    print("Processing with LangGraph multi-agent workflow...")
    
    result = integration.enhanced_query_processing(
        query=query,
        use_langgraph=True,  # Use LangGraph workflow
        documents=documents[:3] if documents else None  # Limit to first 3 docs for demo
    )
    
    print(f"\nMethod: {result['method']}")
    print(f"Confidence: {result['confidence_score']}")
    print(f"Completed Steps: {', '.join(result['completed_steps'])}")
    
    if result['errors']:
        print(f"Errors: {result['errors']}")
    
    print(f"\nMain Response:")
    print("-" * 30)
    print(result['response'][:800] + "..." if len(result['response']) > 800 else result['response'])
    
    if result.get('case_brief'):
        print(f"\nCase Brief:")
        print("-" * 30)
        brief = result['case_brief']
        print(brief[:500] + "..." if len(brief) > 500 else brief)
    
    if result.get('precedent_analysis'):
        print(f"\nPrecedent Analysis:")
        print("-" * 30)
        analysis = result['precedent_analysis']
        print(analysis[:500] + "..." if len(analysis) > 500 else analysis)
    
    if result.get('citations'):
        print(f"\nCitations ({len(result['citations'])}):")
        print("-" * 30)
        for i, citation in enumerate(result['citations'][:5], 1):  # Show first 5
            print(f"{i}. {citation}")
    
    if result.get('recommendations'):
        print(f"\nRecommendations:")
        print("-" * 30)
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"{i}. {rec}")

def run_workflow_status_example():
    """Example of checking workflow status"""
    print("\n" + "="*50)
    print("WORKFLOW STATUS EXAMPLE")
    print("="*50)
    
    try:
        workflow = get_legal_workflow()
        status = workflow.get_workflow_status()
        
        print(f"Workflow Nodes: {status['workflow_nodes']}")
        print(f"Workflow Edges: {status['workflow_edges']}")
        print(f"Status: {status['status']}")
        print(f"Last Updated: {status['last_updated']}")
        
    except Exception as e:
        print(f"Error getting workflow status: {e}")

def main():
    """Main demonstration function"""
    print("Legal Research Engine - LangGraph Integration Demo")
    print("=" * 60)
    
    # Check if LM Studio is accessible
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="local-model",
            openai_api_base="http://127.0.0.1:1234/v1",
            openai_api_key="lm-studio",
            temperature=0.1,
            max_tokens=100
        )
        test_response = llm.invoke("Hello")
        print("✅ LM Studio connection verified")
    except Exception as e:
        print(f"❌ LM Studio connection failed: {e}")
        print("Please make sure LM Studio is running on localhost:1234")
        return
    
    # Run examples
    try:
        # 1. Traditional RAG example
        run_traditional_rag_example()
        
        # 2. LangGraph workflow example
        run_langgraph_workflow_example()
        
        # 3. Workflow status example
        run_workflow_status_example()
        
        print("\n" + "="*60)
        print("Demo completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()