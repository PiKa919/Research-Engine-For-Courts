# src/api.py

import sys
import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, List

from .retrieval import create_rag_chain
from .monitoring import get_monitor, log_performance_summary
from .langgraph_integration import LangGraphRAGIntegration

# Initialize FastAPI app
app = FastAPI(
    title="Legal Research Engine API with LangGraph",
    description="Advanced API for the AI-Driven Research Engine with LangGraph workflows.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="."), name="static")

# Create both sync and async RAG chains on startup
async_rag_chain, sync_rag_chain = create_rag_chain()

# Initialize LangGraph integration
langgraph_integration = LangGraphRAGIntegration()

# Pydantic models
class Query(BaseModel):
    text: str

class AdvancedQuery(BaseModel):
    text: str
    use_langgraph: bool = True
    include_case_brief: bool = True
    include_precedent_analysis: bool = True
    include_citations: bool = True

@app.get("/", summary="Root endpoint")
async def root():
    """
    Provides a welcome message and API information.
    """
    return {
        "message": "Welcome to the Legal Research Engine API with LangGraph!",
        "version": "1.0.0",
        "features": [
            "LangGraph multi-agent workflows",
            "Advanced legal research analysis",
            "Case brief generation",
            "Precedent analysis",
            "Citation extraction",
            "Async query processing",
            "Enhanced retrieval with query expansion",
            "Multi-level caching",
            "Legal document analysis",
            "Knowledge graph generation"
        ],
        "endpoints": {
            "query": "/query/ (async)",
            "advanced_query": "/query/advanced (LangGraph)",
            "sync_query": "/query/sync",
            "health": "/health",
            "workflow_status": "/workflow/status",
            "docs": "/docs"
        }
    }

@app.get("/health", summary="Health check endpoint")
async def health_check():
    """
    Provides health status and system information.
    """
    monitor = get_monitor()
    system_stats = monitor.get_system_stats()

    return {
        "status": "healthy",
        "async_support": True,
        "caching_enabled": True,
        "query_processing": "enhanced",
        "system_stats": system_stats,
        "timestamp": system_stats["timestamp"]
    }

@app.get("/metrics", summary="Performance metrics endpoint")
async def get_metrics(minutes: int = 60):
    """
    Returns performance metrics for the last N minutes.
    """
    monitor = get_monitor()
    summary = monitor.get_performance_summary(last_n_minutes=minutes)

    return {
        "metrics": summary,
        "time_range_minutes": minutes,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/metrics/log", summary="Log performance summary")
async def log_metrics():
    """
    Logs current performance metrics to the application logs.
    """
    log_performance_summary()
    return {"message": "Performance summary logged"}

@app.get("/metrics/export", summary="Export metrics to file")
async def export_metrics(filepath: str = "performance_metrics.json"):
    """
    Exports performance metrics to a JSON file.
    """
    monitor = get_monitor()
    monitor.export_metrics(filepath)
    return {"message": f"Metrics exported to {filepath}"}

# Import datetime for timestamp
from datetime import datetime


@app.post("/query/", summary="Perform a RAG query")
async def perform_query(query: Query):
    """
    Accepts a user query, processes it through the async RAG chain, and returns
    the answer with enhanced context and metadata.
    """
    try:
        # Use async RAG chain by default
        response = await async_rag_chain(query.text)
        return {
            "answer": response["answer"],
            "metadata": response.get("metadata", {}),
            "processing_type": "async"
        }
    except Exception as e:
        # Fallback to sync version
        response = sync_rag_chain(query.text)
        return {
            "answer": response["answer"],
            "metadata": response.get("metadata", {}),
            "processing_type": "sync_fallback",
            "error": str(e)
        }


@app.post("/query/advanced", summary="Perform advanced LangGraph query")
async def perform_advanced_query(query: AdvancedQuery):
    """
    Performs advanced legal research using LangGraph multi-agent workflow.
    Provides comprehensive analysis including case briefs, precedent analysis, and citations.
    """
    try:
        result = langgraph_integration.enhanced_query_processing(
            query=query.text,
            use_langgraph=query.use_langgraph,
            documents=None  # API doesn't pass documents directly
        )
        
        return {
            "success": True,
            "query": query.text,
            "method": result.get("method", "unknown"),
            "response": result.get("response", "No response generated"),
            "confidence_score": result.get("confidence_score", 0.0),
            "completed_steps": result.get("completed_steps", []),
            "errors": result.get("errors", []),
            "citations": result.get("citations", []),
            "recommendations": result.get("recommendations", []),
            "case_brief": result.get("case_brief"),
            "precedent_analysis": result.get("precedent_analysis"),
            "processing_time": result.get("processing_time"),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "query": query.text,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.get("/workflow/status", summary="Get workflow status")
async def get_workflow_status():
    """
    Returns the current status of the LangGraph workflow system.
    """
    try:
        from .langgraph_workflow import get_legal_workflow
        workflow = get_legal_workflow()
        status = workflow.get_workflow_status()
        
        return {
            "success": True,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.post("/query/sync", summary="Perform a synchronous RAG query")
async def perform_sync_query(query: Query):
    """
    Accepts a user query, processes it through the sync RAG chain, and returns
    the answer. Useful for debugging or when async processing is not needed.
    """
    response = sync_rag_chain(query.text)
    return {
        "answer": response["answer"],
        "metadata": response.get("metadata", {}),
        "processing_type": "sync"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)