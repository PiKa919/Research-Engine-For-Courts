# LangGraph Integration Guide

## 🚀 What is LangGraph?

LangGraph is a powerful framework for building stateful, multi-agent applications with LLMs. Unlike traditional LangChain chains, LangGraph provides:

- **Multi-agent orchestration** - Coordinate multiple specialized AI agents
- **Stateful execution** - Maintain context across complex workflows  
- **Error recovery** - Handle failures gracefully with checkpoints
- **Human-in-the-loop** - Built-in support for human approval workflows
- **Production reliability** - Enterprise-grade monitoring and observability

## 🏗️ Architecture Overview

The Legal Research Engine now includes a sophisticated LangGraph workflow with specialized agents:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Orchestrator  │───▶│ Document Retrieval│───▶│   Case Brief    │
│    Planner      │    │      Agent       │    │     Agent       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Report          │◀───│   Citation       │◀───│   Precedent     │
│  Synthesizer    │    │  Extraction      │    │   Analysis      │
└─────────────────┘    │     Agent        │    │     Agent       │
                       └──────────────────┘    └─────────────────┘
```

## 🤖 Available Agents

### 1. **Orchestrator** 🎯
- Plans the research workflow
- Coordinates agent execution
- Manages workflow state and routing

### 2. **Document Retrieval Agent** 📚  
- Performs RAG-based document search
- Retrieves relevant legal documents
- Provides initial research context

### 3. **Case Brief Agent** 📄
- Generates structured case summaries
- Extracts key legal facts and holdings
- Creates professional legal briefs

### 4. **Precedent Analysis Agent** ⚖️
- Identifies similar cases and legal principles
- Analyzes legal precedents
- Provides comparative case analysis

### 5. **Citation Extraction Agent** 📝
- Extracts legal citations automatically
- Validates citation formats
- Categorizes citation types

### 6. **Report Synthesizer** 🔄
- Combines all agent outputs
- Generates comprehensive reports
- Provides recommendations and confidence scores

## 🛠️ Installation & Setup

LangGraph is already included in your updated requirements.txt. If you need to install it manually:

```bash
conda activate legal
pip install langgraph
```

## 📚 Usage Examples

### 1. **Streamlit Interface**

The main application now includes advanced LangGraph functionality:

```python
# Run the updated Streamlit app
streamlit run app.py
```

Navigate to the "🤖 Advanced Chat" tab and select "Advanced LangGraph Workflow (Comprehensive)" for multi-agent analysis.

### 2. **Python API**

```python
from src.langgraph_integration import LangGraphRAGIntegration

# Initialize integration
integration = LangGraphRAGIntegration()

# Simple query with traditional RAG
result = integration.enhanced_query_processing(
    query="What are commercial court provisions?",
    use_langgraph=False  # Fast traditional RAG
)

# Advanced query with LangGraph workflow  
result = integration.enhanced_query_processing(
    query="What are commercial court provisions?",
    use_langgraph=True,  # Multi-agent analysis
    documents=your_documents
)

print(f"Confidence: {result['confidence_score']}")
print(f"Steps completed: {result['completed_steps']}")
print(f"Response: {result['response']}")
```

### 3. **Direct Workflow Usage**

```python
from src.langgraph_workflow import get_legal_workflow

# Get workflow instance
workflow = get_legal_workflow()

# Execute research
result = workflow.execute_research(
    query="Legal question here",
    documents=document_list  # Optional
)

print(result['final_report'])
```

### 4. **REST API**

Start the API server:
```bash
uvicorn src.api:app --reload --port 8000
```

Use the advanced endpoint:
```python
import requests

response = requests.post("http://localhost:8000/query/advanced", json={
    "text": "What are the commercial court provisions?",
    "use_langgraph": True,
    "include_case_brief": True,
    "include_precedent_analysis": True,
    "include_citations": True
})

result = response.json()
print(result['response'])
```

## 🔍 Example Demo

Run the included example script:

```bash
conda activate legal
python example_langgraph_usage.py
```

This will demonstrate:
- Traditional RAG vs LangGraph comparison
- Multi-agent workflow execution
- Workflow monitoring and status

## ⚡ Performance Comparison

| Feature | Traditional RAG | LangGraph Workflow |
|---------|----------------|-------------------|
| **Response Time** | ⚡ Fast (2-5s) | 🕐 Moderate (10-30s) |
| **Analysis Depth** | 🔵 Basic | 🟢 Comprehensive |
| **Error Handling** | 🔴 Limited | 🟢 Advanced |
| **Citation Extraction** | 🔴 None | 🟢 Automatic |
| **Precedent Analysis** | 🔴 None | 🟢 Detailed |
| **Case Brief Generation** | 🔴 None | 🟢 Structured |
| **Multi-step Reasoning** | 🔴 Limited | 🟢 Multi-agent |
| **Human-in-Loop** | 🔴 None | 🟢 Built-in |
| **Production Ready** | 🟡 Moderate | 🟢 Enterprise |

## 🎯 When to Use What?

### Use **Traditional RAG** when:
- You need fast responses (< 5 seconds)
- Simple Q&A is sufficient
- Budget/compute constraints exist
- Prototyping or testing

### Use **LangGraph Workflow** when:
- Comprehensive legal analysis is needed
- Multiple perspectives are required
- Production reliability is critical
- Human oversight is necessary
- Complex legal reasoning is involved

## 🔧 Configuration

Key configuration options in `src/config.py`:

```python
# LangGraph-specific settings
LANGGRAPH_CONFIG = {
    "max_workflow_time": 300,  # seconds
    "enable_checkpoints": True,
    "max_retries": 3,
    "parallel_agents": True
}
```

## 📊 Monitoring & Observability

The workflow includes comprehensive monitoring:

1. **Workflow Status**: Track node/edge execution
2. **Performance Metrics**: Response times and success rates  
3. **Error Tracking**: Detailed error reporting
4. **Confidence Scoring**: Analysis quality assessment

Access monitoring via:
- Streamlit: "🔄 Workflow Monitor" tab
- API: `/workflow/status` endpoint
- Direct: `workflow.get_workflow_status()`

## 🚨 Troubleshooting

### Common Issues:

1. **LM Studio Connection**
   ```bash
   # Verify LM Studio is running on localhost:1234
   curl http://localhost:1234/v1/models
   ```

2. **Slow Performance**
   - Use smaller models in LM Studio
   - Reduce document batch sizes
   - Enable parallel processing

3. **Memory Issues**
   - Increase system RAM
   - Use model quantization
   - Process fewer documents at once

4. **Workflow Errors**
   - Check individual agent logs
   - Verify document availability
   - Ensure proper model loading

## 🎓 Learning Resources

- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [Multi-Agent Patterns](https://python.langchain.com/docs/concepts/multi_agent)
- [Workflow Examples](https://github.com/langchain-ai/langgraph/tree/main/examples)

## 🤝 Contributing

To contribute to the LangGraph integration:

1. Fork the repository
2. Create feature branches for new agents
3. Add comprehensive tests
4. Update documentation
5. Submit pull requests

## 📝 License

This LangGraph integration maintains the same MIT license as the main project.

---

**Ready to experience advanced legal research with AI agents? Start with the Streamlit interface and explore the comprehensive multi-agent analysis!** 🚀