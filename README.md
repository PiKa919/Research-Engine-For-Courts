# 🏛️ Legal Research Engine - Your AI Legal Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.49.1-red.svg)](https://streamlit.io)

**A simple, powerful AI tool that helps lawyers, judges, and law students research legal cases quickly and efficiently.**

> 💡 **Think of it like having a super-smart legal assistant that can instantly read through thousands of legal documents and answer your questions!**

---

## 🎯 What Does This Tool Do?

This Legal Research Engine is like having a **personal AI lawyer** that can:

✅ **Answer legal questions** by searching through your law documents  
✅ **Find similar cases** to the one you're working on  
✅ **Create case summaries** automatically  
✅ **Show connections** between different legal cases  
✅ **Extract important citations** from legal documents  
✅ **Compare different legal documents** side by side  

### 🌟 **Perfect For:**
- **Lawyers** researching case law and precedents
- **Judges** reviewing similar cases and legal principles
- **Law Students** studying and understanding legal concepts
- **Legal Researchers** analyzing large volumes of legal documents
- **Court Staff** preparing case briefs and summaries

---

## 🚀 Quick Start (Super Easy!)

### **Step 1: Get the Prerequisites**
You'll need these programs installed on your computer:

1. **Python 3.8+** - [Download here](https://www.python.org/downloads/)
2. **LM Studio** - [Download here](https://lmstudio.ai/) *(This runs the AI on your computer)*
3. **Git** - [Download here](https://git-scm.com/) *(To download the code)*

### **Step 2: Download the Legal Research Engine**
Open your computer's terminal/command prompt and type:

```bash
# Download the legal research engine
git clone https://github.com/PiKa919/Research-Engine-For-Courts.git

# Go into the folder
cd "Research-Engine-For-Courts"
```

### **Step 3: Set Up Python Environment**
```bash
# Create a special Python environment for this project
conda create -n legal python=3.11 -y

# Switch to this environment
conda activate legal

# Install all the required tools
pip install -r requirements.txt
```

### **Step 4: Set Up LM Studio (Your AI Brain)**
1. **Open LM Studio** (the app you downloaded)
2. **Download Models:**
   - Go to the "Search" tab
   - Download: `text-embedding-embeddinggemma-300m-qat` *(for understanding documents)*
   - Download any chat model like: `Llama-2-7B-Chat` or `Mistral-7B` *(for answering questions)*
3. **Start the Server:**
   - Go to "Developer" tab
   - Load both models
   - Click "Start Server" - it should run on `http://127.0.0.1:1234`

### **Step 5: Add Your Legal Documents**
```bash
# Put your PDF legal documents in the data folder
# Then tell the system to read them
python src/ingest.py
```

### **Step 6: Start the Legal Research Engine**
```bash
# Make sure you're using the legal environment
conda activate legal

# Start the web application
streamlit run app.py
```

### **Step 7: Open in Your Browser**
- **Open your web browser** (Chrome, Firefox, etc.)
- **Go to:** `http://localhost:8501`
- **Start researching!** 🎉

---

## 📖 How to Use (Simple Guide)

### **🔍 Tab 1: Chat - Ask Legal Questions**
**What it does:** Ask questions about your legal documents and get instant answers with references.

**How to use:**
1. Type your question like: *"What are the main provisions of the Commercial Courts Act?"*
2. Press Enter
3. Get a detailed answer with citations from your documents
4. Ask follow-up questions for more details

**Example Questions:**
- *"What is the limitation period for commercial disputes?"*
- *"How are commercial court judges appointed?"*
- *"What are the key differences between civil and commercial procedures?"*

---

### **🎨 Tab 2: Themes - Change the Look**
**What it does:** Change how the app looks to suit your preference.

**Available Themes:**
- **Light Theme** - Bright and clean (default)
- **Dark Theme** - Easy on the eyes
- **Legal Blue** - Professional blue colors
- **Legal Classic** - Traditional legal styling
- **Auto** - Matches your system theme

---

### **📊 Tab 3: Citation Graph - See Case Connections**
**What it does:** Shows you how legal cases are connected to each other in a visual network.

**How to use:**
1. The system automatically analyzes your documents
2. Click and drag to explore the network
3. See which cases cite which other cases
4. Understand the relationships between legal precedents

---

### **📈 Tab 4: Citation Analysis - Document Statistics**
**What it does:** Shows you statistics about your legal document collection.

**You'll see:**
- Most cited cases
- Citation frequency charts
- Document usage patterns
- Legal trend analysis

---

### **📋 Tab 5: Case Brief Generator - Auto-Create Summaries**
**What it does:** Automatically creates professional case briefs from legal documents.

**How to use:**
1. Select a document from your collection
2. Click "Generate Brief"
3. Get a structured summary with:
   - Case facts
   - Legal issues
   - Court decision
   - Key legal principles
   - Citations

---

### **⚖️ Tab 6: Precedent Analysis - Find Similar Cases**
**What it does:** Finds cases similar to your current legal issue.

**How to use:**
1. Describe your legal situation
2. The AI finds similar cases from your documents
3. See how courts decided similar issues
4. Use these as precedents for your case

---

### **📄 Tab 7: Document Comparison - Compare Legal Docs**
**What it does:** Compares two or more legal documents side by side.

**How to use:**
1. Select documents to compare
2. See similarities and differences highlighted
3. Understand how legal positions differ
4. Export comparison reports

---

### **📤 Tab 8: Export - Save Your Research**
**What it does:** Save your research results in different formats.

**Available Formats:**
- **📄 PDF** - Professional reports for printing
- **📝 Word Document** - Editable legal documents
- **💾 Text File** - Simple text format
- **📊 JSON** - Data format for further analysis
- **🌐 HTML** - Web page format

---

### **� Tab 9: Analytics - Performance Dashboard**
**What it does:** Shows you how well the system is working and usage statistics.

**You'll see:**
- Query response times
- System performance metrics
- Usage patterns
- Success rates

---

### **🔧 Tab 10: Workflow Builder - Custom Research Processes**
**What it does:** Create custom research workflows for specific types of legal work.

**How to use:**
1. Choose your research type (case law, statutory analysis, etc.)
2. Configure the steps
3. Save as a template for future use
4. Share workflows with colleagues

---

### **📚 Tab 11: Knowledge Graph - Legal Concept Network**
**What it does:** Shows relationships between legal concepts, cases, and statutes.

**Features:**
- Interactive concept mapping
- Legal principle connections
- Statute-case relationships
- Searchable knowledge network

---

### **⚙️ Tab 12: System Settings - Configure the Tool**
**What it does:** Adjust settings to customize how the system works.

**Options:**
- Model selection
- Performance settings
- Export preferences
- Theme customization
- Cache management

---

## 🎯 **Real-World Use Cases**

### **For Lawyers:**
- **Case Research:** *"Find all cases related to breach of contract in commercial disputes"*
- **Precedent Analysis:** *"Show me similar cases to my current client's situation"*
- **Brief Preparation:** *"Generate a case brief for [specific case]"*

### **For Judges:**
- **Case Review:** *"What are the key legal principles in this type of case?"*
- **Precedent Research:** *"How have similar cases been decided?"*
- **Legal Analysis:** *"What are the statutory provisions relevant to this matter?"*

### **For Law Students:**
- **Study Aid:** *"Explain the main points of the Contract Act"*
- **Case Analysis:** *"Break down this judgment for me"*
- **Research Help:** *"Find cases that illustrate this legal principle"*

---

## 🔧 **Troubleshooting (If Something Goes Wrong)**

### **❌ Can't Connect to LM Studio**
**Problem:** The AI can't talk to LM Studio  
**Solution:** 
1. Make sure LM Studio is running
2. Check that the server is started in LM Studio
3. Verify the URL is `http://127.0.0.1:1234`

### **❌ App Won't Start**
**Problem:** Error messages when starting the app  
**Solution:**
1. Make sure you're in the right folder
2. Activate the conda environment: `conda activate legal`
3. Reinstall packages: `pip install -r requirements.txt`

### **❌ No Documents Found**
**Problem:** The system can't find your legal documents  
**Solution:**
1. Put PDF files in the `data/` folder
2. Run: `python src/ingest.py`
3. Wait for processing to complete

### **❌ Slow Responses**
**Problem:** AI takes too long to answer  
**Solution:**
1. Use smaller AI models in LM Studio
2. Close other programs to free up memory
3. Process fewer documents at once

### **❌ Out of Memory**
**Problem:** Computer runs out of memory  
**Solution:**
1. Increase RAM allocation in LM Studio
2. Use smaller models
3. Process documents in smaller batches
4. Restart the application

---

## 💡 **Tips for Best Results**

### **📚 Document Preparation:**
- **Use clear, text-based PDFs** (not scanned images)
- **Name files clearly** (e.g., "Supreme_Court_2023_Contract_Case.pdf")
- **Organize by topic** or case type for better results

### **❓ Asking Good Questions:**
- **Be specific:** Instead of *"Tell me about contracts"*, ask *"What are the essential elements of a valid contract under Indian law?"*
- **Use legal terms** when you know them
- **Ask follow-up questions** to get more detail

### **� Using Results:**
- **Always verify** AI answers with original sources
- **Cross-reference** with multiple cases
- **Use citations** provided to find original documents

---

## 🤝 **Getting Help**

### **Need Support?**
- **Check this README** first
- **Look at the troubleshooting section** above
- **Open an issue** on GitHub if you find bugs
- **Review LM Studio documentation** for AI model issues

### **Want to Improve the Tool?**
- **Suggest new features** by opening a GitHub issue
- **Report bugs** with detailed descriptions
- **Contribute code** if you're a developer

---

## 📄 **License**

This project is free to use under the MIT License. You can use it, modify it, and share it for any purpose, including commercial use.

---

## 🎉 **Final Words**

**Congratulations!** You now have a powerful AI legal research assistant at your fingertips. This tool can help you:

- **Save hours** of manual research time
- **Find relevant cases** you might have missed
- **Create professional documents** quickly
- **Understand complex legal relationships**
- **Make your legal work more efficient**

**Remember:** This AI assistant is a tool to help you, not replace your legal expertise. Always verify important information and use your professional judgment.

**Happy Legal Researching!** ⚖️✨

---

*Built with ❤️ for the legal community | Last updated: September 29, 2025*
- **🆕 Human-in-the-Loop**: Built-in approval workflows for critical decisions

### 📊 Advanced Tools
- **Citation Graph Visualization**: Interactive network of legal citations
- **🆕 Automated Case Brief Generation**: AI-generated structured legal summaries
- **🆕 Advanced Precedent Analysis**: Multi-agent case law research
- **Document Comparison**: Side-by-side analysis of multiple documents
- **Timeline Builder**: Chronological case progression tracking
- **🆕 Workflow Monitoring**: Real-time tracking of analysis progress

### 🎯 Legal Research Capabilities
- **Indian Law Focus**: Specialized for Indian legal system and courts
- **🆕 Automatic Citation Extraction**: AI-powered legal reference identification
- **Case Law Research**: Intelligent precedent finding and analysis
- **Document Similarity**: Semantic search across legal documents
- **🆕 Confidence Scoring**: Quality assessment of analysis results

## Installation

### Prerequisites
- **Python 3.8+**: Programming language runtime
- **LM Studio**: Local AI model server ([download](https://lmstudio.ai/))
- **Git**: Version control system

### System Requirements
- **RAM**: Minimum 8GB, Recommended 16GB+
- **Storage**: 5GB+ free space for models and documents
- **OS**: Windows 10/11, macOS, or Linux

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/PiKa919/Research-Engine-For-Courts.git
   cd Research-Engine-For-Courts
   ```

2. **Create conda environment**
   ```bash
   conda create -n legal python=3.11 -y
   conda activate legal
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install LM Studio**
   - Download from [lmstudio.ai](https://lmstudio.ai/)
   - Install and launch LM Studio
   - Download models:
     - Embedding: `text-embedding-embeddinggemma-300m-qat`
     - Chat: Any compatible model (Llama 2, Mistral, etc.)

5. **Configure LM Studio**
   - Go to "Developer" tab
   - Load your downloaded models
   - Start local server (should run on `http://127.0.0.1:1234`)

## Usage

### Prepare Documents
```bash
# Add PDF documents to the data/ folder
# Then run ingestion
python src/ingest.py
```

### Run the Application
```bash
# Start the Streamlit app
streamlit run app.py --server.headless true --server.port 8501

# Access at: http://localhost:8501
```

### Main Features

#### 💬 **Chat** - Interactive Legal Research
Ask questions about your legal documents and get AI-powered answers with citations.

#### 📊 **Citation Graph** - Visual Citation Network
Interactive visualization of legal citations and case relationships.

#### � **Citation Analysis** - Document Analytics
Frequency analysis and citation patterns in your document collection.

#### 📋 **Case Brief Generator** - Automated Briefs
Generate structured case briefs with key legal elements.

#### ⚖️ **Precedent Analysis** - Case Similarity
Find similar legal cases and analyze legal principles.

#### 📄 **Document Comparison** - Side-by-Side Analysis
Compare multiple legal documents with legal implications.

#### 📊 **Evaluation** - System Performance
DeepEval metrics and system accuracy assessment.
- **Embedding Model**: `text-embedding-embeddinggemma-300m-qat` or similar
- **Chat Model**: Any compatible model (Llama 2, Mistral, etc.)

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/PiKa919/Research-Engine-For-Courts.git
cd "Research Engine For Courts"
```

### 2. Create Conda Environment
```bash
# Create new environment
conda create -n legal python=3.11 -y

# Activate environment
conda activate legal
## Recent Updates

### v1.0.1 - September 2025
- **✅ Fixed** `langchain_experimental` import error - added missing package dependency
- **✅ Fixed** RAG chain TypeError - corrected chain structure for proper string input handling
- **✅ Improved** error handling and logging throughout the application
- **✅ Enhanced** documentation and setup instructions

## Project Structure

```
Research-Engine-For-Courts/
├── app.py                          # Main Streamlit application
├── src/
│   ├── __init__.py
│   ├── api.py                      # FastAPI REST endpoints
│   ├── config.py                   # Configuration management
│   ├── retrieval.py                # RAG implementation & vector search
│   ├── ingest.py                   # Document ingestion pipeline
│   ├── enhanced_document_processor.py  # Advanced document processing
│   ├── precedent_analyzer.py       # Legal precedent analysis
│   ├── document_comparator.py      # Document comparison tools
│   ├── case_brief_generator.py     # Automated case brief generation
│   ├── knowledge_graph.py          # Citation graph visualization
│   ├── evaluation.py               # System evaluation & metrics
│   ├── caching.py                  # Performance caching system
│   ├── monitoring.py               # Performance monitoring
│   └── evaluation.py               # Model evaluation tools
├── data/                           # Legal document storage
├── chroma/                         # Vector database storage
├── requirements.txt                # Python dependencies
├── pyrightconfig.json              # Python type checking
└── README.md                       # This file
```

## Configuration

### Model Configuration
```python
# LM Studio models (local)
EMBEDDING_MODEL = "text-embedding-embeddinggemma-300m-qat"
LLM_MODEL = "local-model"

# Performance settings
MAX_TOKENS = 2048
TEMPERATURE = 0.1
TOP_P = 0.9
```

### Environment Variables
```bash
# Optional: Create .env file for additional configuration
LM_STUDIO_BASE_URL=http://127.0.0.1:1234
DATA_PATH=./data
CHROMA_PATH=./chroma
```

## API Reference

### REST API Endpoints

The system provides REST API endpoints for integration:

- `GET /health` - Health check
- `POST /query/` - Synchronous legal research query
- `POST /query/async` - Asynchronous legal research query
- `GET /documents/` - List available documents
- `POST /documents/ingest` - Ingest new documents

### Python API

```python
from src.retrieval import create_rag_chain

# Initialize the system
async_rag_chain, sync_rag_chain = create_rag_chain()

# Perform legal research
result = sync_rag_chain("What are the provisions for commercial courts?")
print(result["answer"])
```

## Troubleshooting

### Common Issues

#### LM Studio Connection
```bash
# Check if LM Studio server is running
curl http://127.0.0.1:1234/v1/models
```

#### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### Vector Store Issues
```bash
# Reset vector database
rm -rf chroma/
python src/ingest.py
```

#### Memory Issues
- Increase RAM allocation in LM Studio
- Use smaller models
- Process documents in smaller batches

## Performance

### Benchmarks
- **Document Processing**: ~50 pages/minute
- **Query Response**: < 3 seconds average
- **Vector Search**: < 100ms for similarity search
- **Memory Usage**: ~2GB base + 0.5GB per 1000 documents

### Optimization Tips
- Use GPU acceleration in LM Studio for better performance
- Batch document ingestion for large collections
- Configure appropriate chunk sizes for your documents
- Monitor system resources during heavy usage

## Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Code Standards
- Follow PEP 8 for Python code
- Add docstrings to all functions
- Include type hints where possible
- Write comprehensive tests

### Adding New Features
1. Create feature in appropriate module in `src/`
2. Update configuration if needed
3. Add to Streamlit interface in `app.py`
4. Update documentation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **LangChain** - Framework for LLM applications
- **Streamlit** - Web application framework
- **LM Studio** - Local AI model server
- **Chroma** - Vector database
- **PyVis** - Network visualization

---

**Built with ❤️ for legal professionals and researchers**

*Last updated: September 22, 2025*
```bash
# Make sure you're in the project directory
cd "c:\FILES_PIKA\Research Engine For Courts"

# Activate environment
conda activate legal

# Run the application
streamlit run app.py --server.headless true --server.port 8501
```

### Access the Application
- **Open your browser**
- **Navigate to**: `http://localhost:8501`
- **The application will load with multiple tabs**

## 📖 Usage

### Main Interface Tabs

#### 1. 💬 **Chat** - Interactive Legal Research
- Ask questions about your legal documents
- Get AI-powered answers with citations
- Explore document relationships

#### 2. 📊 **Citation Graph** - Visual Citation Network
- Interactive visualization of legal citations
- Explore connections between cases
- Network analysis of legal precedents

#### 3. 📈 **Citation Analysis** - Document Analytics
- Frequency analysis of citations
- Document usage statistics
- Citation patterns and trends

#### 4. 📋 **Case Brief Generator** - Automated Briefs
- Generate structured case briefs
- Extract key legal elements
- Professional legal document formatting

#### 5. ⚖️ **Precedent Analysis** - Case Similarity
- Find similar legal cases
- Analyze legal principles and holdings
- Precedent research and validation

#### 6. 📄 **Document Comparison** - Side-by-Side Analysis
- Compare multiple legal documents
- Identify similarities and differences
- Legal implications analysis

#### 7. 📊 **Evaluation** - System Performance
- DeepEval metrics and benchmarks
- System accuracy assessment
- Performance optimization insights

### Example Usage

#### Basic Q&A
```
Question: "What are the key principles of Section 138 of the Negotiable Instruments Act?"

Response: The system will search your documents and provide:
- Relevant case law
- Section references
- Legal analysis with citations
```

#### Document Upload
```python
# Add new documents
# 1. Place PDF in data/ folder
# 2. Run: python src/ingest.py
# 3. New documents are automatically indexed
```

## 📁 Project Structure

```
Research Engine For Courts/
├── app.py                          # Main Streamlit application
├── src/
│   ├── __init__.py
│   ├── app.py                     # Alternative app interface
│   ├── config.py                  # Configuration settings
│   ├── retrieval.py               # RAG implementation
│   ├── ingest.py                  # Data ingestion pipeline
│   ├── enhanced_document_processor.py  # Document processing
│   ├── precedent_analyzer.py      # Legal precedent analysis
│   ├── document_comparator.py     # Document comparison
│   ├── case_brief_generator.py    # Case brief generation
│   ├── knowledge_graph.py         # Citation graph visualization
│   └── evaluation.py              # System evaluation
├── data/                          # Legal document storage
│   ├── Commercial Courts Act, 2015.pdf
│   ├── Commercial Courts Rules, 2019.pdf
│   └── ...
├── chroma/                        # Vector database storage
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── pyrightconfig.json             # Python type checking
```

## ⚙️ Configuration

### Model Configuration (`src/config.py`)
```python
# LM Studio models (local)
EMBEDDING_MODEL = "text-embedding-embeddinggemma-300m-qat"
LLM_MODEL = "local-model"  # LM Studio will use loaded model

# Paths
DATA_PATH = "data/"
CHROMA_PATH = "chroma/"
```

### Environment Variables
Create a `.env` file for sensitive configurations:
```bash
# .env file
GOOGLE_API_KEY=your_key_here  # Only if using Google models (not recommended)
```

## 🔧 Troubleshooting

### Common Issues

#### 1. **Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### 2. **LM Studio Connection Issues**
- **Check if LM Studio is running**
- **Verify server URL**: `http://127.0.0.1:1234`
- **Ensure models are loaded in LM Studio**
- **Check firewall settings**

#### 3. **Vector Store Errors**
```bash
# Reset vector store
rm -rf chroma/
python src/ingest.py
```

#### 4. **Memory Issues**
- **Increase RAM allocation in LM Studio**
- **Use smaller models**
- **Process documents in smaller batches**

#### 5. **Port Conflicts**
```bash
# Use different port
streamlit run app.py --server.port 8502
```

### Performance Optimization

#### For Better Speed:
1. **Use GPU in LM Studio** (if available)
2. **Optimize model parameters**
3. **Use smaller embedding models**
4. **Increase batch sizes in ingestion**

#### For Better Accuracy:
1. **Use larger models**
2. **Fine-tune prompts**
3. **Add more legal documents**
4. **Improve document preprocessing**

## 🤝 Contributing

### Development Setup
1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Test thoroughly**
5. **Submit a pull request**

### Code Standards
- **Follow PEP 8** for Python code
- **Add docstrings** to all functions
- **Include type hints** where possible
- **Write comprehensive tests**

### Adding New Features
1. **Create feature in appropriate module**
2. **Update configuration if needed**
3. **Add to Streamlit interface**
4. **Update documentation**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain** - Framework for LLM applications
- **Streamlit** - Web application framework
- **LM Studio** - Local AI model server
- **Chroma** - Vector database
- **PyVis** - Network visualization

## 📞 Support

For support and questions:
- **Open an issue** on GitHub
- **Check the troubleshooting section**
- **Review LM Studio documentation**

---

**Built with ❤️ for legal professionals and researchers**

*Last updated: September 16, 2025*
