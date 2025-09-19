# 🏛️ Legal Research Engine for Courts

A comprehensive AI-powered legal research assistant built with LangChain, Streamlit, and local LM Studio models. This system provides intelligent document analysis, case law research, precedent analysis, and citation management for Indian legal documents.

## 📋 Table of Contents

- [Features](#-features)
- [Project History](#-project-history)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Setup](#-setup)
- [Running the Application](#-running-the-application)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

### 🔍 **Intelligent Document Processing**
- **Enhanced PDF Processing**: Advanced text extraction from legal documents
- **Metadata Extraction**: Automatic extraction of sections, court mentions, and legal terms
- **Quality Assessment**: Document quality scoring and validation
- **Adaptive Text Splitting**: Smart chunking optimized for legal content

### 🤖 **AI-Powered Analysis**
- **Local LLM Integration**: Uses LM Studio for complete privacy and cost control
- **Retrieval-Augmented Generation (RAG)**: Context-aware legal research
- **Multi-Model Support**: Separate models for embeddings and chat
- **Legal-Specific Prompts**: Tailored prompts for Indian legal system

### 📊 **Advanced Tools**
- **Citation Graph Visualization**: Interactive network of legal citations
- **Case Brief Generator**: Automated structured case brief creation
- **Precedent Analysis**: Find similar cases and legal principles
- **Document Comparison**: Side-by-side analysis of multiple documents
- **Timeline Builder**: Chronological case progression tracking

### 🎯 **Legal Research Capabilities**
- **Indian Law Focus**: Specialized for Indian legal system and courts
- **Citation Extraction**: Automatic identification of legal references
- **Case Law Research**: Intelligent precedent finding and analysis
- **Document Similarity**: Semantic search across legal documents

## � Project History & Development Journey

### 🎯 **Project Evolution**

This legal research engine has undergone significant evolution to become a robust, production-ready system. Here's the journey of challenges faced and solutions implemented:

### ✅ **Phase 1: Initial Implementation (LangChain RAG Foundation)**
- **✅ Implemented** comprehensive LangChain RAG best practices
- **✅ Built** enhanced document processing pipeline
- **✅ Added** adaptive text splitting for legal documents
- **✅ Integrated** metadata extraction and quality validation
- **✅ Created** modular architecture with separate components

### ⚠️ **Phase 2: API Rate Limiting Challenges**
- **❌ Encountered** Google Gemini API rate limiting issues
- **❌ Faced** token usage restrictions (15 RPM, 250K TPM, 1000 RPD)
- **❌ Experienced** API costs and dependency concerns
- **❌ Dealt with** complex rate limiting wrapper implementations

### 🔄 **Phase 3: Migration to Local Models**
- **✅ Migrated** from Google Gemini to LM Studio local models
- **✅ Implemented** OpenAI-compatible API integration
- **✅ Configured** local embedding model (`text-embedding-embeddinggemma-300m-qat`)
- **✅ Set up** local chat model integration
- **✅ Achieved** complete data privacy and cost control

### 🧹 **Phase 4: Codebase Cleanup & Optimization**
- **✅ Removed** all rate limiting code (300+ lines eliminated)
- **✅ Eliminated** `RateLimitedEmbeddings` wrapper classes
- **✅ Cleaned** all Gemini API references and dependencies
- **✅ Simplified** architecture with direct LM Studio API calls
- **✅ Updated** all modules to use local models exclusively

### 🛠️ **Technical Achievements**

#### **🔧 Architecture Improvements:**
- **Modular Design**: Clean separation of concerns across 10+ modules
- **Error Handling**: Comprehensive error handling and logging
- **Configuration Management**: Centralized config with environment variables
- **Database Integration**: Chroma vector database for efficient storage

#### **📊 Advanced Features Implemented:**
- **Citation Graph**: Interactive network visualization of legal citations
- **Precedent Analysis**: AI-powered case similarity and legal principle extraction
- **Document Comparison**: Side-by-side analysis with legal implications
- **Case Brief Generator**: Automated structured legal document creation
- **Timeline Builder**: Chronological case progression tracking

#### **🎯 Legal Specialization:**
- **Indian Law Focus**: Specialized prompts and processing for Indian legal system
- **Citation Extraction**: Advanced pattern matching for legal references
- **Court Recognition**: Automatic identification of Indian court hierarchies
- **Legal Term Analysis**: Context-aware legal terminology processing

### 📈 **Performance Optimizations**
- **Local Processing**: Zero API latency, complete data privacy
- **Batch Processing**: Efficient document ingestion with progress tracking
- **Memory Management**: Optimized for large legal document collections
- **GPU Support**: LM Studio integration for accelerated processing

### 🔒 **Security & Privacy**
- **Local Models**: No data sent to external APIs
- **Offline Capability**: Works without internet connection
- **Data Sovereignty**: Complete control over sensitive legal data
- **Cost Control**: Zero API costs, predictable resource usage

### 📚 **Lessons Learned**
- **Local vs Cloud**: Local models provide better privacy and cost control
- **Modular Architecture**: Clean separation enables easier maintenance
- **Legal Domain Expertise**: Specialized processing improves accuracy
- **User Experience**: Intuitive interface crucial for legal professionals

### 🎯 **Current Status**
- **✅ Production Ready**: Fully functional legal research system
- **✅ Local Only**: No external API dependencies
- **✅ Comprehensive**: 7 integrated research tools
- **✅ Scalable**: Modular architecture for future enhancements
- **✅ Documented**: Complete setup and usage documentation

### 🚀 **Future Roadmap**
- **Model Fine-tuning**: Domain-specific model training
- **Advanced Analytics**: Deeper legal analytics and insights
- **Multi-language Support**: Support for regional languages
- **Integration APIs**: REST API for external integrations
- **Mobile Interface**: Responsive design for mobile devices

---

**This project demonstrates the successful evolution from a cloud-dependent prototype to a robust, privacy-focused, production-ready legal research platform.**

## �🔧 Prerequisites

### Required Software
- **Python 3.8+**: Programming language runtime
- **LM Studio**: Local AI model server (free download)
- **Git**: Version control system

### System Requirements
- **RAM**: Minimum 8GB, Recommended 16GB+
- **Storage**: 5GB+ free space for models and documents
- **OS**: Windows 10/11, macOS, or Linux

### AI Models (via LM Studio)
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
```

### 3. Install Dependencies
```bash
# Install Python packages
pip install -r requirements.txt
```

### 4. Install LM Studio
1. **Download LM Studio**: Visit [https://lmstudio.ai/](https://lmstudio.ai/)
2. **Install and Launch**: Follow the installation wizard
3. **Download Models**:
   - Search for `text-embedding-embeddinggemma-300m-qat` (embedding model)
   - Download a chat model (e.g., `llama-2-7b-chat` or `mistral-7b-instruct`)

## ⚙️ Setup

### 1. Configure LM Studio
1. **Open LM Studio**
2. **Go to "Developer" tab**
3. **Load Models**:
   - Load your embedding model
   - Load your chat model
4. **Start Local Server**:
   - Click "Start Server"
   - Verify server runs on `http://127.0.0.1:1234`

### 2. Prepare Legal Documents
1. **Create data directory** (if not exists):
   ```bash
   mkdir data
   ```
2. **Add PDF documents** to the `data/` folder:
   - Legal case files
   - Court judgments
   - Legal statutes
   - Research documents

### 3. Ingest Documents
```bash
# Run data ingestion
python src/ingest.py
```
This will:
- Process all PDFs in the `data/` folder
- Create embeddings using LM Studio
- Store vectors in Chroma database

## 🚀 Running the Application

### Start the Streamlit App
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
