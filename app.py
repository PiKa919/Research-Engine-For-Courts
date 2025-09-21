import streamlit as st
import os
import re
import networkx as nx
from pyvis.network import Network
from langchain_openai import ChatOpenAI
import collections
from src.ingest import ingest_data
from src.retrieval import create_rag_chain  # RAG chain
from src.config import DATA_PATH
from langchain_community.document_loaders import PyPDFDirectoryLoader
from src.evaluation import run_sample_evaluation
from src.knowledge_graph import create_knowledge_graph, visualize_knowledge_graph
from src.case_brief_generator import CaseBriefGenerator
from src.precedent_analyzer import PrecedentAnalyzer, LegalIssueExtractor
from src.document_comparator import DocumentComparator, CaseTimelineBuilder
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src import config as cfg
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Note: This app uses local LM Studio models for all processing

# --- App Configuration ---
st.set_page_config(page_title="Legal Research Engine", layout="wide")
st.title("Legal Research Engine")
st.write("This app allows you to chat with your legal documents using local models from LM Studio.")

# --- Functions ---
@st.cache_data
def load_documents():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    return documents

def load_vectorstore():
    """
    Load the Chroma vector store from the configured path.
    Returns None if the vector store doesn't exist or can't be loaded.
    """
    try:
        if not os.path.exists(cfg.CHROMA_PATH):
            st.warning(f"Vector store not found at {cfg.CHROMA_PATH}. Please run data ingestion first.")
            return None
            
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(
            model=cfg.EMBEDDING_MODEL,
            openai_api_base="http://127.0.0.1:1234/v1",
            openai_api_key="lm-studio",
            check_embedding_ctx_length=False
        )
        vectorstore = Chroma(
            persist_directory=cfg.CHROMA_PATH,
            embedding_function=embeddings,
            collection_metadata=cfg.CHROMA_CONFIG
        )
        return vectorstore
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return None

def extract_citations(text):
    """
    Improved citation extraction for Indian legal documents
    """
    citations = []
    
    # Clean the text - remove extra spaces and normalize
    text = re.sub(r'\s+', ' ', text)
    
    # 1. Supreme Court citations: 2019 SCC 123, (2019) 1 SCC 123
    sc_pattern = r'(?:\(?\d{4}\)?\s*\d*\s*SCC\s+\d+|\d{4}\s+SCC\s+\d+)'
    sc_citations = re.findall(sc_pattern, text, re.IGNORECASE)
    citations.extend([f"SCC: {cite.strip()}" for cite in sc_citations])
    
    # 2. High Court citations: 2019 1 Mad LJ 123, (2019) 1 Bom CR 123
    hc_pattern = r'(?:\(?\d{4}\)?\s*\d*\s*(?:Mad|Del|Bom|Cal|All|Ker|Kar|AP|Tel|Raj|MP|Guj|Ori|Pat|Gau|P&H)\s*(?:LJ|HC|CR|LW)\s+\d+)'
    hc_citations = re.findall(hc_pattern, text, re.IGNORECASE)
    citations.extend([f"HC: {cite.strip()}" for cite in hc_citations])
    
    # 3. All India Reporter: AIR 2019 SC 123, AIR 2019 Mad 123
    air_pattern = r'AIR\s+\d{4}\s+(?:SC|Mad|Del|Bom|Cal|All|Ker|Kar|AP|Tel|Raj|MP|Guj|Ori|Pat|Gau|P&H)\s+\d+'
    air_citations = re.findall(air_pattern, text, re.IGNORECASE)
    citations.extend([f"AIR: {cite.strip()}" for cite in air_citations])
    
    # 4. Constitutional provisions: Article 14, Article 21
    article_pattern = r'Article\s+\d+(?:\s*\([^)]*\))?(?:\s+of\s+(?:the\s+)?Constitution)?'
    article_citations = re.findall(article_pattern, text, re.IGNORECASE)
    citations.extend([cite.strip() for cite in article_citations])
    
    # 5. Specific Acts with years - more precise pattern
    act_pattern = r'(?:The\s+)?[A-Z][a-zA-Z\s&]+Act,?\s+\d{4}'
    act_citations = re.findall(act_pattern, text)
    # Filter out common false positives
    act_citations = [act for act in act_citations if len(act.split()) >= 3 and 'Section' not in act and 'Rule' not in act]
    citations.extend(act_citations)
    
    # 6. Criminal citations: Cr.L.J, Crl.L.J
    criminal_pattern = r'(?:\d{4}\s+)?Cr\.?L\.?J\.?\s+\d+'
    criminal_citations = re.findall(criminal_pattern, text, re.IGNORECASE)
    citations.extend([f"Criminal: {cite.strip()}" for cite in criminal_citations])
    
    # 7. Section references with context (only if preceded by act name)
    section_pattern = r'(?:(?:under\s+)?Section\s+\d+(?:\([^)]*\))*(?:\s+(?:of|to)\s+(?:the\s+)?[A-Z][a-zA-Z\s&]+Act,?\s+\d{4})?)'
    section_citations = re.findall(section_pattern, text, re.IGNORECASE)
    # Only include if they reference a specific act
    section_citations = [sec for sec in section_citations if 'Act' in sec]
    citations.extend(section_citations)
    
    # Remove duplicates and clean
    unique_citations = list(set([cite.strip() for cite in citations if len(cite.strip()) > 5]))
    
    # Additional filtering to remove obvious false positives
    filtered_citations = []
    for cite in unique_citations:
        # Skip if it's just numbers and common words
        if not re.match(r'^\d+\s+\w+\s+\d+$', cite):
            # Skip standalone section/rule numbers without context
            if not re.match(r'^(?:Section|Rule|Order)\s+\d+$', cite, re.IGNORECASE):
                filtered_citations.append(cite)
    
    return filtered_citations

def build_citation_graph(documents, min_citations=2):
    """
    Build a more intelligent citation graph with filtering
    """
    graph = nx.DiGraph()
    
    # Count citation frequency across all documents
    citation_count = collections.Counter()
    doc_citations = {}
    
    # First pass: collect all citations and count them
    for doc in documents:
        doc_name = os.path.basename(doc.metadata['source'])
        citations = extract_citations(doc.page_content)
        doc_citations[doc_name] = citations
        citation_count.update(citations)
    
    # Filter citations that appear at least min_citations times or are legal authorities
    important_citations = set()
    for citation, count in citation_count.items():
        if (count >= min_citations or 
            any(keyword in citation.upper() for keyword in ['SCC', 'AIR', 'ARTICLE', 'ACT,'])):
            important_citations.add(citation)
    
    # Add nodes for documents and important citations
    for doc_name in doc_citations.keys():
        graph.add_node(doc_name, type='document', size=20)
    
    for citation in important_citations:
        graph.add_node(citation, type='citation', 
                      frequency=citation_count[citation], size=10)
    
    # Add edges between documents and their citations
    for doc_name, citations in doc_citations.items():
        for citation in citations:
            if citation in important_citations:
                graph.add_edge(doc_name, citation, weight=1)
    
    return graph

def extract_citations(text):
    """
    Improved citation extraction for Indian legal documents
    """
    citations = []
    
    # Clean the text - remove extra spaces and normalize
    text = re.sub(r'\s+', ' ', text)
    
    # 1. Supreme Court citations: 2019 SCC 123, (2019) 1 SCC 123
    sc_pattern = r'(?:\(?\d{4}\)?\s*\d*\s*SCC\s+\d+|\d{4}\s+SCC\s+\d+)'
    sc_citations = re.findall(sc_pattern, text, re.IGNORECASE)
    citations.extend([f"SCC: {cite.strip()}" for cite in sc_citations])
    
    # 2. High Court citations: 2019 1 Mad LJ 123, (2019) 1 Bom CR 123
    hc_pattern = r'(?:\(?\d{4}\)?\s*\d*\s*(?:Mad|Del|Bom|Cal|All|Ker|Kar|AP|Tel|Raj|MP|Guj|Ori|Pat|Gau|P&H)\s*(?:LJ|HC|CR|LW)\s+\d+)'
    hc_citations = re.findall(hc_pattern, text, re.IGNORECASE)
    citations.extend([f"HC: {cite.strip()}" for cite in hc_citations])
    
    # 3. All India Reporter: AIR 2019 SC 123, AIR 2019 Mad 123
    air_pattern = r'AIR\s+\d{4}\s+(?:SC|Mad|Del|Bom|Cal|All|Ker|Kar|AP|Tel|Raj|MP|Guj|Ori|Pat|Gau|P&H)\s+\d+'
    air_citations = re.findall(air_pattern, text, re.IGNORECASE)
    citations.extend([f"AIR: {cite.strip()}" for cite in air_citations])
    
    # 4. Constitutional provisions: Article 14, Article 21
    article_pattern = r'Article\s+\d+(?:\s*\([^)]*\))?(?:\s+of\s+(?:the\s+)?Constitution)?'
    article_citations = re.findall(article_pattern, text, re.IGNORECASE)
    citations.extend([cite.strip() for cite in article_citations])
    
    # 5. Specific Acts with years - more precise pattern
    act_pattern = r'(?:The\s+)?[A-Z][a-zA-Z\s&]+Act,?\s+\d{4}'
    act_citations = re.findall(act_pattern, text)
    # Filter out common false positives
    act_citations = [act for act in act_citations if len(act.split()) >= 3 and 'Section' not in act and 'Rule' not in act]
    citations.extend(act_citations)
    
    # 6. Criminal citations: Cr.L.J, Crl.L.J
    criminal_pattern = r'(?:\d{4}\s+)?Cr\.?L\.?J\.?\s+\d+'
    criminal_citations = re.findall(criminal_pattern, text, re.IGNORECASE)
    citations.extend([f"Criminal: {cite.strip()}" for cite in criminal_citations])
    
    # 7. Section references with context (only if preceded by act name)
    section_pattern = r'(?:(?:under\s+)?Section\s+\d+(?:\([^)]*\))*(?:\s+(?:of|to)\s+(?:the\s+)?[A-Z][a-zA-Z\s&]+Act,?\s+\d{4})?)'
    section_citations = re.findall(section_pattern, text, re.IGNORECASE)
    # Only include if they reference a specific act
    section_citations = [sec for sec in section_citations if 'Act' in sec]
    citations.extend(section_citations)
    
    # Remove duplicates and clean
    unique_citations = list(set([cite.strip() for cite in citations if len(cite.strip()) > 5]))
    
    # Additional filtering to remove obvious false positives
    filtered_citations = []
    for cite in unique_citations:
        # Skip if it's just numbers and common words
        if not re.match(r'^\d+\s+\w+\s+\d+$', cite):
            # Skip standalone section/rule numbers without context
            if not re.match(r'^(?:Section|Rule|Order)\s+\d+$', cite, re.IGNORECASE):
                filtered_citations.append(cite)
    
    return filtered_citations

def build_citation_graph(documents, min_citations=2):
    """
    Build a more intelligent citation graph with filtering
    """
    graph = nx.DiGraph()
    
    # Count citation frequency across all documents
    citation_count = collections.Counter()
    doc_citations = {}
    
    # First pass: collect all citations and count them
    for doc in documents:
        doc_name = os.path.basename(doc.metadata['source'])
        citations = extract_citations(doc.page_content)
        doc_citations[doc_name] = citations
        citation_count.update(citations)
    
    # Filter citations that appear at least min_citations times or are legal authorities
    important_citations = set()
    for citation, count in citation_count.items():
        if (count >= min_citations or 
            any(keyword in citation.upper() for keyword in ['SCC', 'AIR', 'ARTICLE', 'ACT,'])):
            important_citations.add(citation)
    
    # Second pass: build graph with filtered citations
    for doc_name, citations in doc_citations.items():
        graph.add_node(doc_name, type='document', size=20)
        
        for citation in citations:
            if citation in important_citations:
                if not graph.has_node(citation):
                    # Determine citation type for coloring
                    if any(keyword in citation.upper() for keyword in ['SCC', 'AIR']):
                        cite_type = 'case_law'
                    elif 'ARTICLE' in citation.upper():
                        cite_type = 'constitutional'
                    elif 'ACT' in citation.upper():
                        cite_type = 'statute'
                    else:
                        cite_type = 'other'
                    
                    graph.add_node(citation, type='citation', cite_type=cite_type, 
                                 frequency=citation_count[citation], size=10)
                
                graph.add_edge(doc_name, citation, weight=1)
    
    return graph

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")

    # Initialize LLM using local LM Studio
    llm = None
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="local-model",  # LM Studio model name (will be overridden by LM Studio)
            openai_api_base="http://127.0.0.1:1234/v1",  # LM Studio local endpoint
            openai_api_key="lm-studio",  # Dummy key for local LM Studio
            temperature=0.1,  # Lower temperature for consistent legal responses
            max_tokens=2048  # Reasonable token limit for responses
        )
        st.success("✅ LM Studio connection established successfully")
    except Exception as e:
        st.error(f"Failed to initialize LM Studio: {e}")
        st.info("Make sure LM Studio is running locally on port 1234 with a chat model loaded.")

    # Data Ingestion
    if st.button("Ingest Data"):
        with st.spinner("Ingesting data..."):
            ingestion_status = ingest_data()
            st.success(ingestion_status)

    # Citation Graph Configuration
    st.subheader("Citation Graph Settings")
    min_citations = st.slider("Minimum citation frequency", 1, 5, 2,
                             help="Only show citations that appear in at least this many documents")
    show_isolated = st.checkbox("Show isolated citations", False,
                               help="Show citations that appear in only one document")

# --- Main App ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Chat", 
    "Citation Graph", 
    "Citation Analysis", 
    "Case Brief Generator", 
    "Precedent Analysis",
    "Document Comparison",
    "Evaluation"
])

with tab1:
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your documents"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if llm:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    async_rag_chain, sync_rag_chain = create_rag_chain()
                    response = sync_rag_chain(prompt)
                    answer = response["answer"]
                    st.markdown(answer)

                    # Display sources
                    if 'context' in response and response['context']:
                        st.write("\n---\n**Sources:**")
                        for doc in response['context']:
                            source = doc.metadata.get('source', 'Unknown')
                            page = doc.metadata.get('page', 'N/A')
                            st.write(f"- **Source:** {os.path.basename(source)}, **Page:** {page}")
                            with st.expander("Show content"):
                                st.write(doc.page_content)

            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            st.info("Please select a model and provide the necessary configuration in the sidebar.")

with tab2:
    st.header("Citation Graph")
    if st.button("Generate Citation Graph"):
        with st.spinner("Building graph..."):
            documents = load_documents()
            
            if documents:
                graph = build_citation_graph(documents, min_citations if not show_isolated else 1)
                
                if graph.nodes:
                    # Create network with simpler configuration
                    net = Network(height="800px", width="100%", bgcolor="#222222", 
                                font_color="white", directed=True)
                    
                    # Add nodes with different colors based on type
                    for node, attrs in graph.nodes(data=True):
                        if attrs.get('type') == 'document':
                            net.add_node(str(node), label=str(node), color="#00bfff", 
                                       size=25, title=f"Document: {node}")
                        else:
                            cite_type = attrs.get('cite_type', 'other')
                            frequency = attrs.get('frequency', 1)
                            
                            # Color based on citation type
                            color_map = {
                                'case_law': '#ff4500',      # Orange for case law
                                'constitutional': '#32cd32',  # Green for constitutional
                                'statute': '#ff69b4',       # Pink for statutes
                                'other': '#ffd700'          # Gold for others
                            }
                            
                            color = color_map.get(cite_type, '#ffd700')
                            size = min(15 + frequency * 3, 30)  # Size based on frequency
                            
                            net.add_node(str(node), label=str(node), color=color, 
                                       size=size, title=f"Citation: {node}\nFrequency: {frequency}")
                    
                    # Add edges
                    for u, v, attrs in graph.edges(data=True):
                        net.add_edge(str(u), str(v))

                    # Generate and display
                    html_file = "citation_graph.html"
                    try:
                        # Try the new method first
                        try:
                            html_content = net.generate_html()
                        except AttributeError:
                            # Fallback for older versions of pyvis
                            net.save_graph(html_file)
                            with open(html_file, 'r', encoding='utf-8') as f:
                                html_content = f.read()
                        
                        # Write to file
                        with open(html_file, 'w', encoding='utf-8') as f:
                            f.write(html_content)
                        
                        # Read and display
                        with open(html_file, 'r', encoding='utf-8') as f:
                            html_data = f.read()
                        st.components.v1.html(html_data, height=800, scrolling=True)
                        
                        # Add legend
                        st.markdown("""
                        **Legend:**
                        - 🔵 Blue: Documents
                        - 🟠 Orange: Case Law (SCC, AIR)
                        - 🟢 Green: Constitutional Provisions
                        - 🩷 Pink: Statutes/Acts
                        - 🟡 Gold: Other Citations
                        
                        *Node size indicates citation frequency*
                        """)
                        
                    except Exception as e:
                        st.error(f"Failed to generate graph: {e}")
                else:
                    st.warning("No citations found that meet the frequency threshold.")
            else:
                st.warning("No documents found.")

with tab3:
    st.header("Citation Analysis")
    if st.button("Analyze Citations"):
        documents = load_documents()
        if documents:
            all_citations = []
            citation_by_doc = {}
            
            for doc in documents:
                doc_name = os.path.basename(doc.metadata['source'])
                citations = extract_citations(doc.page_content)
                citation_by_doc[doc_name] = citations
                all_citations.extend(citations)
            
            if all_citations:
                # Citation frequency analysis
                citation_freq = collections.Counter(all_citations)
                
                st.subheader("Most Frequently Cited")
                freq_df = st.dataframe(
                    {
                        'Citation': list(citation_freq.keys())[:20],
                        'Frequency': list(citation_freq.values())[:20]
                    }
                )
                
                st.subheader("Citations by Document")
                for doc_name, citations in citation_by_doc.items():
                    if citations:
                        st.write(f"**{doc_name}:**")
                        for cite in citations[:10]:  # Show top 10
                            st.write(f"  • {cite}")
                        if len(citations) > 10:
                            st.write(f"  ... and {len(citations) - 10} more")
                    else:
                        st.write(f"**{doc_name}:** No citations found")
                
            else:
                st.warning("No citations found in the documents.")
        else:
            st.warning("No documents found.")

with tab4:
    st.header("📋 Case Brief Generator")
    st.write("Generate structured case briefs from your legal documents automatically.")

    # Configuration
    col1, col2 = st.columns(2)
    with col1:
        brief_format = st.selectbox(
            "Brief Format",
            ["Standard", "Detailed", "Summary"],
            help="Choose the level of detail for the case brief"
        )
    with col2:
        max_docs = st.slider("Max Documents to Process", 1, 10, 3,
                           help="Limit the number of documents to process for performance")

    # Document selection
    vectorstore = load_vectorstore()
    if vectorstore:
        documents = vectorstore.get()["documents"]
        metadatas = vectorstore.get()["metadatas"]

        if documents:
            # Show available documents
            doc_options = [f"{os.path.basename(meta.get('source', f'Doc {i+1}'))}"
                          for i, meta in enumerate(metadatas)]
            selected_docs = st.multiselect(
                "Select documents to generate briefs for:",
                options=list(range(len(documents))),
                format_func=lambda i: doc_options[i],
                max_selections=max_docs
            )

            if st.button("Generate Case Briefs") and selected_docs:
                with st.spinner("Generating case briefs..."):
                    try:
                        generator = CaseBriefGenerator()

                        for idx in selected_docs:
                            doc = Document(
                                page_content=documents[idx],
                                metadata=metadatas[idx]
                            )

                            brief = generator.generate_case_brief(doc)
                            formatted_brief = generator.format_brief_for_display(brief)

                            st.markdown("---")
                            st.markdown(formatted_brief)

                            # Download option
                            brief_text = formatted_brief.replace("#", "").replace("*", "")
                            st.download_button(
                                label=f"📥 Download Brief - {brief.case_name[:50]}...",
                                data=brief_text,
                                file_name=f"case_brief_{idx+1}.txt",
                                mime="text/plain"
                            )

                    except Exception as e:
                        st.error(f"Error generating case briefs: {e}")
                        st.info("Make sure you have set your GOOGLE_API_KEY in the .env file.")
        else:
            st.warning("No documents found in the vector store. Please ingest some documents first.")
    else:
        st.error("Vector store could not be loaded.")

with tab5:
    st.header("⚖️ Precedent Analysis")
    st.write("Find similar cases and analyze legal precedents to support your judicial decision-making.")

    # Query input
    query = st.text_area(
        "Enter case description or legal issue:",
        height=100,
        placeholder="Describe the case facts, legal issues, or key elements you want to find precedents for..."
    )

    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Number of similar cases", 3, 10, 5,
                         help="How many similar cases to retrieve")
    with col2:
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Precedent Analysis", "Legal Issue Extraction", "Both"],
            help="Choose the type of analysis to perform"
        )

    if st.button("Analyze Precedents") and query:
        with st.spinner("Searching for similar cases and analyzing precedents..."):
            try:
                analyzer = PrecedentAnalyzer(cfg.CHROMA_PATH)

                # Find similar cases
                similar_cases = analyzer.find_similar_cases(query, top_k=top_k)

                if similar_cases:
                    st.success(f"Found {len(similar_cases)} similar cases")

                    # Display similar cases
                    st.subheader("📚 Similar Cases Found")
                    for i, (doc, score) in enumerate(similar_cases, 1):
                        with st.expander(f"Case {i}: {os.path.basename(doc.metadata.get('source', 'Unknown'))}"):
                            st.write(f"**Relevance Score:** {score:.3f}")
                            st.write(f"**Content Preview:** {doc.page_content[:500]}...")
                            if st.button(f"📋 Generate Brief for Case {i}", key=f"brief_{i}"):
                                generator = CaseBriefGenerator()
                                brief = generator.generate_case_brief(doc)
                                formatted_brief = generator.format_brief_for_display(brief)
                                st.markdown(formatted_brief)

                    # Perform precedent analysis
                    if len(similar_cases) >= 2:
                        current_case = Document(page_content=query, metadata={"source": "User Query"})
                        similar_docs = [doc for doc, _ in similar_cases]

                        analysis = analyzer.analyze_precedents(current_case, similar_docs)
                        report = analyzer.generate_precedent_report(current_case, analysis)

                        st.subheader("📊 Precedent Analysis Report")
                        st.markdown(report)

                    # Legal Issue Extraction
                    if analysis_type in ["Legal Issue Extraction", "Both"]:
                        st.subheader("🔍 Legal Issues Extracted")
                        extractor = LegalIssueExtractor()

                        for i, (doc, _) in enumerate(similar_cases[:3], 1):  # Limit to top 3
                            issues = extractor.extract_legal_issues(doc)
                            with st.expander(f"Legal Issues - Case {i}"):
                                for category, items in issues.items():
                                    if items:
                                        st.write(f"**{category.replace('_', ' ').title()}:**")
                                        for item in items:
                                            st.write(f"• {item}")
                                        st.write("")

                else:
                    st.warning("No similar cases found. Try rephrasing your query or check if documents are properly ingested.")

            except Exception as e:
                st.error(f"Error during precedent analysis: {e}")
                st.info("Make sure you have set your GOOGLE_API_KEY in the .env file and have ingested documents.")

with tab6:
    st.header("📊 Document Comparison")
    st.write("Compare multiple case documents to identify similarities, differences, and legal implications.")

    vectorstore = load_vectorstore()
    if vectorstore:
        documents = vectorstore.get()["documents"]
        metadatas = vectorstore.get()["metadatas"]

        if documents:
            # Document selection for comparison
            doc_options = [f"{os.path.basename(meta.get('source', f'Doc {i+1}'))}"
                          for i, meta in enumerate(metadatas)]

            selected_indices = st.multiselect(
                "Select documents to compare (2-4 recommended):",
                options=list(range(len(documents))),
                format_func=lambda i: doc_options[i],
                max_selections=4
            )

            if len(selected_indices) >= 2 and st.button("Compare Documents"):
                with st.spinner("Analyzing documents for similarities and differences..."):
                    try:
                        comparator = DocumentComparator()

                        # Prepare documents for comparison
                        selected_docs = [
                            Document(page_content=documents[i], metadata=metadatas[i])
                            for i in selected_indices
                        ]

                        # Perform comparison
                        comparison = comparator.compare_documents(selected_docs)

                        # Display results
                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("✅ Similarities")
                            for similarity in comparison.similarities:
                                st.write(f"• {similarity}")

                        with col2:
                            st.subheader("❌ Differences")
                            for difference in comparison.differences:
                                st.write(f"• {difference}")

                        st.subheader("⚖️ Legal Implications")
                        for implication in comparison.legal_implications:
                            st.write(f"• {implication}")

                        if comparison.precedential_conflicts:
                            st.subheader("⚠️ Precedential Conflicts")
                            for conflict in comparison.precedential_conflicts:
                                st.warning(f"• {conflict}")

                        st.subheader("💡 Recommendation")
                        st.info(comparison.recommendation)

                        # Timeline analysis
                        st.subheader("📅 Case Timeline")
                        timeline_builder = CaseTimelineBuilder()
                        timeline = timeline_builder.build_timeline(selected_docs)
                        timeline_report = timeline_builder.generate_timeline_report(timeline)
                        st.markdown(timeline_report)

                        # Common legal terms
                        st.subheader("🏷️ Common Legal Terms")
                        common_terms = comparator.find_common_legal_terms(selected_docs)
                        for category, terms in common_terms.items():
                            with st.expander(f"{category.title()} Law Terms"):
                                for term in terms:
                                    st.write(f"• {term}")

                    except Exception as e:
                        st.error(f"Error comparing documents: {e}")
                        st.info("Make sure you have set your GOOGLE_API_KEY in the .env file.")

            elif len(selected_indices) < 2:
                st.info("Please select at least 2 documents to compare.")
        else:
            st.warning("No documents found. Please ingest some documents first.")
    else:
        st.error("Vector store could not be loaded.")

with tab7:
    st.header("📈 System Evaluation")
    st.write("Evaluate the RAG system's performance using DeepEval metrics.")

    if st.button("Run Sample Evaluation"):
        with st.spinner("Running evaluation..."):
            try:
                results = run_sample_evaluation()

                if "error" in results:
                    st.error(f"Evaluation failed: {results['error']}")
                else:
                    st.success("Evaluation completed!")

                    st.subheader("Results Summary")
                    st.write(f"Total test cases: {results['total_test_cases']}")

                    st.subheader("Metrics")
                    for metric_name, metric_data in results['metrics_results'].items():
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**{metric_name}**")
                        with col2:
                            if metric_data['score'] is not None:
                                st.write(f"Score: {metric_data['score']:.3f}")
                            else:
                                st.write("Score: N/A")

                        if metric_data['reason']:
                            st.write(f"Reason: {metric_data['reason']}")

            except Exception as e:
                st.error(f"Error running evaluation: {e}")
                st.info("Make sure DeepEval is installed: pip install deepeval")