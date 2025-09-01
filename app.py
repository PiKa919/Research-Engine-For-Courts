import streamlit as st
import os
import re
import networkx as nx
from pyvis.network import Network
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
import collections

# --- App Configuration ---
st.set_page_config(page_title="Legal Research Engine", layout="wide")
st.title("Legal Research Engine")
st.write("This app allows you to chat with your legal documents using either Gemini or a local model from LM Studio.")

# --- Constants ---
DATA_PATH = "data"
CHROMA_PATH = "chroma"

# --- Functions ---
@st.cache_data
def load_documents():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    return documents

@st.cache_resource
def ingest_data():
    """
    Ingests PDF documents from the data directory, splits them into chunks, 
    creates embeddings, and stores them in a Chroma vector store.
    """
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        return "Created directory: data. Please add your PDF files."

    documents = load_documents()
    
    if not documents:
        return "No documents found in the 'data' directory."

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
    
    return f"Successfully ingested {len(documents)} documents and created a vector store."

@st.cache_resource
def create_rag_chain(_llm):
    """
    Creates a RAG chain with a chosen LLM.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    retriever = vectorstore.as_retriever()

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Use three sentences maximum and keep the answer concise."
        "\n\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])

    question_answer_chain = create_stuff_documents_chain(_llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain

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
    
    # Model Selection
    model_choice = st.radio("Choose your model:", ("Gemini", "LM Studio"))

    llm = None
    if model_choice == "Gemini":
        google_api_key = st.text_input("Enter your Google API key:", type="password", key="google_api_key")
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key
            try:
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
            except Exception as e:
                st.error(f"Failed to initialize Gemini: {e}")
        else:
            st.warning("Please enter your Google API key to use the Gemini model.")
    else:
        st.write("Please ensure you have LM Studio running with the 'gemma3n-1b' model loaded and the server started.")
        try:
            llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
        except Exception as e:
            st.error(f"Failed to connect to LM Studio: {e}")

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
tab1, tab2, tab3 = st.tabs(["Chat", "Citation Graph", "Citation Analysis"])

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
                    rag_chain = create_rag_chain(llm)
                    response = rag_chain.invoke({"input": prompt})
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