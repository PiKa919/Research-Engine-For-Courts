import os
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from langchain.docstore.document import Document
import streamlit as st
import spacy
from typing import List, Optional, Dict, Any, Tuple
import logging
import re
import json
import pickle
from pathlib import Path
import pandas as pd

# Import caching
from .caching import get_cache
from . import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the spacy model with error handling
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("en_core_web_sm not found. Installing...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

class LegalKnowledgeGraph:
    """
    Enhanced Knowledge Graph for Legal Documents

    Features:
    - Legal-specific entity extraction (courts, cases, sections, parties)
    - Citation network analysis
    - Interactive visualization with Pyvis
    - Graph analysis and statistics
    - Relationship mining from legal text
    """

    def __init__(self):
        self.graph = nx.DiGraph()  # Directed graph for legal relationships
        self.cache = get_cache()
        self.cache_key_prefix = "knowledge_graph"
        self.entity_types = {
            'court': {'color': '#FF6B6B', 'shape': 'diamond', 'size': 25},
            'case': {'color': '#4ECDC4', 'shape': 'dot', 'size': 20},
            'section': {'color': '#45B7D1', 'shape': 'square', 'size': 18},
            'party': {'color': '#96CEB4', 'shape': 'triangle', 'size': 15},
            'judge': {'color': '#FFEAA7', 'shape': 'star', 'size': 22},
            'law': {'color': '#DDA0DD', 'shape': 'hexagon', 'size': 20},
            'document': {'color': '#98D8C8', 'shape': 'box', 'size': 16},
            'entity': {'color': '#F7DC6F', 'shape': 'ellipse', 'size': 12}
        }
        self.relationship_types = {
            'cites': {'color': '#FF6B6B', 'width': 2},
            'references': {'color': '#4ECDC4', 'width': 1.5},
            'contains': {'color': '#45B7D1', 'width': 1},
            'related_to': {'color': '#96CEB4', 'width': 1.5}
        }

    def _get_cache_key(self, documents: List[Document]) -> str:
        """Generate cache key based on document metadata"""
        doc_ids = sorted([os.path.basename(doc.metadata.get('source', '')) for doc in documents])
        return f"{self.cache_key_prefix}_{hash(str(doc_ids))}"

    def load_cached_graph(self, cache_key: str) -> Optional[nx.DiGraph]:
        """Load graph from cache if available"""
        try:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                logger.info("Loading knowledge graph from cache")
                return pickle.loads(cached_data)
        except Exception as e:
            logger.warning(f"Error loading cached graph: {e}")
        return None

    def save_graph_cache(self, cache_key: str, graph: nx.DiGraph) -> None:
        """Save graph to cache"""
        try:
            graph_data = pickle.dumps(graph)
            self.cache.set(cache_key, graph_data, ttl=config.CACHE_CONFIG["ttl_seconds"])
            logger.info("Knowledge graph cached successfully")
        except Exception as e:
            logger.warning(f"Error caching graph: {e}")

    def update_graph_incremental(self, new_documents: List[Document]) -> None:
        """Update existing graph with new documents incrementally"""
        logger.info(f"Updating knowledge graph with {len(new_documents)} new documents")

        # Extract entities from new documents only
        new_entities = []
        for doc in new_documents:
            if isinstance(doc, Document):
                source = doc.metadata.get('source', 'Unknown')
                entities = self.extract_legal_entities(doc.page_content, source)
                new_entities.extend(entities)

        # Add new entities to graph
        for entity in new_entities:
            entity_type = entity['type']
            node_props = self.entity_types.get(entity_type, self.entity_types['entity'])

            if entity['id'] not in self.graph.nodes:
                self.graph.add_node(entity['id'],
                                  type=entity_type,
                                  label=entity['name'],
                                  title=f"{entity_type.title()}: {entity['name']}\nSource: {entity['source']}",
                                  color=node_props['color'],
                                  shape=node_props['shape'],
                                  size=node_props['size'])

                # Connect entity to its source document
                doc_id = f"doc_{os.path.basename(entity['source'])}"
                if doc_id not in self.graph.nodes:
                    self.graph.add_node(doc_id,
                                      type='document',
                                      label=os.path.basename(entity['source']),
                                      title=f"Source: {entity['source']}",
                                      color=self.entity_types['document']['color'],
                                      shape=self.entity_types['document']['shape'],
                                      size=self.entity_types['document']['size'])

                if doc_id in self.graph.nodes:
                    self.graph.add_edge(doc_id, entity['id'],
                                      label='contains',
                                      color=self.relationship_types['contains']['color'],
                                      width=self.relationship_types['contains']['width'],
                                      title=f"{os.path.basename(entity['source'])} contains {entity['name']}")

        # Create relationships between new entities and existing entities
        self._create_incremental_relationships(new_entities)

        logger.info(f"Incremental update completed. Graph now has {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")

        self.relationship_types = {
            'cites': {'color': '#FF6B6B', 'width': 2},
            'references': {'color': '#4ECDC4', 'width': 1},
            'overruled_by': {'color': '#FF4757', 'width': 3, 'dashes': True},
            'distinguishes': {'color': '#FFA726', 'width': 2, 'dashes': [5, 5]},
            'follows': {'color': '#66BB6A', 'width': 2},
            'interprets': {'color': '#42A5F5', 'width': 2},
            'applies': {'color': '#AB47BC', 'width': 1},
            'contains': {'color': '#26A69A', 'width': 1}
        }

    def extract_legal_entities(self, text: str, source: str = "") -> List[Dict[str, Any]]:
        """
        Extract legal-specific entities from text

        Args:
            text: Input text to analyze
            source: Source document identifier

        Returns:
            List of extracted entities with metadata
        """
        entities = []

        # Extract courts
        courts = self._extract_courts(text)
        for court in courts:
            entities.append({
                'id': f"court_{court.replace(' ', '_').lower()}",
                'type': 'court',
                'name': court,
                'text': court,
                'source': source
            })

        # Extract case citations
        cases = self._extract_case_citations(text)
        for case in cases:
            entities.append({
                'id': f"case_{case.replace(' ', '_').replace('(', '').replace(')', '').lower()}",
                'type': 'case',
                'name': case,
                'text': case,
                'source': source
            })

        # Extract legal sections
        sections = self._extract_sections(text)
        for section in sections:
            entities.append({
                'id': f"section_{section.replace(' ', '_').lower()}",
                'type': 'section',
                'name': section,
                'text': section,
                'source': source
            })

        # Extract parties (basic implementation)
        parties = self._extract_parties(text)
        for party in parties:
            entities.append({
                'id': f"party_{party.replace(' ', '_').lower()}",
                'type': 'party',
                'name': party,
                'text': party,
                'source': source
            })

        # Extract general named entities using spaCy
        spacy_entities = self._extract_spacy_entities(text, source)
        entities.extend(spacy_entities)

        return entities

    def _extract_courts(self, text: str) -> List[str]:
        """Extract court names from text"""
        courts = []
        court_patterns = [
            r'Supreme Court of India',
            r'High Court of ([A-Za-z\s]+)',
            r'([A-Za-z\s]+) High Court',
            r'District Court',
            r'Tribunal',
            r'Court of Appeal'
        ]

        for pattern in court_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            courts.extend(matches)

        return list(set(courts))  # Remove duplicates

    def _extract_case_citations(self, text: str) -> List[str]:
        """Extract case citations from text"""
        citations = []

        # Indian legal citation patterns
        patterns = [
            r'\(\d{4}\)\s*\d+\s*SCC\s+\d+',  # (2023) 1 SCC 123
            r'\d{4}\s+SCC\s+\d+',            # 2023 SCC 123
            r'\(\d{4}\)\s*\d+\s*[A-Z]+\s+\d+',  # (2023) 1 SC 123
            r'AIR\s+\d{4}\s+[A-Z]+\s+\d+',   # AIR 2023 SC 123
            r'\d{4}\s*\(\d+\)\s*[A-Z]+\s*\(\d+\)',  # 2023 (1) SC (123)
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            citations.extend(matches)

        return list(set(citations))  # Remove duplicates

    def _extract_sections(self, text: str) -> List[str]:
        """Extract legal section references from text"""
        sections = []

        # Indian legal section patterns
        patterns = [
            r'Section\s+\d+[A-Z]*',           # Section 138, Section 138A
            r'Art\.\s*\d+',                   # Art. 14
            r'Article\s+\d+',                 # Article 14
            r'Rule\s+\d+[A-Z]*',             # Rule 123, Rule 123A
            r'Order\s+\d+[A-Z]*',            # Order 39, Order 39A
            r'Chapter\s+[IVXLCDM]+',          # Chapter IV
            r'Schedule\s+[IVXLCDM\d]+'       # Schedule I
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            sections.extend(matches)

        return list(set(sections))  # Remove duplicates

    def _extract_parties(self, text: str) -> List[str]:
        """Extract party names from legal text"""
        parties = []

        # Look for patterns like "petitioner", "respondent", "appellant", etc.
        party_indicators = [
            r'([A-Za-z\s]+)\s*\([^)]*petitioner[^)]*\)',
            r'([A-Za-z\s]+)\s*\([^)]*respondent[^)]*\)',
            r'([A-Za-z\s]+)\s*\([^)]*appellant[^)]*\)',
            r'([A-Za-z\s]+)\s*\([^)]*defendant[^)]*\)',
            r'([A-Za-z\s]+)\s*\([^)]*plaintiff[^)]*\)'
        ]

        for pattern in party_indicators:
            matches = re.findall(pattern, text, re.IGNORECASE)
            parties.extend([match.strip() for match in matches if len(match.strip()) > 3])

        return list(set(parties))  # Remove duplicates

    def _extract_spacy_entities(self, text: str, source: str) -> List[Dict[str, Any]]:
        """Extract named entities using spaCy"""
        entities = []

        try:
            spacy_doc = nlp(text)

            for ent in spacy_doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LAW'] and len(ent.text.strip()) > 2:
                    entity_type = 'entity'
                    if ent.label_ == 'PERSON':
                        entity_type = 'judge'  # Assume persons in legal context are judges
                    elif ent.label_ == 'ORG':
                        entity_type = 'party'
                    elif ent.label_ == 'LAW':
                        entity_type = 'law'

                    entities.append({
                        'id': f"{entity_type}_{ent.text.replace(' ', '_').lower()}",
                        'type': entity_type,
                        'name': ent.text.strip(),
                        'text': ent.text.strip(),
                        'source': source,
                        'spacy_label': ent.label_
                    })
        except Exception as e:
            logger.warning(f"Error extracting spaCy entities: {e}")

        return entities

    def build_graph_from_documents(self, documents: List[Document], use_cache: bool = True) -> nx.DiGraph:
        """
        Build knowledge graph from legal documents with caching support

        Args:
            documents: List of Document objects
            use_cache: Whether to use caching for performance

        Returns:
            NetworkX DiGraph representing the knowledge graph
        """
        logger.info(f"Building knowledge graph from {len(documents)} documents")

        if use_cache:
            cache_key = self._get_cache_key(documents)
            cached_graph = self.load_cached_graph(cache_key)

            if cached_graph:
                logger.info("Using cached knowledge graph")
                self.graph = cached_graph
                return self.graph

        # Extract entities from all documents
        all_entities = []
        for doc in documents:
            if isinstance(doc, Document):
                source = doc.metadata.get('source', 'Unknown')
                entities = self.extract_legal_entities(doc.page_content, source)
                all_entities.extend(entities)

                # Add document node
                doc_id = f"doc_{os.path.basename(source)}"
                self.graph.add_node(doc_id,
                                  type='document',
                                  label=os.path.basename(source),
                                  title=f"Source: {source}",
                                  color=self.entity_types['document']['color'],
                                  shape=self.entity_types['document']['shape'],
                                  size=self.entity_types['document']['size'])

        # Add entity nodes
        for entity in all_entities:
            entity_type = entity['type']
            node_props = self.entity_types.get(entity_type, self.entity_types['entity'])

            self.graph.add_node(entity['id'],
                              type=entity_type,
                              label=entity['name'],
                              title=f"{entity_type.title()}: {entity['name']}\nSource: {entity['source']}",
                              color=node_props['color'],
                              shape=node_props['shape'],
                              size=node_props['size'])

            # Connect entity to its source document
            doc_id = f"doc_{os.path.basename(entity['source'])}"
            if doc_id in self.graph.nodes:
                self.graph.add_edge(doc_id, entity['id'],
                                  label='contains',
                                  color=self.relationship_types['contains']['color'],
                                  width=self.relationship_types['contains']['width'],
                                  title=f"{os.path.basename(entity['source'])} contains {entity['name']}")

        # Create relationships between entities
        self._create_entity_relationships(documents)

        # Cache the graph if caching is enabled
        if use_cache:
            self.save_graph_cache(cache_key, self.graph)

        logger.info(f"Knowledge graph built with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
        return self.graph

    def _create_entity_relationships(self, documents: List[Document]) -> None:
        """Create relationships between entities based on document analysis"""
        # Simple co-occurrence based relationships
        entity_sources = {}

        # Group entities by source
        for doc in documents:
            if isinstance(doc, Document):
                source = doc.metadata.get('source', 'Unknown')
                entities = self.extract_legal_entities(doc.page_content, source)

                for entity in entities:
                    if entity['id'] not in entity_sources:
                        entity_sources[entity['id']] = []
                    entity_sources[entity['id']].append(source)

        # Create citation relationships
        for doc in documents:
            if isinstance(doc, Document):
                content = doc.page_content
                source = doc.metadata.get('source', 'Unknown')
                doc_id = f"doc_{os.path.basename(source)}"

                # Find citations in this document
                citations = self._extract_case_citations(content)

                for citation in citations:
                    citation_id = f"case_{citation.replace(' ', '_').replace('(', '').replace(')', '').lower()}"
                    if citation_id in self.graph.nodes:
                        self.graph.add_edge(doc_id, citation_id,
                                          label='cites',
                                          color=self.relationship_types['cites']['color'],
                                          width=self.relationship_types['cites']['width'],
                                          title=f"{os.path.basename(source)} cites {citation}")

    def _create_incremental_relationships(self, new_entities: List[Dict[str, Any]]) -> None:
        """Create relationships for newly added entities with existing graph"""
        logger.info(f"Creating incremental relationships for {len(new_entities)} new entities")

        # Group new entities by source
        new_entity_sources = {}
        for entity in new_entities:
            entity_id = entity['id']
            source = entity['source']
            if entity_id not in new_entity_sources:
                new_entity_sources[entity_id] = []
            new_entity_sources[entity_id].append(source)

        # Create relationships between new entities and existing entities
        existing_entity_ids = set(self.graph.nodes.keys()) - {entity['id'] for entity in new_entities}

        # Simple co-occurrence based relationships (only for new entities)
        for new_entity_id in new_entity_sources.keys():
            if new_entity_id in self.graph.nodes:
                new_entity_sources_list = new_entity_sources[new_entity_id]

                # Connect to existing entities that appear in the same sources
                for existing_entity_id in existing_entity_ids:
                    if existing_entity_id in self.graph.nodes:
                        # Check if they share sources (simplified co-occurrence)
                        existing_sources = self.graph.nodes[existing_entity_id].get('sources', [])
                        shared_sources = set(new_entity_sources_list) & set(existing_sources)

                        if shared_sources:
                            # Add relationship if they co-occur in documents
                            if not self.graph.has_edge(new_entity_id, existing_entity_id):
                                self.graph.add_edge(new_entity_id, existing_entity_id,
                                                  label='related_to',
                                                  color=self.relationship_types['related_to']['color'],
                                                  width=self.relationship_types['related_to']['width'],
                                                  title=f"Co-occur in {len(shared_sources)} document(s)")

        logger.info("Incremental relationships creation completed")

    def visualize_graph(self, height: str = "800px", width: str = "100%") -> str:
        """
        Create interactive visualization using Pyvis

        Args:
            height: Height of the visualization
            width: Width of the visualization

        Returns:
            HTML string of the visualization
        """
        try:
            net = Network(height=height, width=width, directed=True,
                         notebook=False, cdn_resources='remote',
                         bgcolor="#ffffff", font_color="black")

            # Configure physics and interaction
            net.set_options("""
            {
              "physics": {
                "enabled": true,
                "barnesHut": {
                  "gravitationalConstant": -80000,
                  "centralGravity": 0.3,
                  "springLength": 250,
                  "springConstant": 0.001,
                  "damping": 0.09,
                  "avoidOverlap": 0
                },
                "stabilization": {
                  "enabled": true,
                  "iterations": 1000
                }
              },
              "interaction": {
                "dragNodes": true,
                "hideEdgesOnDrag": false,
                "hideNodesOnDrag": false,
                "hover": true,
                "multiselect": true,
                "navigationButtons": true
              },
              "edges": {
                "smooth": {
                  "enabled": true,
                  "type": "dynamic"
                },
                "arrows": {
                  "to": {
                    "enabled": true,
                    "scaleFactor": 0.5
                  }
                }
              },
              "nodes": {
                "font": {
                  "size": 12,
                  "face": "arial"
                }
              }
            }
            """)

            # Add nodes
            for node_id, node_data in self.graph.nodes(data=True):
                net.add_node(node_id,
                           label=node_data.get('label', str(node_id)),
                           title=node_data.get('title', ''),
                           color=node_data.get('color', '#97c2fc'),
                           shape=node_data.get('shape', 'dot'),
                           size=node_data.get('size', 20))

            # Add edges
            for source, target, edge_data in self.graph.edges(data=True):
                edge_props = {
                    'label': edge_data.get('label', ''),
                    'title': edge_data.get('title', ''),
                    'color': edge_data.get('color', '#999999'),
                    'width': edge_data.get('width', 1)
                }

                # Handle dashed edges
                if edge_data.get('dashes', False):
                    edge_props['dashes'] = True

                net.add_edge(source, target, **edge_props)

            # Generate HTML
            html = net.generate_html()
            return html

        except Exception as e:
            logger.error(f"Error creating graph visualization: {e}")
            return ""

    def analyze_graph(self) -> Dict[str, Any]:
        """
        Perform graph analysis and return statistics

        Returns:
            Dictionary containing graph statistics
        """
        analysis = {
            'basic_stats': {
                'num_nodes': len(self.graph.nodes),
                'num_edges': len(self.graph.edges),
                'num_components': nx.number_weakly_connected_components(self.graph),
                'density': nx.density(self.graph) if len(self.graph.nodes) > 1 else 0
            },
            'entity_distribution': {},
            'relationship_distribution': {}
        }

        # Analyze entity distribution
        entity_counts = {}
        for node_id, node_data in self.graph.nodes(data=True):
            entity_type = node_data.get('type', 'unknown')
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        analysis['entity_distribution'] = entity_counts

        # Analyze relationship distribution
        relationship_counts = {}
        for source, target, edge_data in self.graph.edges(data=True):
            rel_type = edge_data.get('label', 'unknown')
            relationship_counts[rel_type] = relationship_counts.get(rel_type, 0) + 1
        analysis['relationship_distribution'] = relationship_counts

        return analysis


def create_knowledge_graph(documents: List[Document], min_entity_length: int = 3) -> nx.Graph:
    """
    Creates a knowledge graph from a list of documents using the enhanced LegalKnowledgeGraph class.
    """
    kg = LegalKnowledgeGraph()
    return kg.build_graph_from_documents(documents)


def visualize_knowledge_graph(G: nx.Graph, height: str = "800px", width: str = "100%") -> str:
    """
    Visualizes a NetworkX graph using Pyvis with enhanced configuration.
    """
    kg = LegalKnowledgeGraph()
    kg.graph = G  # Use the provided graph
    return kg.visualize_graph(height, width)


def knowledge_graph_page(vectorstore) -> None:
    """
    Renders the enhanced knowledge graph page in the Streamlit app.
    """
    st.header("🕸️ Legal Knowledge Graph")

    st.markdown("""
    This page generates an interactive knowledge graph from your legal documents.
    The graph shows relationships between legal entities like courts, cases, sections, and parties.
    """)

    # Configuration options
    col1, col2, col3 = st.columns(3)
    with col1:
        min_entity_length = st.slider("Minimum entity length", 2, 10, 3,
                                    help="Minimum length of entities to extract")
    with col2:
        max_nodes = st.slider("Maximum nodes to display", 50, 1000, 200,
                            help="Limit nodes for performance")
    with col3:
        include_spacy_entities = st.checkbox("Include spaCy entities", value=True,
                                           help="Include named entities detected by spaCy")

    # Get all documents from the vector store
    try:
        if st.button("🔍 Generate Knowledge Graph", type="primary"):
            with st.spinner("Loading documents..."):
                store_data = vectorstore.get()
                documents = store_data.get("documents", [])
                metadatas = store_data.get("metadatas", [])

                docs = [Document(page_content=doc, metadata=meta)
                       for doc, meta in zip(documents, metadatas)]

            if not docs:
                st.warning("⚠️ No documents found to generate a knowledge graph.")
                return

            with st.spinner("🔍 Analyzing documents and extracting entities..."):
                # Create knowledge graph
                kg = LegalKnowledgeGraph()
                G = kg.build_graph_from_documents(docs)

                if G.number_of_nodes() == 0:
                    st.warning("⚠️ Knowledge graph is empty. No entities were extracted.")
                    return

                # Limit nodes for performance
                if G.number_of_nodes() > max_nodes:
                    # Keep only the most connected nodes
                    degrees = dict(G.degree())
                    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
                    G = G.subgraph(top_nodes).copy()
                    st.info(f"📊 Limited to top {max_nodes} most connected nodes for performance.")

                # Display graph statistics
                analysis = kg.analyze_graph()

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📄 Documents", analysis['basic_stats']['num_nodes'])
                with col2:
                    st.metric("🔗 Relationships", analysis['basic_stats']['num_edges'])
                with col3:
                    st.metric("🏛️ Components", analysis['basic_stats']['num_components'])

                # Display entity distribution
                if analysis['entity_distribution']:
                    st.subheader("📊 Entity Distribution")
                    entity_df = pd.DataFrame(
                        list(analysis['entity_distribution'].items()),
                        columns=['Entity Type', 'Count']
                    )
                    st.bar_chart(entity_df.set_index('Entity Type'))

                # Generate and display visualization
                with st.spinner("🎨 Creating interactive visualization..."):
                    html = kg.visualize_graph(height="600px")

                    if html:
                        st.subheader("🕸️ Interactive Knowledge Graph")
                        components.html(html, height=600, scrolling=True)

                        # Download options
                        st.download_button(
                            label="📥 Download Graph HTML",
                            data=html,
                            file_name="legal_knowledge_graph.html",
                            mime="text/html"
                        )

                        st.success(f"✅ Generated graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
                    else:
                        st.error("❌ Failed to generate graph visualization.")

    except Exception as e:
        st.error(f"❌ Error generating knowledge graph: {e}")
        logger.error(f"Knowledge graph error: {e}")