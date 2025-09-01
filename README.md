# Legal Research Engine

This project is an AI-powered legal research assistant designed to help legal professionals analyze and understand Indian legal documents. It provides a chat interface to ask questions about a corpus of legal documents, and it can also visualize the relationships between documents and citations in a citation graph.

## Features

*   **Conversational Q&A:** Chat with your legal documents using natural language. The system uses a Retrieval-Augmented Generation (RAG) pipeline to provide answers based on the content of the documents.
*   **Choice of LLMs:** Supports both the Gemini API and local models running on LM Studio, giving you flexibility and control over your data.
*   **Citation Graph:** Visualize the network of citations within your documents. This helps to understand the relationships between different legal cases and statutes.
*   **Citation Analysis:** Get a breakdown of the most frequently cited documents and the citations found in each document.
*   **Easy Data Ingestion:** Simply place your PDF legal documents in the `data` directory and the app will do the rest.

## Tech Stack

*   **Backend:** Python, Streamlit, LangChain, FastAPI
*   **Frontend:** Streamlit
*   **Vector Database:** ChromaDB
*   **LLMs:** Google Gemini, LM Studio (local models)
*   **Data Processing:** PyPDF, NetworkX, Pyvis

## Getting Started

### Prerequisites

*   Python 3.8+
*   An API key for Google Gemini (if you want to use the Gemini model)
*   LM Studio running locally (if you want to use a local model)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/research-engine-for-courts.git
    cd research-engine-for-courts
    ```

2.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### Usage

1.  **Add your documents:** Place your PDF legal documents in the `data` directory.

2.  **Set up your API key (optional):** If you are using the Gemini model, create a `.env` file in the root of the project and add your Google API key:

    ```
    GOOGLE_API_KEY="your-google-api-key"
    ```

3.  **Run the Streamlit app:**

    ```bash
    streamlit run app.py
    ```

4.  **Open the app in your browser:** The app will be available at `http://localhost:8501`.

5.  **Ingest the data:** In the sidebar of the app, click the "Ingest Data" button to process your documents and create the vector store.

6.  **Start chatting:** You can now ask questions about your documents in the chat interface.

## Project Structure

```
.
├── app.py                  # The main Streamlit application
├── citation_graph.html     # The generated citation graph
├── data_collector.py       # Script to download legal documents
├── project.md              # Detailed project plan
├── rag_pipeline.ipynb      # Jupyter notebook with the RAG pipeline
├── timeline.md             # Project timeline
├── requirements.txt        # Python dependencies
├── chroma/                 # Chroma vector store
└── data/                   # Directory for your PDF documents
```

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
