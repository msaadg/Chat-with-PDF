import streamlit as st
import os
import time
import hashlib
from datetime import datetime

from langchain_community.document_loaders import PDFPlumberLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_nomic import NomicEmbeddings

# Configuration
DEFAULT_MODEL = "deepseek-r1:1.5b"
EMBEDDING_MODEL = "nomic-embed-text-v1.5"
VECTOR_DB_PATH = "vector_db/"
PDF_STORAGE = "pdf_storage/"
MAX_FILE_SIZE_MB = 50
MAX_HISTORY = 20

# Initialize session state
def init_session():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "processing_times" not in st.session_state:
        st.session_state.processing_times = {}
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None

init_session()

# Security functions
def validate_pdf(file):
    if file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise ValueError(f"File size exceeds {MAX_FILE_SIZE_MB}MB limit")
    # Add more security checks as needed
    return True

# Enhanced template with chain of thought
TEMPLATE = """<|system|>
You are an expert research assistant. Analyze the question and documents thoroughly.
Follow these steps:
1. Understand the question and identify key concepts
2. Review the provided context carefully
3. Consider potential misunderstandings
4. Formulate a comprehensive response
5. Verify accuracy against the context
</|system|>

<|user|>
Question: {question}

Context:
{context}
</|user|>

<|assistant|>
"""

# Add this near the start of the file, after the imports
def clear_vector_store():
    """Clear the vector store directory"""
    if os.path.exists(VECTOR_DB_PATH):
        import shutil
        shutil.rmtree(VECTOR_DB_PATH)
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)

# Database management
class VectorDBManager:
    def __init__(self):
        self.embeddings = NomicEmbeddings(model=EMBEDDING_MODEL)
        self.db = None
        
    def initialize_db(self):
        self.db = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=self.embeddings,
            collection_name="pdf_collection"
        )
        
    def add_documents(self, documents):
        if not self.db:
            self.initialize_db()
        self.db.add_documents(documents)
        
    def similarity_search(self, query, k=5):
        if not self.db:
            self.initialize_db()
        # Using MMR instead of standard similarity search
        return self.db.max_marginal_relevance_search(
            query, 
            k=k,
            fetch_k=20,  # Fetch more documents initially for better diversity
            lambda_mult=0.7  # Balance between relevance (1.0) and diversity (0.0)
        )
        
vector_db_manager = VectorDBManager()

# Document processing pipeline
class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            add_start_index=True
        )
        
    def process_pdf(self, file_path):
        start_time = time.time()
        
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
        except Exception as e:
            loader = PDFPlumberLoader(file_path)
            pages = loader.load()
            
        chunks = self.text_splitter.split_documents(pages)
        processing_time = time.time() - start_time
        
        return {
            "chunks": chunks,
            "page_count": len(pages),
            "processing_time": processing_time
        }

# Model management
class AIManager:
    def __init__(self):
        self.model = OllamaLLM(model=DEFAULT_MODEL)
        self.memory = ConversationBufferWindowMemory(k=5)
        
    def generate_response(self, question, context):
        prompt = ChatPromptTemplate.from_template(TEMPLATE)
        chain = prompt | self.model
        response = chain.invoke({"question": question, "context": context})
        return self._parse_response(response)
        
    def _parse_response(self, response):
        thinking = ""
        answer = response
        if "<think>" in response:
            parts = response.split("</think>")
            thinking = parts[0].replace("<think>", "").strip()
            answer = parts[1].strip()
        return thinking, answer

# UI Components
def sidebar_controls():
    with st.sidebar:
        st.header("Settings")
        selected_model = st.selectbox(
            "Choose AI Model",
            ["deepseek-r1:1.5b", "llama2", "mistral"],
            index=0
        )
        temperature = st.slider("Creativity", 0.0, 1.0, 0.3)
        max_length = st.slider("Max Response Length", 100, 2000, 500)
        
        st.divider()
        st.subheader("Document Management")
        if st.session_state.uploaded_files:
            selected_doc = st.selectbox("Active Documents", st.session_state.uploaded_files)
            if st.button("Clear Documents"):
                st.session_state.uploaded_files = []
                clear_vector_store()
                vector_db_manager.db = None
                
        return {
            "model": selected_model,
            "temperature": temperature,
            "max_length": max_length
        }

def document_uploader():
    uploaded_files = st.file_uploader(
        "Upload Research PDFs",
        type="pdf",
        accept_multiple_files=True,
        help="Upload multiple PDF documents for analysis"
    )
    
    for file in uploaded_files:
        if file.name not in st.session_state.uploaded_files:
            try:
                validate_pdf(file)
                file_hash = hashlib.md5(file.getvalue()).hexdigest()
                file_path = os.path.join(PDF_STORAGE, f"{file_hash}.pdf")
                
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                    
                processor = DocumentProcessor()
                result = processor.process_pdf(file_path)
                
                vector_db_manager.add_documents(result["chunks"])
                st.session_state.uploaded_files.append(file.name)
                st.session_state.processing_times[file.name] = {
                    "pages": result["page_count"],
                    "time": result["processing_time"],
                    "chunks": len(result["chunks"])
                }
                
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")

def display_chat():
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant":
                if msg.get("thinking"):
                    with st.expander("Reasoning Process"):
                        st.write(msg["thinking"])
                if msg.get("sources"):
                    with st.expander("Source Documents"):
                        for doc in msg["sources"]:
                            st.markdown(format_doc_with_page(doc))
                            st.divider()

def analytics_dashboard():
    with st.expander("System Analytics"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Processed Documents", len(st.session_state.uploaded_files))
        with col2:
            total_chunks = sum(v["chunks"] for v in st.session_state.processing_times.values())
            st.metric("Total Text Chunks", total_chunks)
        with col3:
            avg_time = sum(v["time"] for v in st.session_state.processing_times.values()) / len(st.session_state.processing_times) if st.session_state.processing_times else 0
            st.metric("Avg Processing Time", f"{avg_time:.2f}s")
        
        st.subheader("Document Details")
        for doc, stats in st.session_state.processing_times.items():
            st.write(f"**{doc}** - {stats['pages']} pages, {stats['chunks']} chunks in {stats['time']:.2f}s")

# Add this function before the main() function
def format_doc_with_page(doc):
    """Format document with page number and content for display"""
    page_num = doc.metadata.get('page_number', 'Unknown page')
    source_file = doc.metadata.get('source', 'Unknown source')
    return f"""
**Page {page_num}** from {os.path.basename(source_file)}
{doc.page_content.strip()}
"""

# Main application
def main():
    st.title("Advanced PDF Research Assistant")
    st.caption("Multi-Document Analysis with Deep Context Understanding")
    
    config = sidebar_controls()
    document_uploader()
    
    ai_manager = AIManager()
    
    if prompt := st.chat_input("Ask about the documents..."):
        st.chat_message("user").write(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        try:
            related_docs = vector_db_manager.similarity_search(prompt, k=5)
            context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(related_docs)])
            
            thinking, answer = ai_manager.generate_response(prompt, context)
            
            with st.chat_message("assistant"):
                st.write(answer)
                chat_entry = {
                    "role": "assistant",
                    "content": answer,
                    "thinking": thinking,
                    "sources": related_docs
                }
                
                with st.expander("Reasoning Process"):
                    st.write(thinking)
                with st.expander("Source References"):
                    for doc in related_docs:
                        st.markdown(format_doc_with_page(doc))
                        st.divider()
                
                st.session_state.chat_history.append(chat_entry)
                
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
    
    display_chat()
    analytics_dashboard()

if __name__ == "__main__":
    if not os.path.exists(PDF_STORAGE):
        os.makedirs(PDF_STORAGE)
    if not os.path.exists(VECTOR_DB_PATH):
        os.makedirs(VECTOR_DB_PATH)
    
    main()