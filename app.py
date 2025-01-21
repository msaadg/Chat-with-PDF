import streamlit as st
import torch
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from web_template import css, bot_template, user_template
from langchain_community.vectorstores import FAISS



def get_pdf_content(documents, method = 1):
    if method == 1:
        raw_text = ""
        for document in documents:
            pdf_reader = PdfReader(document)
            for page in pdf_reader.pages:
                raw_text += page.extract_text()
        print(f"Extracted {len(raw_text)} characters from PDF")
        return raw_text
    if method == 2:
        list_of_files = documents
        list_of_docs = []
        for file in list_of_files:
            print(file)

        loader = PyPDFLoader(file)
        #     docs = loader.load()
        #     list_of_docs.append(docs)
        # print(f"Loaded {len(list_of_docs)} pages from PDF")
        # return list_of_docs
    


def get_chunks(text, max_tokens=512, method = 1):
    if method == 1:
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        text_chunks = text_splitter.split_text(text)
        print(f"Split text into {len(text_chunks)} chunks")
        return text_chunks
    if method == 2:
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
        )
        text_chunks = text_splitter.create_documents(text)
        print(f"Split text into {len(text_chunks)} chunks")
        return text_chunks
    else:
        return "Invalid method"
    
def get_embeddings(chunks, method = 1):
    if method == 1:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vector_storage = FAISS.from_texts(texts=chunks, embedding=embeddings)
        return vector_storage
    if method == 2:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        vector_store = FAISS(embedding_function=embeddings)
        ids = vector_store.add_documents(documents=chunks)
        return vector_store


def initialize_local_llm():

    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15,
        do_sample=True,
    )
    
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

def start_conversation(vector_embeddings, method = 1):
    llm = initialize_local_llm()
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    if method == 1:
        conversation = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_embeddings.as_retriever(),
            memory=memory
        )
    if method == 2:
        conversation = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_embeddings.as_retriever( search_type="similarity", search_kwargs={"k": 1},),
            memory=memory
        )
    return conversation
    


def process_query(query_text):
    if st.session_state.conversation is None: # ask user to upload pdf files first and avoid an error
        st.warning("Please upload and process the PDF files first.")
        return
    with st.spinner("Processing your query. This may take a moment..."):
        response = st.session_state.conversation({'question': query_text})
        st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i%2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()

    st.set_page_config(page_title="Chatbot", page_icon=":books:", layout="wide")

    st.write(css, unsafe_allow_html=True)

    st.header("What can I help with?")
    query = st.text_input("Enter your query here")

    

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if query: #repositioned this block to ensure that the conversation object is properly initialized
        process_query(query)


    with st.sidebar:
        st.subheader("PDF documents")
        documents = st.file_uploader(
            "Upload your PDF files", type=["pdf"], accept_multiple_files=True
        )
        if st.button("Run"):
            with st.spinner("Processing..."):
                # extract text from pdf documents
                extracted_text = get_pdf_content(documents, 2)
                # # convert text to chunks of data
                # text_chunks = get_chunks(extracted_text, 1)
                # # create vector embeddings
                # vector_embeddings = get_embeddings(text_chunks, 1)
                # # create conversation
                # st.session_state.conversation = start_conversation(vector_embeddings, 1)


if __name__ == "__main__":
    main()
