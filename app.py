import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
# import spacy

from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# from langchain.chat_models import ChatOpenAI
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch

from web_template import css, bot_template, user_template
# nlp = spacy.load("en_core_web_sm")


def get_pdf_content(documents):
    raw_text = ""

    for document in documents:
        pdf_reader = PdfReader(document)
        for page in pdf_reader.pages:
            raw_text += page.extract_text()

    return raw_text


def get_chunks(text, max_tokens=512):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    text_chunks = text_splitter.split_text(text)
    return text_chunks

    # doc = nlp(text)
    # sentences = [sent.text for sent in doc.sents]
    
    # chunks = []
    # current_chunk = ""
    # current_length = 0
    
    # for sentence in sentences:
    #     sentence_length = len(sentence.split())  # Approximate token count
    #     if current_length + sentence_length > max_tokens:
    #         if current_chunk:
    #             chunks.append(current_chunk.strip())
    #             current_chunk = sentence
    #             current_length = sentence_length
    #         else:
    #             # Single sentence longer than max_tokens
    #             chunks.append(sentence.strip())
    #             current_chunk = ""
    #             current_length = 0
    #     else:
    #         current_chunk += " " + sentence
    #         current_length += sentence_length
    
    # if current_chunk:
    #     chunks.append(current_chunk.strip())
    
    # return chunks


def get_embeddings(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_storage = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vector_storage


def initialize_local_llm():
    # Initialize TinyLlama model
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Configure quantization
    # quantization_config = BitsAndBytesConfig(
    #     load_in_8bit=True,
    #     bnb_8bit_compute_dtype=torch.float16
    # )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        # quantization_config=quantization_config  # Use the new config instead of load_in_8bit
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

def start_conversation(vector_embeddings):
    # llm = ChatOpenAI()
    llm = initialize_local_llm()
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_embeddings.as_retriever(),
        memory=memory
    )

    return conversation


def process_query(query_text):
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

    if query:
        process_query(query)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None


    with st.sidebar:
        st.subheader("PDF documents")
        documents = st.file_uploader(
            "Upload your PDF files", type=["pdf"], accept_multiple_files=True
        )
        if st.button("Run"):
            with st.spinner("Processing..."):
                # extract text from pdf documents
                extracted_text = get_pdf_content(documents)
                # convert text to chunks of data
                text_chunks = get_chunks(extracted_text)
                # create vector embeddings
                vector_embeddings = get_embeddings(text_chunks)
                # create conversation
                st.session_state.conversation = start_conversation(vector_embeddings)


if __name__ == "__main__":
    main()
