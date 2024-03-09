import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever , create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.llms import HuggingFaceHub


load_dotenv()

def get_vectorstore(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vector_store


def get_context_retriever_chain(vector_store):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    retriever = vector_store.as_retriever()  
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain
    
def get_conversational_rag_chain(retriever_chain): 
    
    llm = ChatOpenAI()
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })
    
    return response['answer']

st.set_page_config(page_title="PDF-QA-AI", page_icon="📚")

with st.sidebar:
    st.title("PDF-QA-AI")
    st.header("We work with any E-Books, Notes, or Reference materials in PDF format")
    pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True)
    if st.button("Process PDFs"):
            with st.spinner("Processing"):
                 st.session_state.vector_store = get_vectorstore(pdf_docs)

    if st.button("Download Chat"):
        text_dwnld = ' '.join(map(str, st.session_state.chat_history))
        st.download_button('Confirm Download?',  text_dwnld)

    if st.button("Delete Chat"):
            st.session_state.chat_history = [
            AIMessage(content="Hi I am PDF-QA-AI, your notes provider!!"),
        ]

        
    
# if "vector_store" not in st.session_state:
#      st.session_state.vector_store = get_vectorstore(pdf_docs)

if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hi I am PDF-QA-AI, your notes provider!!"),
        ]

user_query = st.chat_input("Ask your question....")



if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
        

for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

