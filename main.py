
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ChatMessageHistory
import streamlit as st
import requests
import sys

# only for deploy
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# only for local develop
# from dotenv import load_dotenv
# load_dotenv()


# load the PDF file from web
pdf_url = "https://github.com/Photobyme/chatbot-langchain-chatgpt-streamlit/blob/main/photobyme-info.pdf?raw=true"
response = requests.get(pdf_url)
if response.status_code == 200:
    with open("photobyme-info.pdf", "wb") as file:
        file.write(response.content)

#
# initialize the chatbot and load data from PDF
chat = ChatOpenAI(model="gpt-3.5-turbo-1106")
loader = PyPDFLoader("photobyme-info.pdf")
data = loader.load()

# embegging
textSplitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
allSplits = textSplitter.split_documents(data)
vectorstore = Chroma.from_documents(
    documents=allSplits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(k=100)

# retrive data from embedding
docs = retriever.invoke("photobyme")
questionAnsweringPrompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're the Photobyme's customer service agent. Answer the customer's questions based on the below context:\n\n{context}",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
documentChain = create_stuff_documents_chain(chat, questionAnsweringPrompt)
chatHistory = ChatMessageHistory()

# Streamlit app
st.title('Photobyme AI Chatbot')

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(
        {"role": "assistant", "content": "Hello 👋 How can I help you today? "})

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask questions?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Wait for it..."):
        chatHistory.add_user_message(prompt)
        result = documentChain.invoke(
            {
                "messages": chatHistory.messages,
                "context": docs,
            }
        )

    response = result
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append(
        {"role": "assistant", "content": response})
