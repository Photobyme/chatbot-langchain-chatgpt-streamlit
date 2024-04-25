# streamlit
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# api key for local
# from dotenv import load_dotenv
# load_dotenv()
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





# Load the PDF file
pdf_url = "https://github.com/Photobyme/chatbot-langchain-chatgpt-streamlit/blob/main/photobyme-info.pdf?raw=true"
response = requests.get(pdf_url)
if response.status_code == 200:
    with open("photobyme-info.pdf", "wb") as file:
        file.write(response.content)

# Initialize the chatbot and document processing
chat = ChatOpenAI(model="gpt-3.5-turbo-1106")
loader = PyPDFLoader("photobyme-info.pdf")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(k=100)
docs = retriever.invoke("photo booth")
question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user's questions based on the below context:\n\n{context}",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
document_chain = create_stuff_documents_chain(chat, question_answering_prompt)
demo_ephemeral_chat_history = ChatMessageHistory()

# Streamlit app
st.title('Photobyme AI Chatbot')
userInput = st.text_input("Have any questions?")
if st.button("Please, answer"):
    with st.spinner("Wait for it..."):
        demo_ephemeral_chat_history.add_user_message(userInput)
        result = document_chain.invoke(
            {
                "messages": demo_ephemeral_chat_history.messages,
                "context": docs,
            }
        )
        st.write(result)
