from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage

load_dotenv()

chat = ChatOpenAI(model="gpt-3.5-turbo-1106"
)
# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a helpful assistant. Answer all questions to the best of your ability.",
#         ),
#         MessagesPlaceholder(variable_name="messages"),
#     ]
# )

# chain = prompt | chat

# result = chain.invoke(
#     {
#         "messages": [
#             HumanMessage(
#                 content="Translate this sentence from English to French: I love programming."
#             ),
#             AIMessage(content="J'adore la programmation."),
#             HumanMessage(content="What did you just say?"),
#         ],
#     }
# )
# print(result)

# from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("photobyme-info.pdf")
data = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

# k is the number of chunks to retrieve
retriever = vectorstore.as_retriever(k=100)

docs = retriever.invoke("photo booth")

from langchain.chains.combine_documents import create_stuff_documents_chain

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


from langchain.memory import ChatMessageHistory

demo_ephemeral_chat_history = ChatMessageHistory()

# demo_ephemeral_chat_history.add_user_message("How much is renting photo booth?")

# result = document_chain.invoke(
#     {
#         "messages": demo_ephemeral_chat_history.messages,
#         "context": docs,
#     }
# )

import streamlit as st 

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
# else:
#     st.button("hi")