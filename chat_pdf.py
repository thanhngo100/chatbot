
# video nộp bài.
# 1. chạy file pdf test deep lerning.

from dotenv import load_dotenv
import os
import openai
import mylib
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

openai.api_key = mylib.api_key ## To configure OpenAI API
os.environ["OPENAI_API_KEY"] = mylib.api_key ## To configure langchain connections with OpenAI

# pdf = 'D:/Develop/AI/assignment1/ccdc.pdf'

pdf = 'D:/Develop/AI/assignment1/Deep_Learning.pdf'

# Initialize an empty list to store chat history
chat_history = []

# Initialize a PdfReader object to read the PDF file
pdf_reader = PdfReader(pdf)

# Initialize an empty string to store the extracted text from the PDF
text = ""

# Iterate through the pages of the PDF and extract text from the first page only
for i, page in enumerate(pdf_reader.pages):
    text += f"### Page {i}:\n\n" + page.extract_text()
    break  # For a single page, we only take the first page

# Append a system message to the chat history to set the context
chat_history.append({"role": "system", "content": f"""You are an information retrieval assistant. Text information: {text} ## Task: Answer questions using solely the information from the above text"""})

def ask_question(question):
    global chat_history
    chat_history.append({"role": "user", "content": question})

    # Define parameters for the OpenAI ChatCompletion API
    params = dict(model="gpt-3.5-turbo", messages=chat_history)

    # Generate a response from the AI model
    response = openai.ChatCompletion.create(**params)
    content = response.choices[0].message.content
    chat_history.append({"role": "assistant", "content": content})

    # Print and return the assistant's response
    print(content)
    
def process_text(text):
    # Split the text into chunks using Langchain's CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator="### ",
        chunk_overlap=200,
    )
    chunks = text_splitter.split_text(text)

    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = OpenAIEmbeddings()
    knowledgeBase = FAISS.from_texts(chunks, embeddings)

    return knowledgeBase

# Process the extracted text to create a knowledge base
knowledgeBase = process_text(text)

# Define a question for which you want to find an answer  
# Đổi file PDF
# question = "Danh sách CCDC gồm những gì?"

# print('Train tài liệu công cụ dụng cụ!')
# print('Câu hỏi: ', question )


question = "What is Logistic regression cost function?"

# question = "What is a (Neural Network) NN?"  => đổi câu hỏi

print('Train tài liệu deep learning!')
print('Câu hỏi: ', question )

# Use embeddings similarity search to search for the closest document (RAG)
docs = knowledgeBase.similarity_search(question)
llm = OpenAI()
chain = load_qa_chain(llm, chain_type='stuff')
# Run the question-answering chain to find the answer to the question
response = chain.run(input_documents=docs, question=question)

# Print the response
print('Câu trả lời là:', response)