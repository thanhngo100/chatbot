import os
import openai
from PyPDF2 import PdfReader
from dotenv import load_dotenv

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI


load_dotenv()
print(os.environ["OPENAI_API_KEY"])
openai.api_key = os.environ["OPENAI_API_KEY"] ## To configure OpenAI API


# # Specify the PDF file to be processed
pdf = 'G:/My Drive/GPT AI - API/Deep Learning.pdf'

# # Initialize an empty list to store chat history
# chat_history = []
# chunks = []
# i=0

# # Initialize a PdfReader object to read the PDF file
# pdf_reader = PdfReader(pdf)

# # Initialize an empty string to store the extracted text from the PDF


# # Iterate through the pages of the PDF and extract text from each page
# for i, page in enumerate(pdf_reader.pages):
#     if(i > 5):
#         print ('quá 5 rồi!')
#         break
#     chunks.append(f"### Page {i}:\n\n" + page.extract_text())
        
# # Print the number of pages in the PDF
# print("Number of pages:", len(pdf_reader.pages))


# # Append a system message to the chat history to set the context
# chat_history.append({"role": "system", "content": f"""You are an information retrieval assistant.Text information: {chunks} ### Task: Answer questions using solely the information from the above text"""})

# def process_text(chunks):
    
#     # Convert the chunks of text into embeddings to form a knowledge base
#     embeddings = OpenAIEmbeddings()
#     knowledgeBase = FAISS.from_texts(chunks, embeddings)
#     return knowledgeBase

# # Process the extracted text to create a knowledge base
# knowledgeBase = process_text(chunks)


# # Define a function to ask questions and interact with the AI model
# def ask_question(question):
#     global chat_history
#     chat_history.append({"role": "user", "content": question})

#     # Define parameters for the OpenAI ChatCompletion API
#     params = dict(model="gpt-3.5-turbo", messages=chat_history)

#     # Generate a response from the AI model
#     response = openai.ChatCompletion.create(**params)
#     content = response.choices[0].message.content
#     chat_history.append({"role": "assistant", "content": content})

#     # Print and return the assistant's response
#     print(content)




# question = "What is Logistic regression cost function?"

# # Use embeddings similarity search to search for the closest document (RAG)
# docs = knowledgeBase.similarity_search(question)
# llm = OpenAI()
# chain = load_qa_chain(llm, chain_type='stuff')

# # Run the question-answering chain to find the answer to the question
# response = chain.run(input_documents=docs, question=question)

# # Print the response
# print(response)