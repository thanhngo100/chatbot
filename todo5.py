# Library
import openai
import os
import streamlit as st
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import pdfplumber
import time

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# Custom Streamlit app title and icon
st.set_page_config(
    page_title="ThanhNgo100 Bot",
    page_icon=":robot_face:",
)

# Set the title
st.title("ChatGPT - PDF")

# Sidebar Configuration
st.sidebar.title("Model Configuration")

# Model Name Selector
model_name = st.sidebar.selectbox(
    "Select a Model",
    ["gpt-3.5-turbo", "gpt-4"],  # Add more model names as needed
    key="model_name",
)

# Temperature Slider
temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.2,
    max_value=2.0,
    value=1.0,
    step=0.1,
    key="temperature",
)

# Max tokens Slider
max_tokens = st.sidebar.slider(
    "Max Tokens",
    min_value=1,
    max_value=4095,
    value=256,
    step=1,
    key="max_tokens",
)

# Top p Slider
top_p = st.sidebar.slider(
    "Top P",
    min_value=0.00,
    max_value=1.00,
    value=1.00,
    step=0.01,
    key="top_p",
)

# Presence penalty Slider
presence_penalty = st.sidebar.slider(
    "Presence penalty",
    min_value=0.00,
    max_value=2.00,
    value=0.00,
    step=0.01,
    key="presence_penalty",
)

# Frequency penalty Slider
frequency_penalty = st.sidebar.slider(
    "Frequency penalty",
    min_value=0.00,
    max_value=2.00,
    value=0.00,
    step=0.01,
    key="frequency_penalty",
)


uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")


# Set OPENAI API
openai.api_key = st.secrets["OPENAI_API_KEY"]
load_dotenv()
# openai.api_key = os.environ['OPENAI_API_KEY']

# Initialize DataFrame to store chat history
chat_history_df = pd.DataFrame(columns=["Timestamp", "Chat"])

# Initialize Chat Messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize full_response outside the user input check
full_response = ""

# Reset Button
if st.sidebar.button("Reset Chat"):
    # Save the chat history to the DataFrame before clearing it
    if st.session_state.messages:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        chat_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
        new_entry = pd.DataFrame({"Timestamp": [timestamp], "Chat": [chat_history]})
        chat_history_df = pd.concat([chat_history_df, new_entry], ignore_index=True)

        # Save the DataFrame to a CSV file
        chat_history_df.to_csv("chat_history.csv", index=False)

    # Clear the chat messages and reset the full response
    st.session_state.messages = []
    full_response = ""

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def extract_data(feed):
    # data = []
    text  = ''
    with pdfplumber.open(feed) as pdf:
        pages = pdf.pages
        # print(pages)
        # for p in pages:
        #     data.append(p.extract_tables())
        for i, page in enumerate(pages):
            if(i >= 9):
                break
            text += f"### Page {i}:\n\n" + page.extract_text()
            # break  # For a single page, we only take the first page
    return text # build more code to return a dataframe 

text = None
if uploaded_file is not None:
    with st.spinner('uploading...'):
        text = extract_data(uploaded_file)
    st.success('upload done!') 
knowledgeBase = None
def process_text(text):
    knowledgeBase = None
    if text is not None:
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

# User Input and AI Response
if knowledgeBase is not None:
    
    if prompt := st.chat_input("What is up?"):
        # Optional
        # st.session_state.messages.append({"role": "system", "content": "You are a helpful assistant named Jarvis"})
        
        # docs = knowledgeBase.similarity_search(prompt)
        # llm = OpenAI()
        # chain = load_qa_chain(llm, chain_type='stuff')
        # # Run the question-answering chain to find the answer to the question
        # response = chain.run(input_documents=docs, question=prompt)

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            for response in openai.ChatCompletion.create(
                model=model_name,  # Use the selected model name
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                temperature=temperature,  # Set temperature
                max_tokens=max_tokens,  # Set max tokens
                top_p=top_p, # Set top p
                frequency_penalty=frequency_penalty, # Set frequency penalty
                presence_penalty=presence_penalty, # Set presence penalty
                stream=True,
            ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})