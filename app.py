import openai
from dotenv import load_dotenv
import streamlit as st
import os
from PyPDF2 import PdfReader

load_dotenv()

st.set_page_config(
    page_title="VietAI Bot",
    page_icon=":robot_face:",
)

st.title("[VietAI-NTI] ChatGPT")

# Set OpenAI API key from Streamlit secrets
# openai.api_key = st.secrets["OPENAI_API_KEY"]

openai.api_key = os.environ['OPENAI_API_KEY']


pdf = 'D:/Develop/AI/assignment1/Deep_Learning.pdf'

# Initialize an empty list to store chat history
chat_history = []


# Initialize a PdfReader object to read the PDF file
pdf_reader = PdfReader(pdf)

# Initialize an empty string to store the extracted text from the PDF
chunks = []

# Iterate through the pages of the PDF and extract text from the first page only
for i, page in enumerate(pdf_reader.pages):
    chunks.append(f"### Page {i}:\n\n" + page.extract_text())

# Append a system message to the chat history to set the context
chat_history.append({"role": "system", "content": f"""You are an information retrieval assistant. Text information: {text} ## Task: Answer questions using solely the information from the above text"""})


# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialize chat history    
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
# Finally, the AI's full response is stored in the chat history.
# '''
if prompt := st.chat_input("Bạn cần hỗ trợ điều gì?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})    