import streamlit as st
import os
import mlflow
import json
import requests
import pandas as pd
import torch
 
import subprocess
from sidebar import build_sidebar
 
from langchain.chains import ConversationChain
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain import PromptTemplate
from langchain.memory import ConversationSummaryMemory
from langchain.vectorstores import Qdrant
 
from qdrant_client import QdrantClient
 
# Number of texts to match (may be less if no suitable match)
NUM_TEXT_MATCHES = 3
 
# Similarity threshold such that queried text with a lower will be discarded 
# Range [0, 1], larger = more similar for cosine similarity
SIMILARITY_THRESHOLD = 0.83
 
 
# Initialize embedding model
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embedding_model_name = "BAAI/bge-small-en"
os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/mnt/model_cache/'
embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en",
                                      model_kwargs=model_kwargs,
                                      encode_kwargs=encode_kwargs
                                     )
 
qdrant_client = QdrantClient(host="59f8f159-fb60-44e8-bfc4-9f35c77ca8d4.us-east4-0.gcp.cloud.qdrant.io",
                             api_key="YTyY8NmWHwAXMwMIWHkERLhpKLffi660UJ0gLbhegPF72EzG9igwUQ")
 
 
 
# App title
st.set_page_config(page_title="UNSDG ChatAssist", layout="wide")
 
 
# App sidebar
with st.sidebar:
    build_sidebar()
 
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you today?"}
    ]
 
# Initialize or re-initialize conversation chain
if "conversation" not in st.session_state.keys() or len(st.session_state.messages) <= 1:
    rag_llm = ChatOpenAI(temperature=0, 
                            model='gpt-3.5-turbo-0613',
                            openai_api_key="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")
    st.session_state.conversation = ConversationChain(
        llm=rag_llm,
        memory=ConversationSummaryMemory(llm=rag_llm),
        verbose=True
    )
 
# And display all stored chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
 
# Seek new input prompts from user
if prompt := st.chat_input("Chat with UNSDG Chat Assist"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
 
 
# Get relevant docs through vector DB
def get_relevant_docs(user_input):
 
    embedded_query = embeddings.embed_query(user_input)
 
    relevant_docs = qdrant_client.search(
            collection_name="unsdg",
            query_vector=embedded_query,
            limit=NUM_TEXT_MATCHES,
    )
 
    urls = [result.payload['metadata']['url'] for result in relevant_docs if result.score > SIMILARITY_THRESHOLD]
    titles = [result.payload['metadata']['Report title'] for result in relevant_docs if result.score > SIMILARITY_THRESHOLD]
    contexts = [result.payload['page_content'] for result in relevant_docs if result.score > SIMILARITY_THRESHOLD]
 
    return urls, titles, contexts
 
 
def build_system_prompt(user_input):
 
    urls, titles, contexts = get_relevant_docs(user_input)
 
    # Create prompt
    template = """ You are a virtual assistant for UNSDG and your task is to answer questions related to UNSDG which includes general information about UNSDG.
 
                    Do not hallucinate. If you don't find an answer, you can point user to the official website here: https://sdgs.un.org/goals . 
 
                    In your response, include the following url links at the end of your response {url_links} and any other relevant URL links that you refered.
 
                    Also, at the end of your response, ask if your response was helpful". 
 
                    Here is some relevant context: {context}"""
 
    prompt_template = PromptTemplate(
        input_variables=["url_links", "context"],
        template=template
    )
    system_prompt = prompt_template.format( url_links=urls, context=contexts)
 
    return system_prompt
 
# Query the Open AI Model
def queryAIModel(user_input):
 
    system_prompt = build_system_prompt(user_input)            
    messages = [
        SystemMessage(
            content=system_prompt
        ),
        HumanMessage(
            content=user_input
        ),
    ]
    output = st.session_state.conversation.predict(input=messages)
 
    return output
 
 
# Function for generating LLM response
def generate_response(prompt):
    response_generated = queryAIModel(prompt)
    return response_generated
 
 
# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(st.session_state.messages[-1]["content"])
            st.write(response)
 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
