import streamlit as st
import os

import json
import requests
import pandas as pd
import pinecone
import subprocess

from sidebar import build_sidebar
from langchain.embeddings import HuggingFaceBgeEmbeddings
from domino_data.vectordb import DominoPineconeConfiguration
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

from langchain.schema import HumanMessage, SystemMessage
from langchain import PromptTemplate
from langchain.memory import ConversationSummaryMemory

# Number of texts to match (may be less if no suitable match)
NUM_TEXT_MATCHES = 3

# Similarity threshold such that queried text with a lower will be discarded 
# Range [0, 1], larger = more similar for cosine similarity
SIMILARITY_THRESHOLD = 0.83


# Initialize Pinecone index
datasource_name = "Rakuten"
conf = DominoPineconeConfiguration(datasource=datasource_name)
api_key = os.environ.get("DOMINO_VECTOR_DB_METADATA", datasource_name)

pinecone.init(
    api_key=api_key,
    environment="domino",
    openapi_config=conf
)

index = pinecone.Index("rakuten")

# Create embeddings to embed queries

model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embedding_model_name = "BAAI/bge-small-en"
os.environ['SENTENCE_TRANSFORMERS_HOME'] = './model_cache/'
embeddings = HuggingFaceBgeEmbeddings(model_name=embedding_model_name,
                                      model_kwargs=model_kwargs,
                                      encode_kwargs=encode_kwargs
                                     )

# App title
st.set_page_config(page_title="ChatAssist", layout="wide")

# App sidebar
with st.sidebar:
    build_sidebar()

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you today?"}
    ]

# Initialize or re-nitialize conversation chain
if "conversation" not in st.session_state.keys() or len(st.session_state.messages) <= 1:
    chat = ChatOpenAI(temperature=0, 
                        model='gpt-3.5-turbo',
                        openai_api_key=os.environ.get("OPENAI_API_KEY"))
    
    st.session_state.conversation = ConversationChain(
        llm=chat,
        memory=ConversationSummaryMemory(llm=chat),
        verbose=True
    )

# And display all stored chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Seek new input prompts from user
if prompt := st.chat_input("Chat with ChatAssist"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)


# Get relevant docs through vector DB
def get_relevant_docs(user_input):
   
    embedded_query = embeddings.embed_query(user_input)
    
    relevant_docs = index.query(
        vector=embedded_query,
        top_k=NUM_TEXT_MATCHES,
        include_values=True,
        include_metadata=True
    )

    matches = relevant_docs["matches"]
    filtered_matches = [match for match in matches if match['score'] >= SIMILARITY_THRESHOLD]
    relevant_docs["matches"] = filtered_matches

    return relevant_docs


def build_system_prompt(user_input):

    # Retrieve context
    relevant_docs = get_relevant_docs(user_input)
    actual_num_matches = len(relevant_docs["matches"])
    url_links = set([relevant_docs["matches"][i]["metadata"]["url"] for i in range(actual_num_matches)])
    context = [relevant_docs["matches"][i]["metadata"]["text"] for i in range(actual_num_matches)]

    # Create prompt
    template = """ If the user asks a question that is related to Domino Pizza, tell them that go to the following link if they want to order pizza: https://www.dominos.com/en/.
    
                    If the user asks a question that is not related to Domino Data Lab, Domino Pizza, AI, or machine learning, respond with the following keyword: https://www.youtube.com/watch?v=dQw4w9WgXcQ. 
    
                    Otherwise, you are a virtual assistant for Domino Data Lab and your task is to answer questions related to Domino Data Lab which includes general AI/machine learning concepts. 
                    
                    When answering questions, only refer to the {domino_docs_version} version of Domino. Do not use information from other versions of Domino.
                    
                    If you don't find an answer to the question the user asked in the {domino_docs_version} version of Domino, 
                    tell them that you looked into the {domino_docs_version} version of Domino but the feature or capability that they're looking for likely does not exist in that version. 
                    
                    Do not hallucinate. If you don't find an answer, you can point user to the official version of the Domino Data Lab docs here: https://docs.dominodatalab.com/. 
                    
                    In your response, include the following url links at the end of your response {url_links} and any other relevant URL links that you refered.
                    
                    Also, at the end of your response, ask if your response was helpful and to please file a ticket with our support team at this link if further help is needed: 
                    https://tickets.dominodatalab.com/hc/en-us/requests/new#numberOfResults=5, embedded into the words "Support Ticket". 
                    
                    Here is some relevant context: {context}"""

    prompt_template = PromptTemplate(
        input_variables=["domino_docs_version", "url_links", "context"],
        template=template
    )
    system_prompt = prompt_template.format(domino_docs_version=domino_docs_version, url_links=url_links, context=context)
    
    return system_prompt

# Query the Open AI Model
def queryOpenAIModel(user_input):

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
    response_generated = queryOpenAIModel(prompt)
    return response_generated


# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(st.session_state.messages[-1]["content"])
            st.write(response)

    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)