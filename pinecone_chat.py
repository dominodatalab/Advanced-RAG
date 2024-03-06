import streamlit as st
import os

import json
import requests
import pandas as pd
import pinecone
import subprocess

from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_experimental.data_anonymizer import PresidioReversibleAnonymizer
from langchain.chains import ConversationChain, HypotheticalDocumentEmbedder, LLMChain
from langchain_community.chat_models import ChatMlflow
from langchain.schema import HumanMessage, SystemMessage
from langchain import PromptTemplate
from langchain.memory import ConversationSummaryMemory
from langchain import hub

from domino_data.vectordb import DominoPineconeConfiguration
from ragatouille import RAGPretrainedModel
from sidebar import build_sidebar

command = "sudo python -m spacy download en_core_web_lg"
process = subprocess.run(command, shell=True, check=True)

llama_guard_api_url = "https://se-demo.domino.tech:443/models/65e3eb9fd69e0f578609eaf8/latest/model"
llama_guard_api_key = os.environ.get('llama_guard_api_key')



anonymizer = PresidioReversibleAnonymizer(
    add_default_faker_operators=False,
    analyzed_fields=["LOCATION","PHONE_NUMBER","US_SSN", "IBAN_CODE", "CREDIT_CARD", "CRYPTO", "IP_ADDRESS",
                    "MEDICAL_LICENSE", "URL", "US_BANK_NUMBER", "US_DRIVER_LICENSE", "US_ITIN", "US_PASSPORT"]
)


def anonymize(input_text):
    if input_text:
        return anonymizer.anonymize(input_text)

    
def get_moderation_result(query,role="Agent"):
 
    response = requests.post(llama_guard_api_url,
        auth=(
            llama_guard_api_key,
            llama_guard_api_key
        ),
        json={
            "data": {"query": query, "role": role}
        }
    )
    return response.json()['result']


# Number of texts to match (may be less if no suitable match)
NUM_TEXT_MATCHES = 5

# Number of texts to return from reranking
NUM_RERANKING_MATCHES = 3

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
# Setup HyDE

hyde_prompt_template = """You are a virtual assistant for Rakuten and your task is to answer questions related to Rakuten which includes general information about Rakuten
"Please answer the user's question below \n 
Question: {question}
Answer:"""
hyde_prompt = PromptTemplate(input_variables=["question"], template=hyde_prompt_template)
chat = ChatMlflow(
    target_uri=os.environ["DOMINO_MLFLOW_DEPLOYMENTS"],
    endpoint="chat-gpt35turbo-sm",
)
hyde_llm_chain = LLMChain(llm=chat, prompt=hyde_prompt)

hyde_embeddings = HypotheticalDocumentEmbedder(
    llm_chain=hyde_llm_chain, base_embeddings=embeddings
)

# Load the reranking model
colbert = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

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
    chat = ChatMlflow(
        target_uri=os.environ["DOMINO_MLFLOW_DEPLOYMENTS"],
        endpoint="chat-gpt35turbo-sm",
    )
    
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
def get_relevant_docs(user_input, num_matches=NUM_TEXT_MATCHES, use_hyde=True):
   
    if use_hyde:
        embedded_query = hyde_embeddings.embed_query(user_input)
    else:
        embedded_query = embeddings.embed_query(user_input)
    
    relevant_docs = index.query(
        vector=embedded_query,
        top_k=num_matches,
        include_values=True,
        include_metadata=True
    )

    matches = relevant_docs["matches"]
    filtered_matches = [match for match in matches if match['score'] >= SIMILARITY_THRESHOLD]
    relevant_docs["matches"] = filtered_matches

    return relevant_docs


def build_system_prompt(user_input, rerank=True, use_hyde=True):
    
    relevant_docs = get_relevant_docs(user_input)
    
    actual_num_matches = len(relevant_docs["matches"])
    urls = set([relevant_docs["matches"][i]["metadata"]["source"] for i in range(actual_num_matches)])
    contexts = [relevant_docs["matches"][i]["metadata"]["text"] for i in range(actual_num_matches)]
    
    if rerank and actual_num_matches >= NUM_RERANKING_MATCHES:
        docs = colbert.rerank(query=user_input, documents=contexts, k=NUM_RERANKING_MATCHES)
        result_indices = [docs[i]["result_index"] for i in range(NUM_RERANKING_MATCHES)]
        contexts = [contexts[index] for index in result_indices]
        urls = [list(urls)[index] for index in result_indices]  

    prompt_template = hub.pull("subirmansukhani/rakuten-qa-rag")
    system_prompt = prompt_template.format( url_links=urls, context=contexts)
 
    return system_prompt

# Query the Open AI Model
def queryOpenAIModel(user_input, use_hyde=True):

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
def generate_response(prompt, anon=True):
    
    if anon:
        prompt = anonymize(prompt)
    response_generated = queryOpenAIModel(prompt)
    return response_generated


# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if "unsafe" in get_moderation_result(prompt,"User"):
                response =  "I am sorry, can you please rephrase that"
            else:
                response = generate_response(st.session_state.messages[-1]["content"])
                if "unsafe" in get_moderation_result(prompt,"Agent"):
                    response =  "I am sorry, I enocuntered an issue and cannot answer this question"
            st.write(response)

    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
