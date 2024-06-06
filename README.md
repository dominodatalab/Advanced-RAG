# Advanced RAG Demo

Base Environment
```
nvcr.io/nvidia/pytorch:23.10-py3
```

If using any IDE other than Jupyter Lab (VSCode), only need the following docker file:

```
# System-level dependency injection runs as root
USER root:root

# Validate base image pre-requisites
# Complete requirements can be found at
# https://docs.dominodatalab.com/en/latest/user_guide/a00d1b/automatic-adaptation-of-custom-images/#_pre_requisites_for_automatic_custom_image_compatibility_with_domino
RUN /opt/domino/bin/pre-check.sh

# Configure /opt/domino to prepare for Domino executions
RUN /opt/domino/bin/init.sh

# Validate the environment
RUN /opt/domino/bin/validate.sh

RUN pip install -q -U trl>=0.7.1 transformers>=4.35.0 accelerate>=0.23.0 peft>=0.6.0 autoawq>=0.1.6 \
datasets>=2.14.5 bitsandbytes>=0.41.1 einops>=0.6.1 evaluate>=0.4.0 langchain-anthropic langchain-pinecone \
Flask Flask-Compress Flask-Cors jsonify uWSGI \
langchain==0.1.8 langchain-openai==0.0.5 langchain-experimental==0.0.52 sentence-transformers==2.3.1 ragatouille \
ipywidgets langchainhub apify-client chromadb tiktoken SQLAlchemy==2.0.1 qdrant-client mlflow[genai] \
presidio-analyzer presidio-anonymizer spacy Faker streamlit spacy pinecone-client dominodatalab-data==5.10.0.dev2


RUN pip uninstall --yes transformer-engine
RUN pip uninstall -y apex
```
```
jupyterlab:
  title: "JupyterLab"
  iconUrl: "/assets/images/workspace-logos/jupyterlab.svg"
  start: ["/opt/domino/bin/jupyterlab-start.sh"]
  httpProxy:
    internalPath: "/{{ownerUsername}}/{{projectName}}/{{sessionPathComponent}}/{{runId}}/{{#if pathToOpen}}tree/{{pathToOpen}}{{/if}}"
    port: 8888
    rewrite: false
    requireSubdomain: false
vscode:
 title: "vscode"
 iconUrl: "/assets/images/workspace-logos/vscode.svg"
 start: [ "/opt/domino/bin/vscode-start.sh" ]
 httpProxy:
    port: 8888
    requireSubdomain: false
```
(Temporary for Domino 5.11)

If running in JupyterLab, add the following snippet to add the Advanced RAG Env. Add to the environment definition after the validate.sh script:

Once in JupyterLab, be sure to select "Advanced RAG Env" kernel in the Jupyter Notebook.

```
# System-level dependency injection runs as root
USER root:root

# Validate base image pre-requisites
# Complete requirements can be found at
# https://docs.dominodatalab.com/en/latest/user_guide/a00d1b/automatic-adaptation-of-custom-images/#_pre_requisites_for_automatic_custom_image_compatibility_with_domino
RUN /opt/domino/bin/pre-check.sh

# Configure /opt/domino to prepare for Domino executions
RUN /opt/domino/bin/init.sh

# Validate the environment
RUN /opt/domino/bin/validate.sh

USER domino
RUN /usr/bin/python -m pip install --user ipykernel && \
/usr/bin/python -m ipykernel install --user --name test_kernel --display-name 'Advanced RAG Env'

RUN pip install -q -U trl>=0.7.1 transformers>=4.35.0 accelerate>=0.23.0 peft>=0.6.0 autoawq>=0.1.6 \
datasets>=2.14.5 bitsandbytes>=0.41.1 einops>=0.6.1 evaluate>=0.4.0 langchain-anthropic langchain-pinecone \
Flask Flask-Compress Flask-Cors jsonify uWSGI \
langchain==0.1.8 langchain-openai==0.0.5 langchain-experimental==0.0.52 sentence-transformers==2.3.1 ragatouille \
ipywidgets langchainhub apify-client chromadb tiktoken SQLAlchemy==2.0.1 qdrant-client mlflow[genai] \
presidio-analyzer presidio-anonymizer spacy Faker streamlit spacy pinecone-client dominodatalab-data==5.10.0.dev2

RUN pip uninstall --yes transformer-engine
RUN pip uninstall -y apex
```


## Required Project Environment Variables

```
ANTHROPIC_API_KEY
APIFY_API_TOKEN	(Optional, only for web scraping)
HUGGING_FACE_HUB_TOKEN
LANGCHAIN_HUB_API_KEY
llama_guard_api_key
OPENAI_API_KEY
PINECONE_API_ENV
PINECONE_API_KEY
QDRANT_KEY
```
