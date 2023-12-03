import boto3
import os
import json
import numpy as np
from pathlib import Path
from pprint import pprint

import sagemaker
from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader
from langchain.embeddings import BedrockEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.llms import Bedrock
from IPython.display import Markdown, display
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

region = os.environ.get("AWS_REGION")
boto3_bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1',
)

def embed_text_input(bedrock_client, prompt_data, modelId="amazon.titan-embed-text-v1"):
    accept = "application/json"
    contentType = "application/json"
    body = json.dumps({"inputText": prompt_data})
    response = bedrock_client.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())
    embedding = response_body.get("embedding")
    return np.array(embedding)

# Function to get response from Bedrock AI
def get_response_from_bedrock(user_input):
    # Perform similarity search
    search_results = vs.similarity_search(user_input, k=3)
    context_string = '\n\n'.join([f'Document {ind+1}: ' + i.page_content for ind, i in enumerate(search_results)])
    prompt_data = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE).format(human_input=user_input, context=context_string)
    
    # Generate response using Bedrock AI
    llm = Bedrock(client=boto3_bedrock, model_id="anthropic.claude-v2", model_kwargs={"max_tokens_to_sample": 500, "temperature": 0.9})
    output = llm(prompt_data).strip()

    return output

## user_input = "What are Jack Sparrow's interests?"
## document_1 = 'his interests are hiking and playing video games'
## document_2 = 'his family is Haochen, Lily, and Priscilla'

## user_input_vector = embed_text_input(boto3_bedrock, user_input)
## document_1_vector = embed_text_input(boto3_bedrock, document_1)
## document_2_vector = embed_text_input(boto3_bedrock, document_2)

## doc_1_match_score = np.dot(user_input_vector, document_1_vector)
## doc_2_match_score = np.dot(user_input_vector, document_2_vector)

## print(f'"{user_input}" matches "{document_1}" with a score of {doc_1_match_score:.1f}')
## print(f'"{user_input}" matches "{document_2}" with a score of {doc_2_match_score:.1f}')

with open('/Users/haochenmiao/Documents/Data_Sciecne_Projects/safeGPT/incidents.txt') as f:
    doc_content = f.read()
docs = [Document(page_content=doc_content)]
split_docs = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=0).split_documents(docs)

# Set up embedding model and vector store
hf_embedding_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5", model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': False})
vs = FAISS.from_documents(split_docs, hf_embedding_model)

RAG_PROMPT_TEMPLATE = '''Here is some important context which can help inform the questions the Human asks.
Make sure to not make anything up to answer the question if it is not provided in the context.

<context>
{context}
</context>

Human: {human_input}

Assistant:
'''
PROMPT = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

# Interactive conversation loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Chatbot: Goodbye!")
        break

    # Perform similarity search and generate response
    search_results = vs.similarity_search(user_input, k=3)
    context_string = '\n\n'.join([f'Document {ind+1}: ' + i.page_content for ind, i in enumerate(search_results)])
    prompt_data = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE).format(human_input=user_input, context=context_string)

    # Generate and output response
    llm = Bedrock(client=boto3_bedrock, model_id="anthropic.claude-v2", model_kwargs={"max_tokens_to_sample": 500, "temperature": 0.9})
    output = llm(prompt_data).strip()
    print("Chatbot:", output)