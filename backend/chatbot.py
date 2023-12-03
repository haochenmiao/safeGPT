import os
import boto3
import botocore
import json
import time
import anthropic
from langchain.chains import ConversationChain
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Function to load incidents data
def load_incidents(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def bedrock_chain():
    region = os.environ.get("AWS_REGION")
    bedrock_service = boto3.client(
        service_name='bedrock',
        region_name='us-east-1',
    )

    bedrock_service.list_foundation_models()

    bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name='us-east-1',
    )
    anthropic_llm = Bedrock(
        model_id="anthropic.claude-v2", client=bedrock, credentials_profile_name=region
    )
    anthropic_llm.model_kwargs = {"temperature": 0.5, "max_tokens_to_sample": 700}
    prompt_template = """System: The following is a friendly conversation between a knowledgeable helpful assistant and a customer.
The assistant is talkative and provides lots of specific details from it's context.

Conversation history:
{history}

Current conversation:
User: {input}
Bot:"""
    PROMPT = PromptTemplate(
        input_variables=["history", "input"], template=prompt_template
    )

    memory = ConversationBufferMemory(human_prefix="User", ai_prefix="Bot")
    conversation = ConversationChain(
        prompt=PROMPT,
        llm=anthropic_llm,
        verbose=True,
        memory=memory,
    )

    return conversation

def run_chain(chain, prompt):
    num_tokens = chain.llm.get_num_tokens(prompt)
    return chain({"input": prompt}), num_tokens


def clear_memory(chain):
    return chain.memory.clear()


if __name__ == "__main__":
    incidents_path = "/Users/haochenmiao/Documents/Data_Sciecne_Projects/safeGPT/incidents.json"
    incidents_data = load_incidents(incidents_path)
    chain = bedrock_chain()

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        # Integrate incidents data into the conversation logic here
        resp, _ = run_chain(chain, user_input)
        print("Bot:", resp)