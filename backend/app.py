from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from chatbot import get_response_from_bedrock  # Import the function


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")


app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    response = get_response_from_bedrock(msg)  # Use Bedrock AI for response
    return jsonify({"response": response})


""" def get_Chat_response(text):

    search_results = vs.similarity_search(user_input, k=3)
    context_string = '\n\n'.join([f'Document {ind+1}: ' + i.page_content for ind, i in enumerate(search_results)])
    prompt_data = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE).format(human_input=user_input, context=context_string)
    
    # Generate response using Bedrock AI
    llm = Bedrock(client=boto3_bedrock, model_id="anthropic.claude-v2", model_kwargs={"max_tokens_to_sample": 500, "temperature": 0.9})
    output = llm(prompt_data).strip() """


if __name__ == '__main__':
    app.run()