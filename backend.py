from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from flask import make_response
#from chatbot_logic import chat_with_chatbot


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

##########FROM CHATBOT_LOGIC.PY (BROUGHT HERE TO AVOID HASSLE FOR THE TIME BEING)################################
#################################################################################################################
#################################################################################################################
#################################################################################################################
### GOAL: take in input and send out response ###
###       INPUT: chat_with_chatbot(userPrompt, currentURL, currScreenHTMLContent) is called from the outside ###
###       OUTPUT: a string is sent to the caller (bot reply) ###
###       However I change the code, I need to keep this in mind ###
import openai
import configparser
import os
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from bs4 import BeautifulSoup
from langchain.document_loaders import TextLoader

import pandas as pd
from ast import literal_eval
import numpy as np
import openai
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity


######################################
# Load API key from config.ini
config = configparser.ConfigParser()
try:
    config.read('config.ini')
except Exception as e:
    print("An error occurred:", e)

openai.api_key = config['openai']['api_key']
os.environ["OPENAI_API_KEY"] = config['openai']['api_key']

llm = OpenAI(model_name="gpt-4")
memory = ConversationBufferMemory()
conversation = ConversationChain(
   llm=llm,
   memory=memory,
   verbose=False
)

#urlsVisited = []
messagesToResponder = [
            {"role":"system", "content": "You take the response from a text embedding bot (text-davinci-002) that is answering a question that the Saatva ecommerce website user has and entering in a chatbot. The text embedding response is more technical and weird. So you act as the middleman to craft a better sounding reply to the user's query."},
            {"role":"system", "content": "So the user in this context is the text embedding bot that is taking user's query and sending out a response."},
            {"role":"system", "content": "The user role replies where the message starts from \'CUSTOMER:\' is the actual customer's prompt or question. The user role replies where the message starts from \'DAVINCI:\' is the text embedding bot's reply. If the bot's response doesn't answer the question properly or wholly, use your discretion to fill in the blanks. To the customer, the responses look like they come from you. So don't write \'DAVINCI:\' at the start of your response."},
            {"role":"system", "content": "Whether you should be brief or answer in detail, use your discretion. But remember that you are a chatbot. So don't beat around the bush too much. If the customer tries to ask questions about something other than the relevant topics at hand, guide them to the topic gently."},
            {"role":"system", "content": "The first message you sent is \"Hi! I am SaatvaAI, a friendly shopping assistant for the Saatva product catalogue. Feel free to ask me anything about our products.\". This is meant as a greeting to greet the customer when they and the chatbot start talking. Be friendly. Use your discretion to decide about the 'fun' ascept."}
            ]
messagesToDavinci = [
            {"role":"system", "content": "You are a sophisticated language processing bot designed to act as an intermediary between customers/users and a text-davinci embedding bot which handles a comprehensive dataset of Saatva's product offerings including mattresses, bedding, and furniture. That davinci bot is the one that actually queries the dataset. Your primary objective is to receive queries from customers regarding these products, and enhance or modify these queries to ensure they are well-structured, precise, and tailored to extract the most relevant and accurate information from the text-davinci bot. Your response goes as the input prompt to the davinci bot. To it, your message should look it came from the user. But your job is to enhance the question in such a way that extracts the most information from the embedding dataset about whatever is being asked. Your translations should adhere to a structured format that aligns with the attributes and categories of the products in the dataset, and should be capable of handling a variety of inquiries including product specifications, comparisons, pricing, and promotions. The ultimate goal is to provide a seamless and informative user experience, delivering accurate and insightful responses to user queries by effectively interfacing with the text-davinci bot."},
            {"role":"system", "content": "In case of ambiguous or unclear queries, structure the query to request further clarification or provide a list of available options based on the partial information provided by the user."},
            {"role":"system", "content": "Extract detailed product features from the embeddings based on user queries. For instance, if a user is interested in the cooling features of a mattress, ensure the query to the text-davinci bot is structured to retrieve information on cooling technology or materials used in the mattresses."},
            {"role":"system", "content": "If a user inquires about ongoing sales or promotions, ensure the query is structured to retrieve any available sales or discount information."},
            {"role":"system", "content": "Handle requests for comparisons by structuring queries to extract comparative information on specified attributes between different products or product categories."},
            {"role":"system", "content": "Translate user queries into structured requests specifying product attributes such as material, size, firmness, price, or special features. For instance, if a user asks about the firmness options for mattresses, translate it into a request for 'firmness options' under the category of 'mattresses'."},
            {"role":"system", "content": "Identify and understand the product category in user queries, such as mattresses, bedding, or furniture, and translate it into a structured request for category-specific information."},
            {"role":"system", "content": "The user enters to this bot - you. You morph the query in a way that looks like it came directly from the user, but in a more description and well-asked manner. The customer cannot be relied upon to ask precise questions. Your response is then set to the embedding davinci bot that searches that dataset based on whatever you respond here, thinking the customer is directly querying it. The davinci bot then sends its response to another GPT chat bot to enhance its answers to sound more pleasing. That is what the final response seen by the user is."},
            {"role":"system", "content": "The user role replies where the message starts from \'CUSTOMER:\' is the actual customer's prompt or question. The assistant role replies where the message starts from \'FINAL-ANSWER:\' is the final response that was sent to the customer to see."},
            {"role":"system", "content": "Don't write 'FINAL-ANSWER:' in your response to the davinci bot."},
            ]

# Define ConversationBufferMemory class
class ConversationBufferMemory:
    def __init__(self):
        self.conversation = []
        
    def add_message(self, sender, message):
        self.conversation.append({'sender': sender, 'message': message})
        
    def get_conversation(self):
        return self.conversation

# Initialize the global memory object
memory = ConversationBufferMemory()

# Entry point to the chatbot logic
def chat_with_chatbot(userPrompt, currentURL):
    userPrompt = userPrompt + currentURL
    # Function to extract text from HTML
    # def extractTextFromHTML():
    #     with open('screen_grab.txt', 'r') as file:
    #         html_content = file.read()
    #     soup = BeautifulSoup(html_content, 'html.parser')
    #     for script_tag in soup(['script', 'style']):
    #         script_tag.extract()
    #     text_content = soup.stripped_strings
    #     final_text = ' '.join(text_content)
    #     with open('output_of_screen_grab.txt', 'w') as file:
    #         file.write(final_text)
    #     print("Text extraction complete.")

    # # Log every URL that the function is being called for
    # if currentURL not in urlsVisited:
    #     urlsVisited.append(currentURL)
        
    #     # Extract text from HTML
    #     extractTextFromHTML()
    
    # Load embeddings
    df = pd.read_csv('./processed/embeddings.csv', index_col=0)
    df['embeddings'] = df['embeddings'].apply(literal_eval).apply(np.array)
    
    ###### QUERY BOT LOGIC ##########
    def chatgpt_call_with_memory_0(messagesToDavinci, model="gpt-4"):
        queryToEmbeddingBot = openai.ChatCompletion.create(
            model=model,
            messages=messagesToDavinci
        )
        return queryToEmbeddingBot.choices[0].message["content"]
    #################################
    messagesToDavinci.append({"role":"user", "content":f"CUSTOMER: {userPrompt}"})
    # Now, ask the embedding bot the question
    answerFromQueryBot = chatgpt_call_with_memory_0(messagesToDavinci)

    # Save the user's message to memory
    memory.add_message('user', answerFromQueryBot)
    
    # Create context function
    def create_context(question, df, max_len=1800, size="ada"):
        """
        Create a context for a question by finding the most similar context from the dataframe
        """

        # Get the embeddings for the question
        q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

        # Get the distances from the embeddings
        df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


        returns = []
        cur_len = 0

        # Sort by distance and add the text to the context until the context is too long
        for i, row in df.sort_values('distances', ascending=True).iterrows():
            
            # Add the length of the text to the current length
            cur_len += row['n_tokens'] + 4
            
            # If the context is too long, break
            if cur_len > max_len:
                break
            
            # Else add it to the text that is being returned
            returns.append(row["text"])
        
        # Get the current conversation from memory
        conversation_context = memory.get_conversation()
        
        # Prepend conversation context to the returned context
        conversation_text = "\n".join([f"{entry['sender']}: {entry['message']}" for entry in conversation_context])
        return conversation_text + "\n\n###\n\n" + "\n\n###\n\n".join(returns)
        
    # Answer question function
    def answer_question(
        df,
        model="text-davinci-003",
        question=" ",
        max_len=1800,
        size="ada",
        debug=False,
        max_tokens=1000,
        stop_sequence=None
    ):
        context = create_context(
            question,
            df,
            max_len=max_len,
            size=size,
        )
        # If debug, print the raw model response
        if debug:
            print("Context:\n" + context)
            print("\n\n")

        try:
            # Create a completions using the questin and context
            response = openai.Completion.create(
                prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:. There is a chatbot on a furniture bedding mattresses selling ecommerce website. The customer enters questions or queries to it. It is passed to a GPT API bot that rephrases the question to suit your API model. Your reply is passed to another GPT Chat API bot that rephrases whatever you write in a more natural sounding manner. So focus on replying with a lot of pertinent information and associated keywords.",
                temperature=0.1,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop_sequence,
                model=model,
            )
        except Exception as e:
            print(e)
            return ""
        
        return response["choices"][0]["text"].strip()
        
    answerFromDavinci = answer_question(df, question=answerFromQueryBot, debug=False)
    
    # Save the bot's answer to memory
    memory.add_message('bot', answerFromDavinci)
    
    ###### RESPONDER BOT LOGIC ######
    def chatgpt_call_with_memory(messagesToResponder, model="gpt-4"):
        responseFromResponder = openai.ChatCompletion.create(
            model=model,
            messages=messagesToResponder
        )
        return responseFromResponder.choices[0].message["content"]
    #################################
    
    messagesToResponder.append({"role":"user", "content":f"CUSTOMER: {userPrompt}"})
    messagesToResponder.append({"role":"user", "content":f"DAVINCI: {answerFromDavinci}"})
    answer = chatgpt_call_with_memory(messagesToResponder)
    messagesToResponder.append({"role":"assistant", "content":f"{answer}"})
    messagesToDavinci.append({"role":"assistant", "content":f"FINAL-ANSWER: {answer}"})
    print("QUERY BOT TO DAVINCI: ", answerFromQueryBot, "\n")
    print("DAVINCI: ", answerFromDavinci, "\n")
    return answer
##########################################
##########################################
##########################################
##########################################
@app.route('/chatbot', methods=['POST'])
def chatbot():
    conversation = request.json['conversation']
    activeTabURL = request.json['url']
    #screenGrab = request.json['screenGrab']
    print("Received conversation:", conversation)  # Debugging line
    print("Received URL:", activeTabURL)  # Debugging line
    #print("Received screen grab of the active page (first 15 chars):", screenGrab[:15])
    response_text = chat_with_chatbot(conversation, activeTabURL)
    print("Sending:", response_text)  # Debugging line

    response = make_response(jsonify({'response': response_text}), 200)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Credentials"] = "true"

    return response

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5030))
    app.run(debug=True, host='0.0.0.0', port=port)
    
    
