import openai
import configparser
import os
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.chains import ConversationChain


# Load API key from config.ini
config = configparser.ConfigParser()
try:
    config.read('./saatvaChatbot/config.ini')
except Exception as e:
    print("An error occurred:", e)

openai.api_key = config['openai']['api_key']
os.environ["OPENAI_API_KEY"] = config['openai']['api_key']

context = [{'role':'system', 'content':'You are the SaatvaAI Chatbot. Your job is to help users about the saatva.com website. Be concise, use as few tokens as possible, be friendly.'},
            ]

llm = OpenAI()
memory = ConversationBufferMemory()
conversation = ConversationChain(
   llm=llm,
   memory=memory,
   verbose=False
)
    
def call_LLM_with_memory(context):
   response = openai.ChatCompletion.create(
       model='gpt-3.5-turbo',
       messages=context,
       max_tokens=50,
       temperature=0.3,
   )
   return response.choices[0].message["content"]

'''chat_with_chatbot() is the entry point to the chatbot logic'''
def chat_with_chatbot(userPrompt, currentURL, currScreenHTMLContent):
    response = conversation.predict(input=str(userPrompt))
    
    #context.append({'role':'user', 'content':f"{userPrompt}"})
    #response = call_LLM_with_memory(context)
    #context.append({'role':'assistant', 'content':f"{response}"})

    # Save screenGrab to a file
    with open('screen_grab.txt', 'w') as f:
        f.write(currScreenHTMLContent)
    
    finalReply = response
    return finalReply
