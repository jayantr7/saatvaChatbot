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

llm = OpenAI()
memory = ConversationBufferMemory()
conversation = ConversationChain(
   llm=llm,
   memory=memory,
   verbose=False
)

urlsVisited = []

memoryInitialization = []

# Initialize the global memory object
memory = ConversationBufferMemory()

# Define ConversationBufferMemory class
class ConversationBufferMemory:
    def __init__(self):
        self.conversation = []
        
    def add_message(self, sender, message):
        self.conversation.append({'sender': sender, 'message': message})
        
    def get_conversation(self):
        return self.conversation

# Entry point to the chatbot logic
def chat_with_chatbot(userPrompt, currentURL, currScreenHTMLContent):
    if len(memoryInitialization) == 0:
        # Initialize the global memory object
        memory = ConversationBufferMemory()
        memoryInitialization.append("something")
    else:
        pass
    # Function to extract text from HTML
    def extractTextFromHTML():
        with open('screen_grab.txt', 'r') as file:
            html_content = file.read()
        soup = BeautifulSoup(html_content, 'html.parser')
        for script_tag in soup(['script', 'style']):
            script_tag.extract()
        text_content = soup.stripped_strings
        final_text = ' '.join(text_content)
        with open('output_of_screen_grab.txt', 'w') as file:
            file.write(final_text)
        print("Text extraction complete.")

    # Log every URL that the function is being called for
    if currentURL not in urlsVisited:
        urlsVisited.append(currentURL)
        
        # Extract text from HTML
        extractTextFromHTML()
    
    # Load embeddings
    df = pd.read_csv('./processed/embeddings.csv', index_col=0)
    df['embeddings'] = df['embeddings'].apply(literal_eval).apply(np.array)

    # Save the user's message to memory
    memory.add_message('user', userPrompt)
    
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
        max_tokens=500,
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
                prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
                temperature=0,
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
        
    answer = answer_question(df, question=userPrompt, debug=False)
    
    # Save the bot's answer to memory
    memory.add_message('bot', answer)
    
    return answer


