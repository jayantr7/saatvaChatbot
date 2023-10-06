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

'''chat_with_chatbot() is the entry point to the chatbot logic'''
def chat_with_chatbot(userPrompt, currentURL, currScreenHTMLContent):
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
    def output_first_half_of_file(file_path):
        # Step 1: Open the file
        with open(file_path, 'r') as f:
            # Step 2: Read the content
            content = f.read()
            
        # Step 3: Find the halfway point
        halfway_point = len(content) // 2
        
        # Step 4: Slice the string to get the first half
        first_half = content[:halfway_point]
        
        # Step 5: Output the result
        with open("out_of_screen_grab.txt", "w") as f:
            f.write(first_half)

    file_path = "output_of_screen_grab.txt"
    output_first_half_of_file(file_path)
    # Extract text from HTML
    extractTextFromHTML()

    # Load the document and split it into chunks
    data = TextLoader('output_of_screen_grab.txt').load()
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=2500, chunk_overlap=40)
    docs = text_splitter.split_documents(data)
    
    # Create OpenAI embeddings and a Chroma vector database
    openai_embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(documents=docs, embedding=openai_embeddings)
    
    # Set up retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    
    # Initialize the ChatOpenAI model
    llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k-0613')
    
    # Create a RetrievalQA chain
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    
    # Generate and return the response
    response = qa(userPrompt)
    answer  = response["result"]
    
    return answer

### PROGRESS: currently taking the output grab and halving it, then training on that.