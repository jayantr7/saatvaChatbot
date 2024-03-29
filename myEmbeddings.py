## CODE MODIFIED FROM THIS TUTORIAL: https://platform.openai.com/docs/tutorials/web-qa-embeddings

################################################################################
### Step 1
################################################################################

import configparser
import requests
import re
import urllib.request
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
import os
import pandas as pd
import tiktoken
import openai
import numpy as np
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
from ast import literal_eval
import spacy
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# Regex pattern to match a URL
#HTTP_URL_PATTERN = r'^https://www\.saatva\.com(/(mattresses|furniture|bedding)/.*|/?)$'
HTTP_URL_PATTERN = r'^https://www\.saatva\.com/furniture/minori/$'

# Load API key from config.ini
config = configparser.ConfigParser()
try:
    config.read('config.ini')
except Exception as e:
    print("An error occurred:", e)

openai.api_key = config['openai']['api_key']
os.environ["OPENAI_API_KEY"] = config['openai']['api_key']

# Define root domain to crawl
domain = "saatva.com"
full_url = "https://www.saatva.com"

# Create a class to parse the HTML and get the hyperlinks
class HyperlinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        # Create a list to store the hyperlinks
        self.hyperlinks = []

    # Override the HTMLParser's handle_starttag method to get the hyperlinks
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)

        # If the tag is an anchor tag and it has an href attribute, add the href attribute to the list of hyperlinks
        if tag == "a" and "href" in attrs:
            self.hyperlinks.append(attrs["href"])

################################################################################
### Step 2
################################################################################

# Function to get the hyperlinks from a URL
def get_hyperlinks(url):
    
    # Try to open the URL and read the HTML
    try:
        # Open the URL and read the HTML
        with urllib.request.urlopen(url) as response:

            # If the response is not HTML, return an empty list
            if not response.info().get('Content-Type').startswith("text/html"):
                return []
            
            # Decode the HTML
            html = response.read().decode('utf-8')
    except Exception as e:
        print(e)
        return []

    # Create the HTML Parser and then Parse the HTML to get hyperlinks
    parser = HyperlinkParser()
    parser.feed(html)

    return parser.hyperlinks

################################################################################
### Step 3
################################################################################

# Function to get the hyperlinks from a URL that are within the same domain
def get_domain_hyperlinks(local_domain, url):
    clean_links = []
    for link in set(get_hyperlinks(url)):
        clean_link = None

        # Skip the main domain
        if link == "https://" + local_domain or link == "https://" + local_domain + "/":
            continue

        # If the link is a URL, check if it is within the same domain
        if re.match(HTTP_URL_PATTERN, link):
            # Parse the URL and check if the domain is the same
            url_obj = urlparse(link)
            if url_obj.netloc == local_domain:
                clean_link = link

        # If the link is not a URL, check if it is a relative link
        else:
            if link.startswith("/furniture/minori/"):
                clean_link = "https://" + local_domain + link

        if clean_link is not None:
            if clean_link.endswith("/"):
                clean_link = clean_link[:-1]
            clean_links.append(clean_link)

    return list(set(clean_links))




################################################################################
### Step 4
################################################################################

def crawl(url):
    # Parse the URL and get the domain
    local_domain = urlparse(url).netloc

    # Create a queue to store the URLs to crawl
    queue = deque([url])

    # Create a set to store the URLs that have already been seen (no duplicates)
    seen = set([url])

    # Create a directory to store the text files
    if not os.path.exists("text/"):
            os.mkdir("text/")

    if not os.path.exists("text/"+local_domain+"/"):
            os.mkdir("text/" + local_domain + "/")

    # Create a directory to store the csv files
    if not os.path.exists("processed"):
            os.mkdir("processed")

    # While the queue is not empty, continue crawling
    while queue:

        # Get the next URL from the queue
        url = queue.pop()
        print(url) # for debugging and to see the progress
        
        # Try extracting the text from the link, if failed proceed with the next item in the queue
        try:
            # Save text from the url to a <url>.txt file
            with open('text/'+local_domain+'/'+url[8:].replace("/", "_") + ".txt", "w", encoding="UTF-8") as f:
                html_content = requests.get(url).text
                # Get the text from the URL using BeautifulSoup
                soup = BeautifulSoup(html_content, "html.parser")
                clean_html_content = re.sub(r'<.*?>|function|var|return|for|while|if|else|document|window|class|style|[^\w\s.,!?;]', ' ', html_content)
                #clean_html_content_2 = re.sub(r'<(/?)(html|body|head|link|meta|script|style|div|span|p|br|h[1-6]|ul|ol|li|a|img|input|button|form|nav|footer|header|table|tr|td|th|em|strong|i|b|u|s|del|ins|q|code|pre|blockquote|dl|dt|dd|hr|small|sub|sup|big|address|tt|kbd|samp|var|cite|dfn|abbr|acronym|fieldset|legend|optgroup|option|textarea|label|select|caption|colgroup|col|area|map|dir|base|basefont|applet|param|object|!--|font|noframes|center|noframe|iframe|xml|rss|![CDATA[|article|aside|details|figcaption|figure|time|mark|ruby|rt|rp|section|summary|wbr|command|datagrid|datalist|keygen|output|progress|meter|audio|video|source|canvas|track|datalist|article|main|figure|picture|figcaption|details|menu|menuitem|summary|content|decorator|element|shadow|template|acronym|applet|bgsound|blink|dir|frameset|frame|noframes|isindex|keygen|listing|xmp|nextid|noembed|plaintext|rb|strike|basefont|big|blink|center|font|marquee|multicol|nobr|spacer|tt|rtc)--?>|<!--.*?-->|function|var|let|const|return|for|while|do|switch|case|break|continue|if|else|else if|true|false|null|undefined|NaN|Infinity|eval|parseInt|parseFloat|isNaN|isFinite|decodeURI|decodeURIComponent|encodeURI|encodeURIComponent|escape|unescape|Object|Array|String|Number|Boolean|Symbol|Date|Promise|RegExp|Error|EvalError|RangeError|ReferenceError|SyntaxError|TypeError|URIError|Math|JSON|console|window|document|navigator|history|location|screen|alert|confirm|prompt|print|setTimeout|setInterval|clearTimeout|clearInterval|requestAnimationFrame|cancelAnimationFrame|fetch|XMLHttpRequest|ActiveXObject|Worker|SharedWorker|navigator|Blob|FileReader|localStorage|sessionStorage|indexedDB|caches|applicationCache|crypto|performance|customElements|style|cssText|length|parentRule|getPropertyValue|removeProperty|getPropertyPriority|setProperty|getPropertyCSSValue|[@!?$%^&*=~\|<>]+','', html_content)

                # Get the text but remove the tags
                text = soup.get_text()

                # If the crawler gets to a page that requires JavaScript, it will stop the crawl
                #if ("You need to enable JavaScript to run this app." in text):
                #    print("Unable to parse page " + url + " due to JavaScript being required")
            
                # Otherwise, write the text to the file in the text directory
                combined_content = f"{url}---HTML Content---\n{clean_html_content}\n---Text Content---\n{text}"
                f.write(combined_content)
        except Exception as e:
            print("Unable to parse page " + url)
            print(e)

        # Get the hyperlinks from the URL and add them to the queue
        for link in get_domain_hyperlinks(local_domain, url):
            if link not in seen:
                queue.append(link)
                seen.add(link)

crawl(full_url)

################################################################################
### Step 5
################################################################################

def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie


################################################################################
### Step 6
################################################################################

# Create a list to store the text files
texts=[]

# Get all the text files in the text directory
for file in os.listdir("text/www." + domain + "/"):

    # Open the file and read the text
    with open("text/www." + domain + "/" + file, "r", encoding="UTF-8") as f:
        text = f.read()

        # Omit the first 11 lines and the last 4 lines, then replace -, _, and #update with spaces.
        texts.append((file[11:-4].replace('-',' ').replace('_', ' ').replace('#update',''), text))

# Create a dataframe from the list of texts
df = pd.DataFrame(texts, columns = ['fname', 'text'])

# Set the text column to be the raw text with the newlines removed
df['text'] = df.fname + ". " + remove_newlines(df.text)
df.to_csv('processed/scraped.csv')
df.head()

################################################################################
### Step 7
################################################################################

# Load the cl100k_base tokenizer which is designed to work with the ada-002 model
tokenizer = tiktoken.get_encoding("cl100k_base")

df = pd.read_csv('processed/scraped.csv', index_col=0)
df.columns = ['title', 'text']

# Tokenize the text and save the number of tokens to a new column
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

# Visualize the distribution of the number of tokens per row using a histogram
df.n_tokens.hist()

################################################################################
### Step 8
################################################################################

max_tokens = 2000


# Function to split the text into chunks of a maximum number of tokens
def split_into_many(text, max_tokens = max_tokens):
    #nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
    #nlp.max_length = 2000000
    #nlp.add_pipe('sentencizer')
    #doc = nlp(text)
    #sentences = [sent.text for sent in doc.sents]
    
    
    # Split the text into sentences
    sentences = text.split('. ')

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    
    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater 
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of 
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1
        
    # Add the last chunk to the list of chunks
    if chunk:
        chunks.append(". ".join(chunk) + ".")

    return chunks
    

shortened = []

# Loop through the dataframe
for row in df.iterrows():

    # If the text is None, go to the next row
    if row[1]['text'] is None:
        continue

    # If the number of tokens is greater than the max number of tokens, split the text into chunks
    if row[1]['n_tokens'] > max_tokens:
        shortened += split_into_many(row[1]['text'])
    
    # Otherwise, add the text to the list of shortened texts
    else:
        shortened.append( row[1]['text'] )

################################################################################
### Step 9
################################################################################

df = pd.DataFrame(shortened, columns = ['text'])
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
df.n_tokens.hist()

################################################################################
### Step 10
################################################################################

# Note that you may run into rate limit issues depending on how many files you try to embed
# Please check out our rate limit guide to learn more on how to handle this: https://platform.openai.com/docs/guides/rate-limits

df['embeddings'] = df.text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
df.to_csv('processed/embeddings.csv')
df.head()

################################################################################
### Step 11
################################################################################

df=pd.read_csv('processed/embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(literal_eval).apply(np.array)

df.head()

################################################################################
### Step 12
################################################################################

def create_context(
    question, df, max_len=1800, size="ada"
):
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

    # Return the context
    return "\n\n###\n\n".join(returns)

def answer_question(
    df,
    model="text-davinci-003",
    question=" ",
    max_len=2000,
    size="ada",
    debug=False,
    max_tokens=2000,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
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
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""

################################################################################
### Step 13
################################################################################

#print(answer_question(df, question="What day is it?", debug=False))

print(answer_question(df, question="What is the best mattress? What price ranges?"))