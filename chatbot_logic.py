import openai
import configparser

# Load API key from config.ini
config = configparser.ConfigParser()
try:
    config.read('./saatvaChatbot/config.ini')
except Exception as e:
    print("An error occurred:", e)

openai.api_key = config['openai']['api_key']

def chat_with_chatbot(conversation, currentURL, currScreenHTMLContent):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content":str(conversation + " Current URL: " + currentURL)}
                ],
            max_tokens=50,
            temperature=0.3,
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error: {e}")
        return None
    
    
    # For testing purposes, a simple example response is provided
    response = "Current URL is: " + str(currentURL)
    
    # Save screenGrab to a file
    with open('screen_grab.txt', 'w') as f:
        f.write(currScreenHTMLContent)

    return response