def chat_with_chatbot(conversation, currentURL, currScreenHTMLContent):
    # For testing purposes, a simple example response is provided
    response = "Current URL is: " + str(currentURL)
    
    # Save screenGrab to a file
    with open('screen_grab.txt', 'w') as f:
        f.write(currScreenHTMLContent)

    return response