To populate the dataset, you can run myEmbeddings_copy.py

To run the program, install the Chrome extension the usual way. Then run backend.py. Takes some time to start, but it will.

To install the dependencies (quite a few have not really been used), run pip3 install -r requirements.txt

My apologies! The code is not properly refactored.

To follow the logic, see:

THESE ARE THE ONLY RELEVANT FILES. IGNORE THE OTHER FILES:
backend.py: Has the logic to start the server and at the top that chatbot logic. I actually had the chatbot logic in a separate file, but some message passing became bug-ridden, so I moved it in. chat_with_chatbot() is the function that takes in all requests and outputs a response.

sidepanel.html: Code for the frontend UI on the Chrome sidePanel.

manifest.json: Needed to build the Chrome extension

icons PNGs

myEmbeddings_copy.py : running it crawls the saatva website's relavant urls using regex. Then it does tokenisation and embedding processing work. Outputs that to a file. The last part just queries the davinci bot with a single question. Ignore that part. That was from the tutorial (link of the tutorial on top of the file) from which I learned how to do this. This file is a modified version of the official OpenAI tutorial code.

popup.js: Javascript logic for the frontend. Sends data to backend and vice-versa

resize-icon.js: helps in resizing the icon in the sidePanel

popup.html: code for the extension (not sidePanel). Not in use since we are using sidepanel while using the extension automatically.

style.css: CSS stylesheet for the frontend

background.js: some basic listener functions


