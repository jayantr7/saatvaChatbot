from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from flask import make_response
from chatbot_logic import chat_with_chatbot

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
@app.route('/chatbot', methods=['POST'])
def chatbot():
    conversation = request.json['conversation']
    print("Received:", conversation)  # Debugging line
    response_text = chat_with_chatbot(conversation)
    print("Sending:", response_text)  # Debugging line

    response = make_response(jsonify({'response': response_text}), 200)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Credentials"] = "true"

    return response

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5030))
    app.run(debug=True, host='0.0.0.0', port=port)
    
    
