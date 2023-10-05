document.addEventListener("DOMContentLoaded", function() {
  var conversation = document.getElementById("conversation");
  var userInput = document.getElementById("user-input");
  var submitButton = document.getElementById("submit-button");

  userInput.addEventListener("keyup", function(event) {
    if (event.key === 'Enter') {
      event.preventDefault();
      submitButton.click();
    }
  });

  submitButton.addEventListener("click", function() {
    var userMessage = userInput.value;
    conversation.innerHTML += "<p>User: " + userMessage + "</p>";
    userInput.value = "";
    
    // Send message to chatbot and handle response
    chat_with_chatbot(userMessage, function(chatbotResponse) {
      conversation.innerHTML += "<p>Chatbot: " + chatbotResponse + "</p>";
    });
  });

  function chat_with_chatbot(message, callback) {
    fetch("http://127.0.0.1:5030/chatbot", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ conversation: message })
    })
    .then(response => response.json())
    .then(data => callback(data.response))
    .catch((error) => console.error("An error occurred:", error));
  }
});
