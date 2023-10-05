document.addEventListener("DOMContentLoaded", function() {
  var conversation = document.getElementById("conversation");
  var userInput = document.getElementById("user-input");
  var submitButton = document.getElementById("submit-button");

  userInput.addEventListener("keyup", function(event) {
    if (event.keyCode === 13) {
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
    console.log("Sending request to server."); // Debugging line
    
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "http://127.0.0.1:5030/chatbot", true);
    xhr.setRequestHeader("Content-Type", "application/json");
    
    // Events for debugging
    xhr.addEventListener("error", function() {
      console.log("An error occurred while making the request.");
    });
    
    xhr.addEventListener("load", function() {
      if (xhr.status >= 200 && xhr.status < 400) {
        console.log("Request was successful");
      } else {
        console.log("Server responded with a status:", xhr.status);
      }
    });
    
    xhr.onreadystatechange = function() {
      console.log("Ready state changed.", xhr.readyState, xhr.status); // Debugging line
      if (xhr.readyState == 4 && xhr.status == 200) {
        var response = JSON.parse(xhr.responseText);
        callback(response.response);
      }
    };

    xhr.send(JSON.stringify({ conversation: message }));
    
    console.log("Request sent."); // Debugging line
  }
});
