document.addEventListener("DOMContentLoaded", function() {
  console.log("DOM Content Loaded");  // Debugging line
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
    console.log("Submit button clicked");  // Debugging line
    var userMessage = userInput.value;
    conversation.innerHTML += `<div class="user-message"><span class="user-label"><strong>User:</strong></span> ${userMessage}</div>`;
    userInput.value = "";
  
    // Scroll to the bottom
    conversation.scrollTop = conversation.scrollHeight;
  
    // Create and append typing indicator to conversation
    var typingIndicator = document.createElement('div');
    typingIndicator.id = 'typing-indicator';
    typingIndicator.innerHTML = '<span><strong>SaatvaAI:</strong> is typing...</span>';
    conversation.appendChild(typingIndicator);
  
    // Scroll to show typing indicator
    conversation.scrollTop = conversation.scrollHeight;
  
    // Get the URL of the active tab
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
      console.log("Inside chrome.tabs.query");  // Debugging line
      var activeTab = tabs[0];
      var activeTabURL = activeTab.url;
      console.log("Active tab URL:", activeTabURL);  // Debugging line
  
      chrome.scripting.executeScript({
        target: {tabId: activeTab.id},
        function: grabContent
      }, (results) => {
        var screenGrab = results[0].result;
  
        var messageObject = {
          conversation: userMessage,
          url: activeTabURL,
          screenGrab: screenGrab
        };
  
        chat_with_chatbot(messageObject, function(chatbotResponse) {
          // Remove typing indicator
          conversation.removeChild(typingIndicator);
  
          // Append chatbot's reply to conversation
          conversation.innerHTML += `<div class="chatbot-message"><span class="chatbot-label"><strong>SaatvaAI:</strong></span> ${chatbotResponse}</div>`;
          
          // Scroll to the bottom again
          conversation.scrollTop = conversation.scrollHeight;
        });
      });
    });
  });
  
  

  function chat_with_chatbot(messageObject, callback) {
    console.log("Inside chat_with_chatbot");  // Debugging line
    fetch("http://127.0.0.1:5030/chatbot", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(messageObject)
    })
    .then(response => response.json())
    .then(data => callback(data.response))
    .catch((error) => console.error("An error occurred:", error));
  }
});

function grabContent() {
  return document.body.innerHTML;
}

function scrollToBottom() {
  const conversationDiv = document.getElementById("conversation");
  conversationDiv.scrollTop = conversationDiv.scrollHeight;
}

// Call this function every time a new message is added.
scrollToBottom();

// for resizing the icon
window.addEventListener("load", function () {
  const h2Element = document.querySelector("h2");
  const iconImage = document.getElementById("icon-image");

  const computedStyle = window.getComputedStyle(h2Element);
  const fontSize = parseFloat(computedStyle.height);

  iconImage.style.height = fontSize/4 + "px";
  iconImage.style.width = fontSize/4 + "px";
});
