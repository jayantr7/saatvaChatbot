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
    conversation.innerHTML += "<p>User: " + userMessage + "</p>";
    userInput.value = "";

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
          conversation.innerHTML += "<p>Chatbot: " + chatbotResponse + "</p>";
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
