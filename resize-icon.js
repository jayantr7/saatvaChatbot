window.addEventListener("load", function () {
    const h2Element = document.querySelector("h2");
    const iconImage = document.getElementById("icon-image");
  
    const computedStyle = window.getComputedStyle(h2Element);
    const fontSize = parseFloat(computedStyle.fontSize);
  
    iconImage.style.height = fontSize + "px";
    iconImage.style.width = "auto";
  });
  