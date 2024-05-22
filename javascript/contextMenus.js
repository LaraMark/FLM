// Based on the implementation from the stable-diffusion-webui repository (v1.6.0), this code defines a context menu for specific elements in the UI.

// The `contextMenuInit` function initializes the context menu functionality and returns an array of functions:
//   - `appendContextMenuOption`: Adds a new context menu option for a given target element selector, entry name, and entry function.
//   - `removeContextMenuOption`: Removes a context menu option by its unique ID.
//   - `addContextMenuEventListener`: Adds event listeners for context menu events.
var contextMenuInit = function() {
    // ... (function body)
};

// Initialize the context menu and extract the individual functions.
var initResponse = contextMenuInit();
var appendContextMenuOption = initResponse[0];
var removeContextMenuOption = initResponse[1];
var addContextMenuEventListener = initResponse[2];

// The `cancelGenerateForever` function clears the interval for generating outputs repeatedly.
let cancelGenerateForever = function() {
    clearInterval(window.generateOnRepeatInterval);
};

// Anonymous function to define context menu items for specific buttons.
(function() {
    // ... (function body)
})();

// Add event listeners for context menu events when the document is fully loaded.
document.onreadystatechange = function () {
    if (document.readyState == "complete") {
        addContextMenuEventListener();
    }
};
