// Set the default height of the main viewer
window.main_viewer_height = 512;

// Function to refresh the grid layout of the final gallery
function refresh_grid() {
    // Select the grid container and final gallery elements
    let gridContainer = document.querySelector('#final_gallery .grid-container');
    let final_gallery = document.getElementById('final_gallery');

    // If both elements exist
    if (gridContainer && final_gallery) {
        // Get the size of the final gallery element
        let rect = final_gallery.getBoundingClientRect();
        // Calculate the number of columns for the grid layout
        let cols = Math.ceil((rect.width - 16.0) / rect.height);
        // Ensure a minimum of 2 columns
        if (cols < 2) cols = 2;
        // Set the number of columns as a CSS variable for the grid container
        gridContainer.style.setProperty('--grid-cols', cols);
    }
}

// Function to refresh the grid layout with a delay
function refresh_grid_delayed() {
    // Call refresh_grid after 100, 500, and 1000 milliseconds
    setTimeout(refresh_grid, 100);
    setTimeout(refresh_grid, 500);
    setTimeout(refresh_grid, 1000);
}

// Function to adjust the height of the main view and refresh the grid layout
function resized() {
    // Calculate the height of the main view
    let windowHeight = window.innerHeight - 260;
    // Limit the height to 745 pixels
    if (windowHeight > 745) windowHeight = 745;
    // Select all elements with the 'main_view' class
    let elements = document.getElementsByClassName('main_view');
    // Set the height of each element
    for (let i = 0; i < elements.length; i++) {
        elements[i].style.height = windowHeight + 'px';
    }
    // Update the global variable for the main viewer height
    window.main_viewer_height = windowHeight;
    // Refresh the grid layout
    refresh_grid();
}

// Function to scroll to the top of the viewer with a delay
function viewer_to_top(delay = 100) {
    // Scroll to the top after the specified delay
    setTimeout(() => window.scrollTo({top: 0, behavior: 'smooth'}), delay);
}

// Function to scroll to the bottom of the viewer with a delay
function viewer_to_bottom(delay = 100) {
    // Select the element to scroll to
    let element = document.getElementById('positive_prompt');
    // Calculate the vertical position to scroll to
    let yPos = window.main_viewer_height;
    if (element) {
        // If the element exists, calculate the position based on its size and location
        yPos = element.getBoundingClientRect().top + window.scrollY;
    }
    // Scroll to the calculated position after the specified delay
    setTimeout(() => window.scrollTo({top: yPos - 8, behavior: 'smooth'}), delay);
}

// Add an event listener to the window object to adjust the height of the main view when the window is resized
window.addEventListener('resize', (e) => {
    resized();
});

// Function to execute code when the UI is loaded
onUiLoaded(async (event) => {
    // Adjust the height of the main view when the UI is loaded
    resized();
});

// Function to execute code when the style selection textarea loses focus
function on_style_selection_blur() {
    // Select the target textarea
    let target = document.querySelector("#gradio_receiver_style_selections textarea");
    // Set the value of the textarea to a random string
    target.value = "on_style_selection_blur " + Math.random();
    // Dispatch an 'input' event to trigger any event listeners on the textarea
    let e = new Event("input", {bubbles: true})
    Object.defineProperty(e, "target", {value: target})
    target.dispatchEvent(e);
}

// Function to execute code when the UI is loaded
onUiLoaded(async (event) => {
    // Select all span elements with the class 'aspect_ratios'
    let spans = document.querySelectorAll('.aspect_ratios span');
    // Replace all instances of '&lt;' and '&gt;' with '<' and '>'
    spans.forEach(function (span) {
        span.innerHTML = span.innerHTML.replace(/&lt;/g, '<').replace(/&gt;/g, '>');
    });
    // Add a 'focusout' event listener to the style selections element
    document.querySelector('.style_selections').addEventListener('focusout', function (event) {
        // If the active element is not a child of the style selections element
        setTimeout(() => {
            if (!this.contains(document.activeElement)) {
                // Call the on_style_selection_blur function
                on_style_selection_blur();
            }
        }, 200);
    });
    // Select all range input elements with the class 'lora_weight'
    let inputs = document.querySelectorAll('.lora_weight input[type="range"]');
    // Set the margin-top property of each input element
    inputs.forEach(function (input) {
        input.style.marginTop = '12px';
    });
});

