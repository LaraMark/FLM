/**
 * closeModal() - Function to close the lightbox modal by setting the display style of the modal to 'none'.
 */
function closeModal() {
    gradioApp().getElementById("lightboxModal").style.display = "none";
}


/**
 * showModal(event) - Function to display the selected image in the lightbox modal.
 * 
 * @param {Event} event - The event object containing information about the event that triggered this function.
 */
function showModal(event) {
    const source = event.target || event.srcElement; // Get the source element of the event
    const modalImage = gradioApp().getElementById("modalImage"); // Get the modal image element
    const lb = gradioApp().getElementById("lightboxModal"); // Get the lightbox modal element

    modalImage.src = source.src; // Set the source of the modal image to the source of the event

    if (modalImage.style.display === 'none') { // If the modal image is not displayed
        lb.style.setProperty('background-image', 'url(' + source.src + ')'); // Set the background image of the lightbox modal to the source of the event
    }

    lb.style.display = "flex"; // Display the lightbox modal
    lb.focus(); // Set the focus on the lightbox modal

    event.stopPropagation(); // Prevent the event from propagating further
}


/**
 * negmod(n, m) - Function to calculate the negative modulus of two numbers.
 * 
 * @param {number} n - The first number.
 * @param {number} m - The second number.
 * @return {number} The negative modulus of n with respect to m.
 */
function negmod(n, m) {
    return ((n % m) + m) % m;
}


/**
 * updateOnBackgroundChange() - Function to update the modal image when the background image changes.
 */
function updateOnBackgroundChange() {
    const modalImage = gradioApp().getElementById("modalImage"); // Get the modal image element

    if (modalImage && modalImage.offsetParent) { // If the modal image exists and has a parent element
        let currentButton = selected_gallery_button(); // Get the currently selected gallery button

        if (currentButton?.children?.length > 0 && modalImage.src != currentButton.children[0].src) { // If the current button has a child element and the source of the modal image is not equal to the source of the child element
            modalImage.src = currentButton.children[0].src; // Set the source of the modal image to the source of the child element
            if (modalImage.style.display === 'none') { // If the modal image is not displayed
                const modal = gradioApp().getElementById("lightboxModal"); // Get the lightbox modal element
                modal.style.setProperty('background-image', `url(${modalImage.src})`); // Set the background image of the lightbox modal to the source of the modal image
            }
        }
    }
}


/**
 * all_gallery_buttons() - Function to get all visible gallery buttons.
 * 
 * @return {NodeListOf<Element>} A nodelist of all visible gallery buttons.
 */
function all_gallery_buttons() {
    var allGalleryButtons = gradioApp().querySelectorAll('.image_gallery .thumbnails > .thumbnail-item.thumbnail-small'); // Get all gallery buttons with the specified class names
    var visibleGalleryButtons = []; // Initialize an empty array to store the visible gallery buttons

    allGalleryButtons.forEach(function(elem) { // Iterate over all gallery buttons
        if (elem.parentElement.offsetParent) { // If the parent element of the gallery button is in the document
            visibleGalleryButtons.push(elem); // Add the gallery button to the array of visible gallery buttons
        }
    });

    return visibleGalleryButtons; // Return the array of visible gallery buttons
}


/**
 * selected_gallery_button() - Function to get the currently selected gallery button.
 * 
 * @return {Element|null} The selected gallery button or null if no button is selected.
 */
function selected_gallery_button() {
    return all_gallery_buttons().find(elem => elem.classList.contains('selected')) ?? null; // Find the first gallery button that has the 'selected' class and return it or null if no button is found
}


/**
 * selected_gallery_index() - Function to get the index of the currently selected gallery button.
 * 
 * @return {number} The index of the selected gallery button or -1 if no button is selected.
 */
function selected_gallery_index() {
    return all_gallery_buttons().findIndex(elem => elem.classList.contains('selected')); // Find the index of the first gallery button that has the 'selected' class or -1 if no button is found
}


/**
 * modalImageSwitch(offset) - Function to switch the modal image by the given offset.
 * 
 * @param {number} offset - The offset to switch the modal image by.
 */
function modalImageSwitch(offset) {
    var galleryButtons = all_gallery_buttons(); // Get all visible gallery buttons

    if (galleryButtons.length > 1) { // If there are more than one gallery buttons
        var currentButton = selected_gallery_button(); // Get the currently selected gallery button

        var result = -1; // Initialize a variable to store the index of the current button
        galleryButtons.forEach(function(v, i) { // Iterate over all gallery buttons
            if (v == currentButton) { // If the current button is the selected gallery button
                result = i; // Store the index of the current button
            }
        });

        if (result != -1) { // If the selected gallery button was found
            var nextButton = galleryButtons[negmod((result + offset), galleryButtons.length)]; // Calculate the index of the next button based on the offset and the length of the gallery buttons
            nextButton.click(); // Click the next button
            const modalImage = gradioApp().getElementById("modalImage"); // Get the modal image element
            const modal = gradioApp().getElementById("lightboxModal"); // Get the lightbox modal element
            modalImage.src = nextButton.children[0].src; // Set the source of the modal image to the source of the child element of the next button
            if (modalImage.style.display === 'none') { // If the modal image is not displayed
                modal.style.setProperty('background-image', `url(${modalImage.src})`); // Set the background image of the lightbox modal to the source of the modal image
            }
            setTimeout(function() {
                modal.focus(); // Set the focus on the lightbox modal after a short delay
            }, 10);
        }
    }
}


/**
 * saveImage() - Function to save the current modal image.
 */
function saveImage() {
    // Implementation not provided
}


/**
 * modalSaveImage(event) - Function to handle the save image event.
 * 
 * @param {Event} event - The event object containing information about the event that triggered this function.
 */
function modalSaveImage(event) {
    event.stopPropagation(); // Prevent the event from propagating further
}


/**
 * modalNextImage(event) - Function to switch to the next modal image.
 * 
 * @param {Event} event - The event object containing information about the event that triggered this function.
 */
function modalNextImage(event) {
    modalImageSwitch(1); // Switch to the next modal image
    event.stopPropagation(); // Prevent the event from propagating further
}


/**
 * modalPrevImage(event) - Function to switch to the previous modal image.
 * 
 * @param {Event} event - The event object containing information about the event that triggered this function.
 */
function modalPrevImage(event) {
    modalImageSwitch(-1); // Switch to the previous modal image
    event.stopPropagation(); // Prevent the event from propagating further
}


/**
 * modalKeyHandler(event) - Function to handle keyboard events on the modal.
 * 
 * @param {KeyboardEvent} event - The event object containing information about the keyboard event that triggered this function.
 */
function modalKeyHandler(event) {
    switch (event.key) { // Switch on the key of the keyboard event
        case "s": // If the key is 's'
            saveImage(); // Save the current modal image
            break;
        case "ArrowLeft": // If the key is the left arrow key
            modalPrevImage(event); // Switch to the previous modal image
            break;
        case "ArrowRight": // If the key is the right arrow key
            modalNextImage(event); // Switch to the next modal image
            break;
        case "Escape": // If the key is the escape key
            closeModal(); // Close the modal
            break;
    }
}


/**
 * setupImageForLightbox(e) - Function to set up an image for the lightbox.
 * 
 * @param {Element} e - The image element to set up for the lightbox.
 */
function setupImageForLightbox(e) {
    if (e.dataset.modded) { // If the image has already been set up for the lightbox
        return; // Return early
    }

    e.dataset.modded = true; // Set a flag to indicate that the image has been set up for the lightbox

    e.style.cursor = 'pointer'; // Set the cursor to a pointer when hovering over the image
    e.style.userSelect = 'none'; // Prevent the image from being selected

    var isFirefox = navigator.userAgent.toLowerCase().indexOf('firefox') > -1; // Check if the browser is Firefox

    // For Firefox, listening on click first switched to next image then shows the lightbox.
    // If you know how to fix this without switching to mousedown event, please.
    // For other browsers the event is click to make it possiblr to drag picture.
    var event = isFirefox ? 'mousedown' : 'click'; // Choose the event to listen for based on the browser

    e.addEventListener(event, function(evt) { // Add an event listener for the chosen event
        if (evt.button == 1) { // If the middle mouse button was clicked
            open(evt.target.src); // Open the image in a new tab
            evt.preventDefault(); // Prevent the default behavior
            return;
        }
        if (evt.button != 0) return; // If a button other than the left mouse button was clicked, return early

        modalZoomSet(gradioApp().getElementById('modalImage'), true); // Set the modal image to the fullscreen state
        evt.preventDefault(); // Prevent the default behavior
        showModal(evt); // Show the modal with the current image
    }, true); // Use capture phase to ensure that the event is captured before it reaches the target element
}


/**
 * modalZoomSet(modalImage, enable) - Function to set the zoom state of the modal image.
 * 
 * @param {Element} modalImage - The modal image element to set the zoom state for.
 * @param {boolean} enable - Whether to enable or disable the zoom state.
 */
function modalZoomSet(modalImage, enable) {
    if (modalImage) modalImage.classList.toggle('modalImageFullscreen', !!enable); // Toggle the 'modalImageFullscreen' class on the modal image based on the enable flag
}


/**
 * modalZoomToggle(event) - Function to toggle the zoom state of the modal image.
 * 
 * @param {Event} event - The event object containing information about the event that triggered this function.
 */
function modalZoomToggle(event) {
    var modalImage = gradioApp().getElementById("modalImage"); // Get the modal image element
    modalZoomSet(modalImage, !modalImage.classList.contains('modalImageFullscreen')); // Toggle the zoom state of the modal image
    event.stopPropagation(); // Prevent the event from propagating further
}


/**
 * modalTileImageToggle(event) - Function to toggle the tiling state of the modal image.
 * 
 * @param {Event} event - The event object containing information about the event
