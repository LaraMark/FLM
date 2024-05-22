// A function to get the gradioApp element.
// If there are multiple gradio-app elements, it returns the first one.
// If there are no gradio-app elements, it returns the document object.
function gradioApp() {
    const elems = document.getElementsByTagName('gradio-app');
    const elem = elems.length == 0 ? document : elems[0];


