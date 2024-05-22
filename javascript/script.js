// A function to get the gradioApp element, which is either the document object or
// the first 'gradio-app' element found in the document.
function gradioApp() {
    const elems = document.getElementsByTagName('gradio-app');
    const elem = elems.length == 0 ? document : elems[0];


