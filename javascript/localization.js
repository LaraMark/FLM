// Regular expression to match decimal numbers
var re_num = /^[.\d]+$/;

// Objects to store original and translated lines
var original_lines = {};
var translated_lines = {};

// Function to check if localization is available
function hasLocalization() {
    return window.localization && Object.keys(window.localization).length > 0;
}

// Function to get text nodes under an element
function textNodesUnder(el) {
    var n, a = [], walk = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, null, false);
    while ((n = walk.nextNode())) a.push(n);
    return a;
}

// Function to check if a node can be translated
function canBeTranslated(node, text) {
    if (!text) return false;
    if (!node.parentElement) return false;
    var parentType = node.parentElement.nodeName;
    if (parentType == 'SCRIPT' || parentType == 'STYLE' || parentType == 'TEXTAREA') return false;
    if (re_num.test(text)) return false;
    return true;
}

// Function to get the translated text for a given text
function getTranslation(text) {
    if (!text) return undefined;

    if (translated_lines[text] === undefined) {
        original_lines[text] = 1;
    }

    var tl = localization[text];
    if (tl !== undefined) {
        translated_lines[tl] = 1;
    }

    return tl;
}

// Function to process a text node
function processTextNode(node) {
    var text = node.textContent.trim();

    if (!canBeTranslated(node, text)) return;

    var tl = getTranslation(text);
    if (tl !== undefined) {
        node.textContent = tl;
        if (text && node.parentElement) {
          node.parentElement.setAttribute("data-original-text", text);
        }
    }
}

// Function to process a node
function processNode(node) {
    if (node.nodeType == 3) {
        processTextNode(node);
        return;
    }

    if (node.title) {
        let tl = getTranslation(node.title);
        if (tl !== undefined) {
            node.title = tl;
        }
    }

    if (node.placeholder) {
        let tl = getTranslation(node.placeholder);
        if (tl !== undefined) {
            node.placeholder = tl;
        }
    }

    // Recursively process child text nodes
    textNodesUnder(node).forEach(function(node) {
        processTextNode(node);
    });
}

// Function to refresh style localization
function refresh_style_localization() {
    processNode(document.querySelector('.style_selections'));
}

// Function to localize the whole page
function localizeWholePage() {
    processNode(gradioApp());

    // Functions to get elements for each component
    function elem(comp) {
        var elem_id = comp.props.elem_id ? comp.props.elem_id : "component-" + comp.id;
        return gradioApp().getElementById(elem_id);
    }

    // Loop through all components and localize them
    for (var comp of window.gradio_config.components) {
        if (comp.props.webui_tooltip) {
            let e = elem(comp);

            let tl = e ? getTranslation(e.title) : undefined;
            if (tl !== undefined) {
                e.title = tl;
            }
        }
        if (comp.props.placeholder) {
            let e = elem(comp);
            let textbox = e ? e.querySelector('[placeholder]') : null;

            let tl = textbox ? getTranslation(textbox.placeholder) : undefined;
            if (tl !== undefined) {
                textbox.placeholder = tl;
            }
        }
    }
}

// Event listener for DOMContentLoaded
document.addEventListener("DOMContentLoaded", function() {
    if (!hasLocalization()) {
        return;
    }

    // Observer to listen for UI updates
    onUiUpdate(function(m) {
        m.forEach(function(mutation) {
            mutation.addedNodes.forEach(function(node) {
                processNode(node);
            });
        });
    });

    // Localize the whole page
    localizeWholePage();

    if (localization.rtl) { // if the language is from right to left,
        (new MutationObserver((mutations, observer) => { // wait for the style to load
            mutations.forEach(mutation => {
                mutation.addedNodes.forEach(node => {
                    if (node.tagName === 'STYLE') { // find all rtl media rules
                        observer.disconnect();

                        for (const x of node.sheet.rules) { // enable them
                            if (Array.from(x.media || []).includes('rtl')) {
                                x.media.appendMedium('all');
                            }
                        }
                    }
                });
            });
        })).observe(gradioApp(), {childList: true});
    }
});

