import os
import args_manager
import modules.config
import json
import urllib.parse

from PIL import Image
from modules.util import generate_temp_filename

# Initialize a dictionary to store log cache
log_cache = {}

# This function returns the current HTML path for storing the log image
def get_current_html_path():
    date_string, local_temp_filename, only_name = generate_temp_filename(folder=modules.config.path_outputs,
                                                                         extension='png')
    html_name = os.path.join(os.path.dirname(local_temp_filename), 'log.html')
    return html_name

# This function logs the image and metadata to an HTML file
def log(img, dic):
    if args_manager.args.disable_image_log:
        return

    # Generate a unique filename for the log image
    date_string, local_temp_filename, only_name = generate_temp_filename(folder=modules.config.path_outputs, extension='png')
    os.makedirs(os.path.dirname(local_temp_filename), exist_ok=True)

    # Save the image to the generated filename
    Image.fromarray(img).save(local_temp_filename)

    # Construct the HTML file name
    html_name = os.path.join(os.path.dirname(local_temp_filename), 'log.html')

    # Define CSS styles for the HTML file
    css_styles = (
        "<style>"
        "body { background-color: #121212; color: #E0E0E0; } "
        "a { color: #BB86FC; } "
        ".metadata { border-collapse: collapse; width: 100%; } "
        ".metadata .key { width: 15%; } "
        ".metadata .value { width: 85%; font-weight: bold; } "
        ".metadata th, .metadata td { border: 1px solid #4d4d4d; padding: 4px; } "
        ".image-container img { height: auto; max-width: 512px; display: block; padding-right:10px; } "
        ".image-container div { text-align: center; padding: 4px; } "
        "hr { border-color: gray; } "
        "button { background-color: black; color: white; border: 1px solid grey; border-radius: 5px; padding: 5px 10px; text-align: center; display: inline-block; font-size: 16px; cursor: pointer; }"
        "button:hover {background-color: grey; color: black;}"
        "</style>"
    )

    # Define JavaScript code for copying the metadata to clipboard
    js = (
        """<script>
        function to_clipboard(txt) { 
        txt = decodeURIComponent(txt);
        if (navigator.clipboard && navigator.permissions) {
            navigator.clipboard.writeText(txt)
        } else {
            const textArea = document.createElement('textArea')
            textArea.value = txt
            textArea.style.width = 0
            textArea.style.position = 'fixed'
            textArea.style.left = '-999px'
            textArea.style.top = '10px'
            textArea.setAttribute('readonly', 'readonly')
            document.body.appendChild(textArea)

            textArea.select()
            document.execCommand('copy')
            document.body.removeChild(textArea)
        }
        alert('Copied to Clipboard!\\nPaste to prompt area to load parameters.\\nCurrent clipboard content is:\\n\\n' + txt);
        }
        </script>"""
    )

    # Define the beginning and ending parts of the HTML file
    begin_part = f"<html><head><title>Fooocus Log {date_string}</title>{css_styles}</head><body>{js}<p>Fooocus Log {date_string} (private)</p>\n<p>All images are clean, without any hidden data/meta, and safe to share with others.</p><!--fooocus-log-split-->\n\n"
    end_part = f'<!--fooocus-log-split--></body></html>'

    # Read the existing middle part of the HTML file
    middle_part = log_cache.get(html_name, "")

    # If the middle part is empty, read the existing HTML file and extract the middle part
    if middle_part == "":
        if os.path.exists(html_name):
            existing_split = open(html_name, 'r', encoding='utf-8').read().split('<!--fooocus-log-split-->')
            if len(existing_split) == 3:
                middle_part = existing_split[1]
            else:
                middle_part = existing_split[0]

    # Construct the div name for the log image and metadata
    div_name = only_name.replace('.', '_')

    # Construct the HTML code for the log image and metadata
    item = f"<div id=\"{div_name}\" class=\"image-container\"><hr><table><tr>\n"
    item += f"<td><a href=\"{only_name}\" target=\"_blank\"><img src='{only_name}' onerror=\"this.closest('.image-container').style.display='none';\" loading='lazy'></img></a><div>{only_name}</div></td>"
    item += "<td><table class='metadata'>"
    for key, value in dic:
        value_txt = str(value).replace('\n', ' </br> ')
        item += f"<tr><td class='key'>{key}</td><td class='value'>{value_txt}</td></tr>\n"
    item += "</table>"

    # Construct the JavaScript code for copying the metadata to clipboard
    js_txt = urllib.parse.quote(json.dumps({k: v for k, v in dic}, indent=0), safe='')
    item += f"</br><button onclick=\"to_clipboard('{js_txt}')\">Copy to Clipboard</button>"

    item += "</td>"
    item += "</tr></table></div>\n\n"

    # Update the middle part of the HTML file with the new log image and metadata
    middle_part = item + middle_part

    # Write the updated HTML file
    with open(html_name, 'w', encoding='utf-8') as f:
        f.write(begin_part + middle_part + end_part)

    # Print the path of the log image
    print(f'Image generated with private log at: {html_name}')

    # Update the log cache with the new middle part
    log_cache[html_name] = middle_part

    return
