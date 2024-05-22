import json
import os

# The 'current_translation' dictionary stores the current translations.
current_translation = {}

# The 'localization_root' variable stores the root directory of the localization files.
localization_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'language')

def localization_js(filename):
    # The function takes a filename as input and loads the corresponding localization file.
    global current_translation

    if isinstance(filename, str):
        # Check if the input is a string.
        full_name = os.path.abspath(os.path.join(localization_root, filename + '.json'))
        # Construct the full path of the localization file.
        if os.path.exists(full_name):
            # Check if the file exists.
            try:
                with open(full_name, encoding='utf-8') as f:
                    # Open the file and load the translations.
                    current_translation = json.load(f)
                    # Check if the loaded data is a dictionary.
                    assert isinstance(current_translation, dict)
                    for k, v in current_translation.items():
                        # Check if the keys and values are strings.
                        assert isinstance(k, str)
                        assert isinstance(v, str)
            except Exception as e:
                # Print an error message if there is an exception.
                print(str(e))
                print(f'Failed to load localization file {full_name}')

    # current_translation = {k: 'XXX' for k in current_translation.keys()}  # use this to see if all texts are covered

    # Return the localization data as a JSON string.
    return f"window.localization = {json.dumps(current_translation)}"


def dump_english_config(components):
    # The function takes a list of components as input and generates an English configuration file.
    all_texts = []
    for c in components:
        # Iterate through each component.
        label = getattr(c, 'label', None)
        # Get the label attribute of the component.
        value = getattr(c, 'value', None)
        # Get the value attribute of the component.
        choices = getattr(c, 'choices', None)
        # Get the choices attribute of the component.
        info = getattr(c, 'info', None)
        # Get the info attribute of the component.

        if isinstance(label, str):
            # Check if the label is a string and add it to the list of texts.
            all_texts.append(label)
        if isinstance(value, str):
            # Check if the value is a string and add it to the list of texts.
            all_texts.append(value)
        if isinstance(info, str):
            # Check if the info is a string and add it to the list of texts.
            all_texts.append(info)
        if isinstance(choices, list):
            # Check if the choices is a list and iterate through each element.
            for x in choices:
                if isinstance(x, str):
                    # Check if the element is a string and add it to the list of texts.
                    all_texts.append(x)
                if isinstance(x, tuple):
                    # Check if the element is a tuple and iterate through each element.
                    for y in x:
                        if isinstance(y, str):
                            # Check if the element is a string and add it to the list of texts.
                            all_texts.append(y)

    # Create a dictionary with unique keys from the list of texts.
    config_dict = {k: k for k in all_texts if k != "" and 'progress-container' not in k}
    full_name = os.path.abspath(os.path.join(localization_root, 'en.json'))

    # Write the dictionary to an English configuration file.
    with open(full_name, "w", encoding="utf-8") as json_file:
        json.dump(config_dict, json_file, indent=4)

    return
