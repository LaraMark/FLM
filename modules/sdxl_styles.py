import os
import re
import json
import math
import modules.config  # cannot use modules.config - validators causing circular imports

from modules.util import get_files_from_folder

# Set the path for the style files
styles_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../sdxl_styles/'))
wildcards_max_bfs_depth = 64  # maximum depth for Breadth-First Search when processing wildcards

def normalize_key(k):
    # Normalize the key by replacing '-' with ' ', capitalizing the first letter of each word,
    # and converting the first letter of the words '3d', 'Sai', 'Mre', and '(s' to uppercase
    k = k.replace('-', ' ')
    words = k.split(' ')
    words = [w[:1].upper() + w[1:].lower() for w in words]
    k = ' '.join(words)
    k = k.replace('3d', '3D')
    k = k.replace('Sai', 'SAI')
    k = k.replace('Mre', 'MRE')
    k = k.replace('(s', '(S')
    return k


styles = {}  # dictionary to store the style information

styles_files = get_files_from_folder(styles_path, ['.json'])

# List of style files to be loaded in a specific order
style_files_to_load = [
    'sdxl_styles_fooocus.json',
    'sdxl_styles_sai.json',
    'sdxl_styles_mre.json',
    'sdxl_styles_twri.json',
    'sdxl_styles_diva.json',
    'sdxl_styles_marc_k3nt3l.json'
]

# Load the style files
for styles_file in styles_files:
    if styles_file in style_files_to_load:
        styles_files.remove(styles_file)
        styles_files.append(styles_file)

for styles_file in styles_files:
    try:
        with open(os.path.join(styles_path, styles_file), encoding='utf-8') as f:
            for entry in json.load(f):
                name = normalize_key(entry['name'])
                prompt = entry['prompt'] if 'prompt' in entry else ''
                negative_prompt = entry['negative_prompt'] if 'negative_prompt' in entry else ''
                styles[name] = (prompt, negative_prompt)
    except Exception as e:
        print(str(e))
        print(f'Failed to load style file {styles_file}')

# List of legal style names
style_keys = list(styles.keys())
fooocus_expansion = "Fooocus V2"
legal_style_names = [fooocus_expansion] + style_keys


def apply_style(style, positive):
    # Apply the specified style to the positive prompt
    p, n = styles[style]
    return p.replace('{prompt}', positive).splitlines(), n.splitlines()


def apply_wildcards(wildcard_text, rng, i, read_wildcards_in_order):
    # Process wildcards in the text up to a maximum depth
    for _ in range(wildcards_max_bfs_depth):
        placeholders = re.findall(r'__([\w-]+)__', wildcard_text)
        if len(placeholders) == 0:
            return wildcard_text

        print(f'[Wildcards] processing: {wildcard_text}')
        for placeholder in placeholders:
            try:
                matches = [x for x in modules.config.wildcard_filenames if os.path.splitext(os.path.basename(x))[0] == placeholder]
                words = open(os.path.join(modules.config.path_wildcards, matches[0]), encoding='utf-8').read().splitlines()
                words = [x for x in words if x != '']
                assert len(words) > 0
                if read_wildcards_in_order:
                    wildcard_text = wildcard_text.replace(f'__{placeholder}__', words[i % len(words)], 1)
                else:
                    wildcard_text = wildcard_text.replace(f'__{placeholder}__', rng.choice(words), 1)
            except:
                print(f'[Wildcards] Warning: {placeholder}.txt missing or empty. '
                      f'Using "{placeholder}" as a normal word.')
                wildcard_text = wildcard_text.replace(f'__{placeholder}__', placeholder)
            print(f'[Wildcards] {wildcard_text}')

    print(f'[Wildcards] BFS stack overflow. Current text: {wildcard_text}')
    return wildcard_text


def get_words(arrays, totalMult, index):
    # Return a list of words from the input arrays based on the index
    if len(arrays) == 1:
        return [arrays[0].split(',')[index]]
    else:
        words = arrays[0].split(',')
        word = words[index % len(words)]
        index -= index % len(words)
        index /= len(words)
        index = math.floor(index)
        return [word] + get_words(arrays[1:], math.floor(totalMult/len(words)), index)


def apply_arrays(text, index):
    # Process arrays in the text and replace them with words from the arrays
    arrays = re.findall(r'\[\[(.*?)\]\]', text)
    if len(arrays) == 0:
        return text

    print(f'[Arrays] processing: {text}')
    mult = 1
    for arr in arrays:
        words = arr.split(',')
        mult *= len(words)
    
    index %= mult
    chosen_words = get_words(arrays, mult, index)
    
    i = 0
    for arr in arrays:
        text = text.replace(f'[[{arr}]]', chosen_words[i], 1)   
        i = i+1
    
    return text

