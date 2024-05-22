import numpy as np
import csv
import onnxruntime as ort

# Import custom modules for loading the model and configuring paths
from PIL import Image
from onnxruntime import InferenceSession
from modules.config import path_clip_vision
from modules.model_loader import load_file_from_url

# Define a global variable for the ONNX model to avoid reloading it for every call to the function
global_model = None

# Define a global variable for the CSV file to avoid reloading it for every call to the function
global_csv = None

def default_interrogator(image_rgb, threshold=0.35, character_threshold=0.85, exclude_tags=""):
    # Load the ONNX model if it has not been loaded before
    if global_model is not None:
        model = global_model
    else:
        # Load the ONNX model from Hugging Face
        model_onnx_filename = load_file_from_url(
            url=f'https://huggingface.co/lllyasviel/misc/resolve/main/{model_name}.onnx',
            model_dir=path_clip_vision,
            file_name=f'{model_name}.onnx',
        )
        model = InferenceSession(model_onnx_filename, providers=ort.get_available_providers())
        global_model = model

    # Load the CSV file if it has not been loaded before
    if global_csv is not None:
        csv_lines = global_csv
    else:
        # Load the CSV file from Hugging Face
        csv_lines = []
        with open(model_csv_filename) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                csv_lines.append(row)
        global_csv = csv_lines

    # Define the tags and their corresponding probabilities
    tags = []
    general_index = None
    character_index = None
    for line_num, row in enumerate(csv_lines):
        # Set the indices of the general and character tags
        if general_index is None and row[2] == "0":
            general_index = line_num
        elif character_index is None and row[2] == "4":
            character_index = line_num
        # Add the tag to the list of tags
        tags.append(row[1])

    # Preprocess the input image
    input = model.get_inputs()[0]
    height = input.shape[1]
    image = Image.fromarray(image_rgb)  # RGB
    ratio = float(height)/max(image.size)
    new_size = tuple([int(x*ratio) for x in image.size])
    image = image.resize(new_size, Image.LANCZOS)
    square = Image.new("RGB", (height, height), (255, 255, 255))
    square.paste(image, ((height-new_size[0])//2, (height-new_size[1])//2))
    image = np.array(square).astype(np.float32)
    image = image[:, :, ::-1]  # RGB -> BGR
    image = np.expand_dims(image, 0)

    # Run the model on the input image
    label_name = model.get_outputs()[0].name
    probs = model.run([label_name], {input.name: image})[0]

    # Filter the tags based on the minimum probability threshold and the list of tags to exclude
    result = list(zip(tags, probs[0]))
    general = [item for item in result[general_index:character_index] if item[1] > threshold]
    character = [item for item in result[character_index:] if item[1] > character_threshold]
    all = character + general
    remove = [s.strip() for s in exclude_tags.lower().split(",")]
    all = [tag for tag in all if tag[0] not in remove]

    # Return a string of the remaining tags, separated by commas
    res = ", ".join((item[0].replace("(", "\\(").replace(")", "\\)") for item in all)).replace('_', ' ')
    return res
