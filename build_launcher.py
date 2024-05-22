import os

# Define the root directory of the win32 application
win32_root = os.path.dirname(os.path.dirname(__file__))

# Define the path to the embedded python executable
python_embeded_path = os.path.join(win32_root, 'python_embeded')

# Check if the win32 application is a standalone build and the python executable exists
is_win32_standalone_build = os.path.exists(python_embeded_path) and os.path.isdir(python_embeded_path)

# Define the command to be executed on win32 systems
win32_cmd = '''
.\python_embeded\python.exe -s Fooocus\entry_with_update.py {cmds} %*
pause
'''

def build_launcher():
    # Check if the win32 application is a standalone build
    if not is_win32_standalone_build:
        return

    # Define a list of presets
    presets = [None, 'anime', 'realistic']

    # Iterate through the list of presets
    for preset in presets:
        # Replace the placeholder in the command with the current preset, if any
        win32_cmd_preset = win32_cmd.replace('{cmds}', '' if preset is None else f'--preset {preset}')
        
        # Define the path to the batch file for the current preset
        bat_path = os.path.join(win32_root, 'run.bat' if preset is None else f'run_{preset}.bat')
        
        # Check if the batch file for the current preset does not exist
        if not os.path.exists(bat_path):
            # Create the batch file for the current preset
            with open(bat_path, "w", encoding="utf-8") as f:
                f.write(win32_cmd_preset)
    return
