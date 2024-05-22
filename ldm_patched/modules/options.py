# Define a global variable 'args_parsing' and initialize it to False
args_parsing = False

def enable_args_parsing(enable=True):
    # This function enables or disables argument parsing by modifying the global 'args_parsing' variable

    # Set the global 'args_parsing' variable to the value of the 'enable' parameter
    # If the 'enable' parameter is not provided, it defaults to True
    global args_parsing
    args_parsing = enable
