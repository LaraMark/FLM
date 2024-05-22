# Import the patched args_parser module from the ldm package
import ldm_patched.modules.args_parser as args_parser

# Create an argument parser object
args_parser.parser = argparse.ArgumentParser()

# Add command-line arguments with their respective types, actions, and help messages
args_parser.parser.add_argument("--share", action='store_true', help="Set whether to share on Gradio.")
args_parser.parser.add_argument("--preset", type=str, default=None, help="Apply specified UI preset.")
args_parser.parser.add_argument("--language", type=str, default='default', 
                                help="Translate UI using json files in [language] folder. "
                                  "For example, [--language example] will use [language/example.json] for translation.")
args_parser.parser.add_argument("--disable-offload-from-vram", action="store_true",
                                help="Force loading models to vram when the unload can be avoided. "
                                  "Some Mac users may need this.")
args_parser.parser.add_argument("--theme", type=str, help="launches the UI with light or dark theme", default=None)
args_parser.parser.add_argument("--disable-image-log", action='store_true',
                                help="Prevent writing images and logs to hard drive.")
args_parser.parser.add_argument("--disable-analytics", action='store_true',
                                help="Disables analytics for Gradio", default=False)

# Set default values for some arguments
args_parser.parser.set_defaults(
    disable_cuda_malloc=True,
    in_browser=True,
    port=None
)

# Parse the command-line arguments and store them in the args_parser.args object
args_parser.args = args_parser.parser.parse_args()

# Set the value of the always_offload_from_vram argument based on the value of the disable_offload_from_vram argument
args_parser.args.always_offload_from_vram = not args_parser.args.disable_offload_from_vram

# If the disable_analytics argument is set to True, set the GRADIO_ANALYTICS_ENVIRONMENT variable to False
if args_parser.args.disable_analytics:
    import os
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

# If the disable_in_browser argument is set to True, set the in_browser argument to False
if args_parser.args.disable_in_browser:
    args_parser.args.in_browser = False

# Assign the args_parser.args object to the args variable
args = args_parser.args
