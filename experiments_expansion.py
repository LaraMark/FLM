# Import the FooocusExpansion class from the 'expansion' module in the 'modules' package.
from modules.expansion import FooocusExpansion

# Initialize an instance of the FooocusExpansion class.
expansion = FooocusExpansion()

# Define a string variable 'text' containing the text to be expanded.
text = 'a handsome man'

# Iterate over a range of 64 values, using each value as the seed for the expansion function.
for i in range(64):
    # Print the expanded text for the current seed value.
    print(expansion(text, seed=i))
