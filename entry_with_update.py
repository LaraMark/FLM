import os
import sys

# Set the root directory to the directory of the current file
root = os.path.dirname(os.path.abspath(__file__))

# Add the root directory to the system path and change the current working directory
sys.path.append(root)
os.chdir(root)

try:
    # Import the pygit2 module and disable owner validation
    import pygit2
    pygit2.option(pygit2.GIT_OPT_SET_OWNER_VALIDATION, 0)

    # Initialize the repository object using the path of the current file's directory
    repo = pygit2.Repository(os.path.abspath(os.path.dirname(__file__)))

    # Get the name of the current branch
    branch_name = repo.head.shorthand

    # Set the name of the remote repository to 'origin'
    remote_name = 'origin'
    remote = repo.remotes[remote_name]

    # Fetch the latest data from the remote repository
    remote.fetch()

    # Set the local branch reference
    local_branch_ref = f'refs/heads/{branch_name}'
    local_branch = repo.lookup_reference(local_branch_ref)

    # Set the remote branch reference
    remote_reference = f'refs/remotes/{remote_name}/{branch_name}'
    remote_commit = repo.revparse_single(remote_reference)

    # Analyze the merge
    merge_result, _ = repo.merge_analysis(remote_commit.id)

    # Check if the local branch is up-to-date
    if merge_result & pygit2.GIT_MERGE_ANALYSIS_UP_TO_DATE:
        print("Already up-to-date")
    # Check if the local branch can be fast-forwarded
    elif merge_result & pygit2.GIT_MERGE_ANALYSIS_FASTFORWARD:
        # Set the target of the local branch to the remote commit
        local_branch.set_target(remote_commit.id)
        # Set the target of the repository head to the remote commit
        repo.head.set_target(remote_commit.id)
        # Checkout the tree of the remote commit
        repo.checkout_tree(repo.get(remote_commit.id))
        # Reset the repository to the state of the remote commit
        repo.reset(local_branch.target, pygit2.GIT_RESET_HARD)
        print("Fast-forward merge")
    # Check if the local branch cannot be merged
    elif merge_result & pygit2.GIT_MERGE_ANALYSIS_NORMAL:
        print("Update failed - Did you modify any file?")

except Exception as e:
    # Print an error message if an exception occurs
    print('Update failed.')
    print(str(e))

# Print a success message
print('Update succeeded.')

# Import the launch module
from launch import *
