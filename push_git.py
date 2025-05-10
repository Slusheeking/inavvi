#!/usr/bin/env python3
"""
Simple Git automation script for the FinGPT AI Day Trading System.
Performs: git status, git add, git commit with timestamp, and git push.
"""

import datetime
import subprocess
import sys


def run_command(command):
    """Run a shell command and return the output."""
    print(f"Executing: {command}")
    
    # Use list form for better compatibility with VSCode
    if isinstance(command, str):
        import shlex
        command_list = shlex.split(command)
    else:
        command_list = command
    
    try:
        # Use subprocess.Popen for better compatibility with VSCode
        process = subprocess.Popen(
            command_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            universal_newlines=True
        )
        
        stdout, stderr = process.communicate()
        
        if stdout:
            print(f"Output:\n{stdout}")
            
        if stderr and process.returncode != 0:
            print(f"Error:\n{stderr}")
            raise Exception(f"Command failed with exit code {process.returncode}")
            
        return stdout
    except FileNotFoundError:
        print(f"Error: Command not found: {command_list[0]}")
        raise
    except Exception as e:
        print(f"Error executing command: {e}")
        raise


def get_current_branch():
    """Get the name of the current git branch."""
    try:
        branch = run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip()
        return branch
    except Exception:
        # Default to main if we can't determine the branch
        print("Warning: Could not determine current branch, defaulting to 'main'")
        return "main"


def main():
    """Main function to execute git commands."""
    try:
        # Check if git is installed
        try:
            run_command(["git", "--version"])
        except Exception:
            print("Error: Git is not installed or not in the PATH")
            return 1

        # Get current git status
        print("\n--- Checking Git Status ---")
        status_output = run_command(["git", "status"])

        # Check if there are changes to commit
        if "nothing to commit, working tree clean" in status_output:
            print("No changes to commit. Exiting.")
            return 0

        # Add all changes
        print("\n--- Adding All Changes ---")
        run_command(["git", "add", "."])

        # Create commit message with timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        commit_message = f"Automated update: {timestamp}"

        # Commit changes
        print(f"\n--- Committing Changes with message: '{commit_message}' ---")
        run_command(["git", "commit", "-m", commit_message])

        # Get current branch
        current_branch = get_current_branch()
        
        # Push changes
        print(f"\n--- Pushing to Remote Repository (branch: {current_branch}) ---")
        run_command(["git", "push", "origin", current_branch])

        print("\n--- Git Operations Completed Successfully ---")

    except subprocess.CalledProcessError as e:
        print(f"\nGit command failed: {e}")
        return 1
    except Exception as e:
        print(f"\nError occurred: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
