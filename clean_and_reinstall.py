#!/usr/bin/env python3
"""
Script to uninstall all packages from requirements.txt and reinstall them globally.
"""
import os
import re
import subprocess
import sys

def parse_requirements(file_path):
    """Parse requirements.txt file and extract package names."""
    packages = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            # Extract package name (remove version specifiers)
            package_name = re.split(r'[<>=~]', line)[0].strip()
            packages.append(package_name)
    return packages

def uninstall_packages(packages):
    """Uninstall all packages."""
    print("Uninstalling packages...")
    for package in packages:
        print(f"Uninstalling {package}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", package], 
                          check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            print(f"Error uninstalling {package}: {e}")

def install_packages(requirements_file):
    """Install all packages from requirements.txt globally."""
    print("Installing packages globally...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_file], 
                      check=True)
        print("All packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        sys.exit(1)

def main():
    """Main function."""
    requirements_file = "requirements.txt"
    
    # Check if requirements.txt exists
    if not os.path.exists(requirements_file):
        print(f"Error: {requirements_file} not found.")
        sys.exit(1)
    
    # Parse requirements.txt
    packages = parse_requirements(requirements_file)
    print(f"Found {len(packages)} packages in {requirements_file}")
    
    # Uninstall packages
    uninstall_packages(packages)
    
    # Install packages globally
    install_packages(requirements_file)

if __name__ == "__main__":
    main()