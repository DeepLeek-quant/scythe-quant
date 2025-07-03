#!/usr/bin/env python3
"""
Simple script to use scythe package locally.
Installs requirements and imports scythe.
"""

import sys
import subprocess
from pathlib import Path

# Install requirements
print("Installing requirements...")
try:
    # Try installing all at once first
    result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                          capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Failed to install all requirements at once. Error: {result.stderr}")
        print("Trying to install packages one by one...")
        
        # Read requirements and install one by one
        with open("requirements.txt", "r") as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        for req in requirements:
            print(f"Installing {req}...")
            result = subprocess.run([sys.executable, "-m", "pip", "install", req], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Failed to install {req}: {result.stderr}")
            else:
                print(f"Successfully installed {req}")
    else:
        print("Requirements installed successfully!")
        
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import scythe
import scythe

print("Successfully imported scythe package!")
print(f"Scythe version: {scythe.__version__}")
print(f"Config directory: {scythe.get_config_dir()}")