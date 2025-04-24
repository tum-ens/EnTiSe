import os
import importlib

# Dynamically import all Python files in the current folder
current_dir = os.path.dirname(__file__)
for filename in os.listdir(current_dir):
    if filename.endswith(".py") and filename != "__init__.py":
        module_name = f"src.methods.{filename[:-3]}"  # Remove .py from the filename
        importlib.import_module(module_name)
