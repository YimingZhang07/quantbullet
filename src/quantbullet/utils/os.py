import os
import shutil

def list_files_in_directory(directory: str, full_path: bool = False) -> list[str]:
    """List files in a directory."""
    if full_path:
        return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def list_files_in_directory_with_keywords(directory: str, keywords: list[str], full_path: bool = False) -> list[str]:
    """List files in a directory that contain any of the keywords."""
    if full_path:
        return [os.path.join(directory, f) for f in os.listdir(directory) if any(keyword in f for keyword in keywords)]
    return [f for f in os.listdir(directory) if any(keyword in f for keyword in keywords)]

def make_dir_if_not_exists(directory: str):
    """Make a directory if it does not exist."""
    os.makedirs(directory, exist_ok=True)
    return directory

def clean_dir(directory: str):
    """Clean a directory by removing all files and subdirectories."""
    for entry in os.listdir(directory):
        path = os.path.join(directory, entry)
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)