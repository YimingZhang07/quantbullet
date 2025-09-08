import os

def list_files_in_directory(directory: str) -> list[str]:
    """List files in a directory."""
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def list_files_in_directory_with_keywords(directory: str, keywords: list[str]) -> list[str]:
    """List files in a directory that contain any of the keywords."""
    return [f for f in list_files_in_directory(directory) if any(keyword in f for keyword in keywords)]