import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Union
from quantbullet.log_config import setup_logger

import pandas as pd

logger = setup_logger( __name__ )

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

def dict_to_clipboard(data: dict):
    """Copy a dictionary to clipboard in tabular form for Excel."""
    df = pd.DataFrame(list(data.items()), columns=["Key", "Value"])
    df.to_clipboard(index=False, header=True, excel=True)


def iter_subfolders(parent: Path, include_parent: bool = True):
    """
    Recursively yield all subfolders (and optionally the parent) of a given directory.
    
    Parameters
    ----------
    parent : Path
        The root folder to search.
    include_parent : bool
        Whether to include the parent folder itself in the results.
    """
    parent = Path(parent)
    if include_parent:
        yield parent
    for subdir in parent.rglob("*"):
        if subdir.is_dir():
            yield subdir

class DiskCacheCleaner:
    """Class to clean disk cache files
    
    This is especially designed for the diskcache decorators that create .json and .pkl files.
    """
    def __init__(self, folder: Union[str, Path], time_limit: timedelta):
        self.folder = Path(folder)
        self.time_limit = time_limit

    def _load_metadata(self, json_path: Path) -> dict:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read {json_path.name}: {e}")
            return {}

    def _is_expired(self, timestamp_str: str) -> bool:
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
            age = datetime.now(timezone.utc) - timestamp.astimezone(timezone.utc)
            return age > self.time_limit
        except Exception:
            return False

    def clean(self, dry_run: bool = True) -> None:
        json_files = self.folder.glob("*.json")

        for json_file in json_files:
            meta = self._load_metadata(json_file)
            timestamp_str = meta.get("timestamp")
            if not timestamp_str:
                continue

            if self._is_expired(timestamp_str):
                base_name = json_file.stem
                pkl_file = json_file.with_name(f"{base_name}.pkl")
                logger.info(f"Expired cache detected: {base_name}")

                if not dry_run:
                    try:
                        json_file.unlink(missing_ok=True)
                        pkl_file.unlink(missing_ok=True)
                        logger.info(f"Deleted {json_file.name} and {pkl_file.name}")
                    except Exception as e:
                        logger.error(f"Failed to delete {base_name}: {e}")

        if dry_run:
            logger.info("Dry run completed. No files deleted.")
