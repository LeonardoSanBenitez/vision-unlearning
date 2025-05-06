import os
from typing import List, Dict, Optional
from enum import Enum

class ValidImageExtensions(Enum):
    """
    A class to hold valid image file extensions.
    """
    JPG = 0
    JPEG = 1
    PNG = 2
    BMP = 3
    TIFF = 4

def verify_images_in_path(path: str) -> bool:
    """
    Verifies if the given path contains image files.
    :param path: Path to the folder to check.
    :return: True if images are found, False otherwise.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The path '{path}' does not exist.")
    if not os.path.isdir(path):
        raise NotADirectoryError(f"The path '{path}' is not a directory.")

    valid_extensions = ValidImageExtensions._member_names_
    for file in os.listdir(path):
        if os.path.splitext(file)[1][1:].upper() in valid_extensions:
            return True
    return False