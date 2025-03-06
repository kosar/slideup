"""
File handling utility for reading and writing various file formats.
"""

import os
import json
import yaml
import shutil
from pathlib import Path
from typing import Union, Dict, List, Any, Optional


class FileHandler:
    """Utility class for handling file operations."""
    
    @staticmethod
    def read_text(file_path: Union[str, Path]) -> str:
        """Read text from a file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    @staticmethod
    def write_text(file_path: Union[str, Path], content: str, create_dirs: bool = True) -> None:
        """Write text to a file."""
        file_path = Path(file_path)
        if create_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
    
    @staticmethod
    def read_binary(file_path: Union[str, Path]) -> bytes:
        """Read binary data from a file."""
        with open(file_path, 'rb') as file:
            return file.read()
    
    @staticmethod
    def write_binary(file_path: Union[str, Path], content: bytes, create_dirs: bool = True) -> None:
        """Write binary data to a file."""
        file_path = Path(file_path)
        if create_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'wb') as file:
            file.write(content)
    
    @staticmethod
    def read_json(file_path: Union[str, Path]) -> Dict:
        """Read and parse JSON from a file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    
    @staticmethod
    def write_json(file_path: Union[str, Path], data: Dict, indent: int = 4, create_dirs: bool = True) -> None:
        """Write data as JSON to a file."""
        file_path = Path(file_path)
        if create_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=indent)
    
    @staticmethod
    def read_yaml(file_path: Union[str, Path]) -> Dict:
        """Read and parse YAML from a file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    
    @staticmethod
    def write_yaml(file_path: Union[str, Path], data: Dict, create_dirs: bool = True) -> None:
        """Write data as YAML to a file."""
        file_path = Path(file_path)
        if create_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as file:
            yaml.dump(data, file, default_flow_style=False)
    
    @staticmethod
    def ensure_dir(directory: Union[str, Path]) -> Path:
        """Ensure a directory exists, creating it if necessary."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    
    @staticmethod
    def list_files(directory: Union[str, Path], pattern: str = "*", recursive: bool = False) -> List[Path]:
        """List files in a directory, optionally matching a pattern."""
        dir_path = Path(directory)
        if recursive:
            return list(dir_path.glob(f"**/{pattern}"))
        return list(dir_path.glob(pattern))
    
    @staticmethod
    def copy_file(source: Union[str, Path], destination: Union[str, Path], create_dirs: bool = True) -> None:
        """Copy a file from source to destination."""
        dest_path = Path(destination)
        if create_dirs:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    
    @staticmethod
    def move_file(source: Union[str, Path], destination: Union[str, Path], create_dirs: bool = True) -> None:
        """Move a file from source to destination."""
        dest_path = Path(destination)
        if create_dirs:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
    
    @staticmethod
    def remove_file(file_path: Union[str, Path]) -> None:
        """Remove a file."""
        Path(file_path).unlink(missing_ok=True)
    
    @staticmethod
    def file_exists(file_path: Union[str, Path]) -> bool:
        """Check if a file exists."""
        return Path(file_path).is_file()
