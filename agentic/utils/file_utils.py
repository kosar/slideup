import os
import base64
import streamlit as st
from pathlib import Path
import shutil

def save_uploaded_file(uploaded_file, destination_path):
    """
    Save an uploaded file to the specified destination path.
    
    Args:
        uploaded_file: The uploaded file object from Streamlit
        destination_path: The path where the file should be saved
        
    Returns:
        Path to the saved file
    """
    # Make sure the directory exists
    destination_dir = os.path.dirname(destination_path)
    os.makedirs(destination_dir, exist_ok=True)
    
    # Save the file
    with open(destination_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return destination_path

def create_download_link(file_path, link_text, mime_type="application/octet-stream"):
    """
    Create a download link for a file.
    
    Args:
        file_path: Path to the file to be downloaded
        link_text: Text to display on the download link
        mime_type: MIME type of the file
        
    Returns:
        HTML code for the download link
    """
    with open(file_path, "rb") as f:
        data = f.read()
    
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:{mime_type};base64,{b64}" download="{os.path.basename(file_path)}">{link_text}</a>'
    
    return href

def get_file_extension(file_path):
    """
    Get the file extension from a file path.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File extension (without the dot)
    """
    return Path(file_path).suffix[1:]

def ensure_dir_exists(directory_path):
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
    """
    Path(directory_path).mkdir(parents=True, exist_ok=True)

def cleanup_temp_files(directory_path):
    """
    Clean up temporary files in a directory.
    
    Args:
        directory_path: Path to the directory to clean
    """
    try:
        shutil.rmtree(directory_path)
    except Exception as e:
        print(f"Error cleaning up temporary files: {e}")
