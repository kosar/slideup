
import os
from flask import jsonify

def check_env_keys():
    """
    Check which API keys are set in the environment and return a JSON response.
    Used by both web applications to display UI indicators.
    """
    return jsonify({
        "openai_key_set": bool(os.getenv('OPENAI_API_KEY')),
        "stability_key_set": bool(os.getenv('STABILITY_API_KEY')),
        "deepseek_key_set": bool(os.getenv('DEEPSEEK_API_KEY'))
    })

def store_original_env():
    """
    Store the current environment variables for later restoration.
    Returns a dictionary of the stored variables.
    """
    original_env = {}
    if 'OPENAI_API_KEY' in os.environ:
        original_env['OPENAI_API_KEY'] = os.environ['OPENAI_API_KEY']
    if 'STABILITY_API_KEY' in os.environ:
        original_env['STABILITY_API_KEY'] = os.environ['STABILITY_API_KEY']
    if 'DEEPSEEK_API_KEY' in os.environ:
        original_env['DEEPSEEK_API_KEY'] = os.environ['DEEPSEEK_API_KEY']
    return original_env

def restore_env(original_env):
    """
    Restore environment variables from the provided dictionary.
    """
    for key, value in original_env.items():
        os.environ[key] = value
