"""
Authentication utilities for SMB connections.
Keep this file private and do not commit it to version control.
"""

import os
import json
from typing import Dict, Tuple, Optional
import getpass

# Default location for auth file
DEFAULT_AUTH_FILE = os.path.join(os.path.dirname(__file__), "smb_credentials.json")


def create_auth_file(
    username: str = None,
    password: str = None,
    domain: str = "",
    file_path: str = DEFAULT_AUTH_FILE,
) -> bool:
    """
    Create an authentication file with SMB credentials.

    Args:
        username: SMB username (if None, will prompt)
        password: SMB password (if None, will prompt securely)
        domain: SMB domain (default is empty string)
        file_path: Path to save the credentials file

    Returns:
        True if successful, False otherwise
    """
    if username is None:
        username = input("Enter SMB username: ")

    if password is None:
        password = getpass.getpass("Enter SMB password: ")

    # Create credentials dictionary
    credentials = {"username": username, "password": password, "domain": domain}

    try:
        # Save credentials to file
        with open(file_path, "w") as f:
            json.dump(credentials, f)

        # Set restrictive permissions (on Unix-like systems)
        try:
            os.chmod(file_path, 0o600)  # Read/write for owner only
        except:
            pass  # May fail on Windows

        print(f"Credentials saved to {file_path}")
        return True
    except Exception as e:
        print(f"Error saving credentials: {e}")
        return False


def get_credentials(file_path: str = DEFAULT_AUTH_FILE) -> Optional[Dict[str, str]]:
    """
    Read credentials from the authentication file.

    Args:
        file_path: Path to the credentials file

    Returns:
        Dictionary with 'username', 'password', and 'domain' keys, or None if file not found
    """
    try:
        with open(file_path, "r") as f:
            credentials = json.load(f)
        return credentials
    except FileNotFoundError:
        print(f"Credentials file not found: {file_path}")
        print("Run create_auth_file() to create it.")
        return None
    except Exception as e:
        print(f"Error reading credentials: {e}")
        return None


if __name__ == "__main__":
    # When run directly, create new credentials file
    create_auth_file()
    print("\nTo use these credentials in your code:")
    print("from validation.auth import get_credentials")
    print("credentials = get_credentials()")
    print(
        "setup_smb_connection(username=credentials['username'], password=credentials['password'])"
    )
