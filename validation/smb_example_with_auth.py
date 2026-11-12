import os
import pandas as pd
import numpy as np
from smb_utils_alternative import (
    setup_smb_connection,
    write_file,
    read_file,
    save_dataframe,
    read_dataframe,
    ensure_dir_exists,
)
from auth import get_credentials


def example_with_auth():
    """Example of using the SMB utilities with authentication"""

    # Get credentials from the auth file
    credentials = get_credentials()
    if credentials:
        # Set up connection with stored credentials
        setup_smb_connection(
            username=credentials["username"],
            password=credentials["password"],
            domain=credentials.get("domain", ""),
        )

        # Try to create a simple test file
        print("Creating test directory...")
        ensure_dir_exists("test")

        # Now use the SMB utilities as normal
        print("Writing test file...")
        success = write_file("test/test_auth.txt", "This is a test with authentication")

        if success:
            print("File written successfully!")
        else:
            print("Failed to write file.")
    else:
        print("Could not load credentials.")


if __name__ == "__main__":
    example_with_auth()
