import os
import tempfile
import shutil
from typing import Optional, List, Union
import logging
from smb.SMBConnection import SMBConnection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SMB server configuration
SMB_SERVER = "mobl-nas.mech.utah.edu"
SMB_SHARE = "mobl-nas"
BASE_OUTPUT_PATH = "/Users/SGilon/Mono/Output"
SMB_CLIENT_NAME = "mono-client"
SMB_SERVER_NAME = "mobl-nas"

# Connection instance
_conn = None
_temp_dir = None


def setup_smb_connection(username=None, password=None, domain=""):
    """Set up the SMB connection"""
    global _conn

    try:
        _conn = SMBConnection(
            username,
            password,
            SMB_CLIENT_NAME,
            SMB_SERVER_NAME,
            domain=domain,
            use_ntlm_v2=True,
        )
        connected = _conn.connect(SMB_SERVER, 445)  # 445 is the standard SMB port

        if connected:
            logger.info(f"Successfully connected to {SMB_SERVER}")
            return True
        else:
            logger.error(f"Failed to connect to {SMB_SERVER}")
            return False
    except Exception as e:
        logger.error(f"Error connecting to SMB server: {e}")
        return False


def get_conn():
    """Get the SMB connection, create it if it doesn't exist"""
    global _conn
    if _conn is None or not _conn.echo("echo"):
        setup_smb_connection()
    return _conn


def get_temp_dir():
    """Get the temporary directory"""
    global _temp_dir
    if _temp_dir is None:
        _temp_dir = tempfile.mkdtemp(prefix="smb_utils_")
    return _temp_dir


def clean_temp_dir():
    """Clean up temporary directory"""
    global _temp_dir
    if _temp_dir and os.path.exists(_temp_dir):
        shutil.rmtree(_temp_dir)
        _temp_dir = None


def normalize_path(path):
    """Convert Windows-style paths to Unix-style"""
    return path.replace("\\", "/").lstrip("/")


def get_smb_path(relative_path):
    """Get the SMB path from a relative path"""
    # Normalize the path
    path = normalize_path(relative_path)

    # Check if BASE_OUTPUT_PATH already ends with the start of relative_path
    if BASE_OUTPUT_PATH.endswith("/") and path.startswith("/"):
        path = path[1:]  # Remove leading slash from path

    return f"{BASE_OUTPUT_PATH}/{path}"


def ensure_dir_exists(relative_path):
    """Make sure a directory exists on the SMB server"""
    conn = get_conn()

    # Normalize the path
    path = normalize_path(relative_path)

    # Split the path into components
    path_parts = path.split("/")

    # Start with the base path
    current_path = BASE_OUTPUT_PATH

    # Try to create each directory level
    for i, part in enumerate(path_parts):
        if not part:  # Skip empty parts
            continue

        current_path = f"{current_path}/{part}"

        # If this is the last part and it might be a filename, skip directory creation
        if i == len(path_parts) - 1 and "." in part:
            continue

        try:
            # Check if directory exists
            try:
                conn.listPath(SMB_SHARE, current_path)
                logger.debug(f"Directory exists: {current_path}")
            except Exception:
                # Directory doesn't exist, create it
                logger.info(f"Creating directory: {current_path}")
                try:
                    conn.createDirectory(SMB_SHARE, current_path)
                except Exception as e:
                    logger.error(f"Failed to create directory {current_path}: {e}")
                    return False
        except Exception as e:
            logger.error(f"Error checking/creating directory {current_path}: {e}")
            return False

    return True


def read_file(relative_path, mode="r"):
    """Read a file from the SMB server"""
    conn = get_conn()
    smb_path = get_smb_path(relative_path)
    temp_file = os.path.join(get_temp_dir(), os.path.basename(relative_path))

    try:
        with open(temp_file, "wb") as file:
            conn.retrieveFile(SMB_SHARE, smb_path, file)

        # Read the temp file according to the requested mode
        with open(temp_file, mode) as file:
            content = file.read()

        return content
    except Exception as e:
        logger.error(f"Failed to read file {smb_path}: {e}")
        raise
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def write_file(relative_path, content, mode="w"):
    """Write content to a file on the SMB server"""
    conn = get_conn()

    # Normalize the path
    path = normalize_path(relative_path)

    # Get the full SMB path
    smb_path = f"{BASE_OUTPUT_PATH}/{path}"

    # Log the full path for debugging
    logger.info(f"Writing to SMB path: {smb_path}")

    temp_file = os.path.join(get_temp_dir(), os.path.basename(relative_path))

    try:
        # Make sure the directory exists
        if not ensure_dir_exists(os.path.dirname(relative_path)):
            logger.error(f"Failed to create directory structure for {relative_path}")
            return False

        # Write content to temp file
        with open(temp_file, mode) as file:
            file.write(content)

        # Upload temp file to SMB
        with open(temp_file, "rb") as file:
            try:
                conn.storeFile(SMB_SHARE, smb_path, file)
                logger.info(f"Successfully wrote file to {smb_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to store file on server: {e}")
                # Try with a different path format
                alt_path = smb_path.replace("/", "\\")
                logger.info(f"Trying alternative path format: {alt_path}")
                try:
                    conn.storeFile(SMB_SHARE, alt_path, file)
                    logger.info(
                        f"Successfully wrote file using alternative path format"
                    )
                    return True
                except Exception as e2:
                    logger.error(f"Alternative path also failed: {e2}")
                    return False
    except Exception as e:
        logger.error(f"Failed to write file {smb_path}: {e}")
        return False
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def list_dir(relative_path):
    """List files in a directory on the SMB server"""
    conn = get_conn()
    smb_path = get_smb_path(relative_path)

    try:
        file_list = conn.listPath(SMB_SHARE, smb_path)
        return [f.filename for f in file_list if f.filename not in [".", ".."]]
    except Exception as e:
        logger.error(f"Failed to list directory {smb_path}: {e}")
        return []


def save_dataframe(df, relative_path, index=False):
    """Save a pandas DataFrame to the SMB server"""
    temp_file = os.path.join(get_temp_dir(), "temp_df.csv")

    try:
        df.to_csv(temp_file, index=index)

        # Upload the file
        conn = get_conn()
        smb_path = get_smb_path(relative_path)

        # Make sure the directory exists
        ensure_dir_exists(os.path.dirname(relative_path))

        with open(temp_file, "rb") as file:
            conn.storeFile(SMB_SHARE, smb_path, file)

        return True
    except Exception as e:
        logger.error(f"Failed to save DataFrame to {relative_path}: {e}")
        return False
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def read_dataframe(relative_path):
    """Read a pandas DataFrame from the SMB server"""
    import pandas as pd

    conn = get_conn()
    smb_path = get_smb_path(relative_path)
    temp_file = os.path.join(get_temp_dir(), os.path.basename(relative_path))

    try:
        with open(temp_file, "wb") as file:
            conn.retrieveFile(SMB_SHARE, smb_path, file)

        return pd.read_csv(temp_file)
    except Exception as e:
        logger.error(f"Failed to read DataFrame from {relative_path}: {e}")
        return None
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


# Initialize connection when module is imported
try:
    setup_smb_connection()
except Exception as e:
    logger.warning(f"Could not initialize SMB connection on import: {e}")

# Cleanup on exit
import atexit

atexit.register(clean_temp_dir)
