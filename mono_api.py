import os
import time
import yaml
import json
import pickle
import torch
from loguru import logger
from decouple import config
from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    Form,
    Query,
    Response,
    Request,
    Header,
    Depends,
)
from fastapi.security import APIKeyHeader
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel
from optimization import run_optimization
from utils import utils_optim as ut
from WHAM.demo import main_wham
from utils.tracking_filters import InsufficientFullBodyKeypointsError
from utils.camera_motion import detect_static_camera
from hashlib import md5
from visualization.utils import generateVisualizerJson
from visualization.automation import automate_recording
from typing import Optional
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import inspect
from utils.convert_to_avi import convert_to_avi
from utils.utilsCameraPy3 import getVideoRotation
import hashlib
import requests
import uvicorn
import traceback
import uuid
from datetime import datetime

# Absolute path to the directory containing this file (the project root for the API).
_API_ROOT = os.path.dirname(os.path.abspath(__file__))

# Subdirectory prefixes that can appear anywhere in an incoming path and be
# re-rooted against _API_ROOT when the original absolute path is not accessible
# (e.g. worker on a different machine or outside Docker).
_REROOTED_PREFIXES = [
    "camera_intrinsics/",
    "opencap/Data/",
    "opencap/data/",
    "opencap/",
    "examples/",
]


def _resolve_path(path: str) -> str:
    """
    Resolve a file path to one that is accessible by the API process.

    Priority:
      1. If the path exists as-is, return it unchanged.
      2. If the path is relative, join it with _API_ROOT.
      3. If the path is absolute but missing (e.g. came from a worker on a
         different machine / outside Docker), strip the machine-specific prefix
         and re-root the relative tail against _API_ROOT.
    """
    if os.path.exists(path):
        return path
    if not os.path.isabs(path):
        return os.path.join(_API_ROOT, path)
    # Absolute path that doesn't exist here — try to salvage the relative tail.
    for prefix in _REROOTED_PREFIXES:
        idx = path.find(prefix)
        if idx >= 0:
            tail = path[idx:]
            candidate = os.path.join(_API_ROOT, tail)
            if os.path.exists(candidate):
                return candidate
    return path  # return original so the caller's error message is informative

# Import enhanced logging and Slack notifier
try:
    from deployment.lib.logging_config import (
        setup_logging,
        log_request_info,
        log_error_with_context,
        log_performance_metrics,
        log_api_request,
        log_api_response,
    )

    # Setup enhanced logging for API
    setup_logging("api")
    LOGGING_ENABLED = True
except ImportError:
    # Fallback to basic logging if enhanced logging is not available
    logger.warning("Enhanced logging not available, using basic logging")
    LOGGING_ENABLED = False

# Import Slack notifier
try:
    from deployment.slack.slack_notifier import SlackNotifier, AlertLevel

    slack_notifier = SlackNotifier()
    SLACK_ENABLED = slack_notifier.enabled
    if SLACK_ENABLED:
        logger.info("Slack notifications enabled")
    else:
        logger.info("Slack notifications disabled (webhook not configured)")
except ImportError:
    SLACK_ENABLED = False
    slack_notifier = None
    AlertLevel = None
    logger.warning("Slack notifier not available")

# API Key Authentication Configuration
# Uses decouple.config() which reads from .env file or environment variables
MONO_API_KEY = config("MONO_API_KEY", default="")
REQUIRE_API_KEY = config("REQUIRE_API_KEY", default="true", cast=bool)
ENABLE_STATIC_CAMERA_CHECK = config(
    "ENABLE_STATIC_CAMERA_CHECK", default="true", cast=bool
)

# Log API key configuration (masked for security)
if MONO_API_KEY:
    logger.info(
        f"API key authentication enabled. Key configured: {MONO_API_KEY[:8]}...{MONO_API_KEY[-4:]} (length: {len(MONO_API_KEY)})"
    )
    logger.info(f"REQUIRE_API_KEY: {REQUIRE_API_KEY}")
else:
    logger.warning(
        "MONO_API_KEY not configured. API key authentication may be disabled or misconfigured."
    )

# API Key security schemes
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    request: Request,
    api_key_header: Optional[str] = Depends(api_key_header),
) -> bool:
    """
    Verify API key from header or query parameter.

    Args:
        request: FastAPI request object
        api_key_header: API key from X-API-Key header

    Returns:
        True if API key is valid or authentication is disabled

    Raises:
        HTTPException: 401 if API key is missing or invalid
    """
    # Skip authentication if disabled
    if not REQUIRE_API_KEY:
        return True

    # Only the root health-check is public
    if request.url.path == "/":
        return True

    # Accept API key from header only (not query string — keys in URLs appear in logs)
    api_key = api_key_header

    # Check if API key is configured
    if not MONO_API_KEY:
        logger.warning("API key authentication is enabled but MONO_API_KEY is not set")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "API key authentication misconfigured",
                "error_type": "ConfigurationError",
                "message": "Server is configured to require API key but no key is set",
            },
        )

    # Verify API key
    if not api_key:
        logger.debug(f"API key missing for request to {request.url.path}")
        raise HTTPException(
            status_code=401,
            detail={
                "error": "API key required",
                "error_type": "AuthenticationError",
                "message": "Missing API key. Provide X-API-Key header or api_key query parameter",
                "suggestion": "Include X-API-Key header or api_key query parameter with your request",
            },
        )

    if api_key != MONO_API_KEY:
        logger.warning(
            f"Invalid API key attempt for {request.url.path} from {request.client.host if request.client else 'unknown'}"
        )
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Invalid API key",
                "error_type": "AuthenticationError",
                "message": "The provided API key is invalid",
                "suggestion": "Check that you are using the correct API key",
            },
        )

    return True


# command to run the API:
# uvicorn mono_api:app --host 0.0.0.0 --port 8000
# navigate to http://0.0.0.0:8000/docs to see the API documentation and test the API
# ssh -R 80:localhost:8000 serveo.net
#     ssh -R 80:localhost:8000 nglocalhost.com

app = FastAPI()

# Configure CORS to allow ALL origins (USE WITH CAUTION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any origin
    allow_credentials=False,  # Must be False when allow_origins is ["*"]
    allow_methods=["*"],  # Allows all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)


def _is_bot_scanner_request(path: str) -> bool:
    """
    Detect if a request path is likely from a bot/scanner probing for vulnerabilities.

    Args:
        path: The request path to check

    Returns:
        True if the path matches common bot/scanner patterns, False otherwise
    """
    # Check if filtering is enabled (default: True)
    filter_enabled = os.getenv("FILTER_BOT_REQUESTS", "true").lower() == "true"
    if not filter_enabled:
        return False

    path_lower = path.lower()

    # Common CMS and framework paths that bots probe
    bot_path_patterns = [
        "/themes/",
        "/modules/",
        "/assets/",
        "/static/",
        "/admin/",
        "/wp-admin/",
        "/wp-content/",
        "/wp-includes/",
        "/phpmyadmin/",
        "/pma/",
        "/wordpress/",
        "/joomla/",
        "/drupal/",
        "/magento/",
        "/prestashop/",
        "/opencart/",
        "/config/",
        "/backup/",
        "/backups/",
        "/.env",
        "/.git/",
        "/.svn/",
        "/.htaccess",
        "/.htpasswd",
        "/phpinfo",
        "/info.php",
        "/test.php",
        "/shell.php",
        "/cgi-bin/",
        "/.well-known/",
    ]

    # Check for bot path patterns
    for pattern in bot_path_patterns:
        if pattern in path_lower:
            return True

    # Check for file extensions that are commonly probed (but not in legitimate API paths)
    # Only flag if the path doesn't start with known API endpoints
    legitimate_api_prefixes = [
        "/run_mono",
        "/download",
        "/batch_status",
        "/docs",
        "/openapi.json",
        "/redoc",
    ]
    is_legitimate_api = any(
        path.startswith(prefix) for prefix in legitimate_api_prefixes
    )

    if not is_legitimate_api:
        # Common file extensions that bots look for
        suspicious_extensions = [
            ".less",
            ".css",
            ".js",
            ".php",
            ".asp",
            ".aspx",
            ".jsp",
            ".cgi",
            ".pl",
            ".sh",
        ]
        for ext in suspicious_extensions:
            if path_lower.endswith(ext):
                return True

    return False


def _user_facing_http_error_msg(error_detail: dict) -> str:
    """Short message for clients; technical detail stays in error_msg_dev."""
    um = error_detail.get("user_message")
    if um is not None and str(um).strip():
        return str(um).strip()
    err = error_detail.get("error")
    suggestion = (error_detail.get("suggestion") or "").strip()
    if err is not None and str(err).strip():
        err_s = str(err).strip()
        if suggestion and suggestion.lower() not in err_s.lower():
            base = err_s.rstrip(".")
            return f"{base}. {suggestion}" if base else suggestion
        return err_s
    error_type = error_detail.get("error_type", "Error")
    message = error_detail.get("message", "")
    stage = error_detail.get("stage", "")
    error_msg = f"{error_type}: {message}"
    if stage:
        error_msg = f"[{stage}] {error_msg}"
    return error_msg


# WHAM raises ValueError with fixed strings for common recording issues; map to friendly API copy.
_WHAM_USER_FRIENDLY_VALUE_ERRORS: dict[str, tuple[str, str]] = {
    "No tracking results found": (
        "No person was detected in this video",
        "Make sure someone is visible in the frame with reasonable lighting. "
        "Videos with no human in view cannot be processed.",
    ),
    "Tracking results is not a dictionary with the key 0": (
        "Pose tracking did not yield a single person to analyze",
        "Use a clip with one clear subject in view; multiple or overlapping people "
        "can prevent reliable tracking.",
    ),
    "There should be only one subject in the dataset": (
        "Multiple people appear in the video",
        "This pipeline needs one person in the recording. Try a segment where only one person is visible.",
    ),
}


def _run_static_camera_check_or_raise(video_path: str):
    result = detect_static_camera(video_path)
    logger.info(f"Static camera check result: {json.dumps(result.to_dict())}")

    if result.is_static:
        return result

    raise HTTPException(
        status_code=422,
        detail={
            "error": "Moving camera detected",
            "error_type": "MovingCameraDetectedError",
            "message": result.reason,
            "user_message": "We currently only support static-camera recordings. Camera movement was detected, so this trial was not processed.",
            "stage": "CameraMotionCheck",
            "suggestion": "Record with a fixed camera or tripod and avoid panning, tilting, or repositioning the device during the trial.",
            "video_path": video_path,
            "metrics": result.to_dict(),
        },
    )


# Add HTTP exception handler to properly format errors
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions and ensure detailed error is returned"""
    error_detail = (
        exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
    )

    # Check if this is a bot/scanner request
    is_bot_request = _is_bot_scanner_request(str(request.url.path))

    # For 404s on bot paths, log at DEBUG level instead of ERROR to reduce noise
    # For legitimate 404s and other errors, keep ERROR level
    if exc.status_code == 404 and is_bot_request:
        logger.debug(
            f"HTTP {exc.status_code} error (bot/scanner request): {json.dumps(error_detail, default=str)} | Path: {request.url.path}"
        )
    else:
        logger.error(
            f"HTTP {exc.status_code} error: {json.dumps(error_detail, default=str)} | Path: {request.url.path}"
        )

    # Build user-friendly error message
    if isinstance(error_detail, dict):
        error_type = error_detail.get("error_type", "Error")
        message = error_detail.get("message", str(exc.detail))
        stage = error_detail.get("stage", "")
        suggestion = error_detail.get("suggestion", "")

        error_msg = _user_facing_http_error_msg(error_detail)

        # Create detailed developer message with all context
        error_msg_dev_parts = [f"Error Type: {error_type}", f"Message: {message}"]

        if stage:
            error_msg_dev_parts.append(f"Stage: {stage}")

        if suggestion:
            error_msg_dev_parts.append(f"Suggestion: {suggestion}")

        # Add input parameters if available
        if "inputs" in error_detail:
            inputs = error_detail["inputs"]
            key_inputs = {
                k: v
                for k, v in inputs.items()
                if k
                in ["video_path", "data_dir", "height_m", "mass_kg", "sex", "activity"]
            }
            if key_inputs:
                error_msg_dev_parts.append(
                    f"Key Inputs: {json.dumps(key_inputs, default=str)}"
                )

        # Add traceback summary (first few lines)
        if "traceback" in error_detail and error_detail["traceback"]:
            tb_lines = error_detail["traceback"].split("\n")
            # Get the actual error line
            for line in reversed(tb_lines):
                if line.strip() and not line.strip().startswith("Traceback"):
                    error_msg_dev_parts.append(f"Error Line: {line.strip()}")
                    break

        error_msg_dev = " | ".join(error_msg_dev_parts)
    else:
        error_msg = str(exc.detail)
        error_msg_dev = f"HTTP {exc.status_code}: {str(exc.detail)}"

    # Build comprehensive response
    response_body = {
        "detail": error_detail,
        "error_msg": error_msg,
        "error_msg_dev": error_msg_dev,
        "status_code": exc.status_code,
    }

    # Send Slack notification for 500 errors
    if exc.status_code >= 500 and SLACK_ENABLED and slack_notifier:
        try:
            slack_notifier.send_error_alert(
                error_type=(
                    error_detail.get("error_type", "HTTPException")
                    if isinstance(error_detail, dict)
                    else "HTTPException"
                ),
                error_message=(
                    error_detail.get("message", str(exc.detail))
                    if isinstance(error_detail, dict)
                    else str(exc.detail)
                ),
                service="api",
                context={
                    "status_code": exc.status_code,
                    "path": str(request.url.path),
                    "method": request.method,
                },
                traceback_str=(
                    error_detail.get("traceback", "")
                    if isinstance(error_detail, dict)
                    else ""
                ),
            )
        except Exception as slack_err:
            logger.warning(f"Failed to send Slack notification: {slack_err}")

    return JSONResponse(status_code=exc.status_code, content=response_body)


# Add custom exception handler to capture all errors
@app.exception_handler(Exception)
async def universal_exception_handler(request: Request, exc: Exception):
    """Catch all unhandled exceptions and return detailed error"""
    error_id = str(uuid.uuid4())

    # Log the full error
    logger.error(f"Unhandled exception [{error_id}]: {type(exc).__name__}: {str(exc)}")
    logger.error(f"Full traceback: {traceback.format_exc()}")

    # Build detailed error response
    error_detail = {
        "error": f"Unhandled {type(exc).__name__}",
        "error_type": type(exc).__name__,
        "message": str(exc),
        "error_id": error_id,
        "timestamp": datetime.now().isoformat(),
        "path": str(request.url.path),
        "method": request.method,
        "traceback": traceback.format_exc(),
    }

    # Send Slack notification for unhandled errors
    if SLACK_ENABLED and slack_notifier:
        try:
            slack_notifier.send_error_alert(
                error_type=type(exc).__name__,
                error_message=str(exc),
                service="api",
                context={
                    "error_id": error_id,
                    "path": str(request.url.path),
                    "method": request.method,
                },
                traceback_str=traceback.format_exc(),
            )
        except Exception as slack_err:
            logger.warning(f"Failed to send Slack notification: {slack_err}")

    return JSONResponse(
        status_code=500,
        content={
            "detail": error_detail,
            "error_msg": str(exc),
            "error_msg_dev": f"{type(exc).__name__}: {str(exc)} [Error ID: {error_id}]",
        },
    )


# Add request/response logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and responses for debugging"""
    request_id = str(uuid.uuid4())
    start_time = time.time()

    # Check if this is a bot/scanner request - skip detailed logging for these
    is_bot_request = _is_bot_scanner_request(str(request.url.path))

    # Log request (skip detailed logging for bot requests, but log at DEBUG level if needed)
    if not is_bot_request:
        try:
            log_api_request(
                method=request.method,
                path=str(request.url.path),
                params=dict(request.query_params),
                request_id=request_id,
            )
        except Exception as e:
            logger.warning(f"Failed to log request: {e}")
    else:
        # Optionally log bot requests at DEBUG level for monitoring
        logger.debug(
            f"Bot/scanner request detected: {request.method} {request.url.path}"
        )

    # Process request
    try:
        response = await call_next(request)
        process_time = time.time() - start_time

        # Log response (skip for bot requests to reduce noise)
        if not is_bot_request:
            try:
                log_api_response(
                    status_code=response.status_code,
                    response_time=process_time,
                    request_id=request_id,
                )
            except Exception as e:
                logger.warning(f"Failed to log response: {e}")

        return response

    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Request failed after {process_time:.2f}s: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise


repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))

# Add these configurations
UPLOAD_DIR = os.path.join(repo_path, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


class RunMonoRequest(BaseModel):
    video_path: str = ""
    metadata_path: str = ""
    calib_path: str = ""
    intrinsics_path: str = ""
    estimate_local_only: bool = False
    rerun: bool = False
    session_id: Optional[str] = None
    activity: Optional[str] = None


class WebRunMonoRequest(BaseModel):
    height_m: float
    mass_kg: float
    sex: str
    estimate_local_only: bool = False
    rerun: bool = False


class DownloadLastVideoRequest(BaseModel):
    session_id: str = "78623732-6072-43b0-87f9-ab7fdaaf4236"  # Default test session ID
    height_m: float = 1.60  # 160 cm in meters
    mass_kg: float = 60.0  # 60 kg
    sex: str = "f"  # Female


class BatchRunMonoRequest(BaseModel):
    video_folder_path: str  # Path to folder containing videos
    metadata_path: str  # Path to metadata file (shared across all videos)
    calib_path: str  # Path to calibration file (shared across all videos)
    intrinsics_path: str  # Path to intrinsics file (shared across all videos)
    estimate_local_only: bool = False
    rerun: bool = False
    session_id: Optional[str] = None
    activity: Optional[str] = None
    supported_extensions: Optional[list] = None  # List of supported video extensions


@app.get("/")
def read_root():
    return {
        "message": "Hello users! This is the Mono API for running the opencap-mono pipeline. Feel free to reach out to us for any queries."
    }


# TODO: load the WHAM model during startup
@app.on_event("startup")
async def startup_event():
    """Load WHAM model during application startup."""
    logger.info("Loading WHAM model...")
    try:
        from WHAM.demo import initialize_wham

        initialize_wham()
        logger.info("WHAM model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load WHAM model during startup: {e}")
        raise e


@app.post("/run_mono")
def run_mono(request: RunMonoRequest, api_key: bool = Depends(verify_api_key)):
    try:
        video_path = _resolve_path(request.video_path)
        metadata_path = _resolve_path(request.metadata_path)
        calib_path = _resolve_path(request.calib_path)
        intrinsics_path = _resolve_path(request.intrinsics_path)
        estimate_local_only = request.estimate_local_only
        rerun = request.rerun
        session_id = request.session_id
        activity = request.activity

        # Validate all input paths first
        logger.info(f"Validating input paths...")

        # Check video file
        if not os.path.exists(video_path):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Video file not found",
                    "error_type": "FileNotFoundError",
                    "file_path": video_path,
                    "message": f"The video file does not exist at path: {video_path}",
                    "suggestion": "Please verify the video file path is correct and the file exists",
                },
            )

        # Check metadata file
        if not os.path.exists(metadata_path):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Metadata file not found",
                    "error_type": "FileNotFoundError",
                    "file_path": metadata_path,
                    "message": f"The metadata file does not exist at path: {metadata_path}",
                    "suggestion": "Please verify the metadata file path is correct and the file exists",
                },
            )

        # Check calibration file
        if not os.path.exists(calib_path):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Calibration file not found",
                    "error_type": "FileNotFoundError",
                    "file_path": calib_path,
                    "message": f"The calibration file does not exist at path: {calib_path}",
                    "suggestion": "Please verify the calibration file path is correct and the file exists",
                },
            )

        # Check intrinsics file
        if not os.path.exists(intrinsics_path):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Intrinsics file not found",
                    "error_type": "FileNotFoundError",
                    "file_path": intrinsics_path,
                    "message": f"The intrinsics file does not exist at path: {intrinsics_path}",
                    "suggestion": "Please verify the intrinsics file path is correct and the file exists",
                },
            )

        # Load metadata with error handling
        logger.info(f"Loading metadata from: {metadata_path}")
        try:
            with open(metadata_path, "r") as f:
                metadata = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Failed to parse metadata file",
                    "error_type": "YAMLParseError",
                    "file_path": metadata_path,
                    "message": f"The metadata file is not valid YAML: {str(e)}",
                    "suggestion": "Please check the metadata file format and ensure it's valid YAML",
                },
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Failed to read metadata file",
                    "error_type": type(e).__name__,
                    "file_path": metadata_path,
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                },
            )

        height_m = metadata.get("height_m", 1.70)  # Default height if not found
        mass_kg = metadata.get("mass_kg", 70.0)  # Default mass if not found
        sex = metadata.get("sex", "male")  # Default sex if not found
        logger.info(f"Height: {height_m} m, Mass: {mass_kg} kg, Sex: {sex}")

        # Start timing for processing
        start_time = time.time()

        # If rerun is False, check for cached results
        if not rerun:
            request_hash = generate_request_hash(video_path, metadata)
            results_dir = os.path.join(repo_path, "results")

            if os.path.exists(results_dir):
                for case_dir in os.listdir(results_dir):
                    cache_file = os.path.join(results_dir, case_dir, "request_hash.txt")
                    if os.path.exists(cache_file):
                        with open(cache_file, "r") as f:
                            stored_hash = f.read().strip()

                        if stored_hash == request_hash:
                            logger.info(f"Found cached results in {case_dir}")
                            video_name = os.path.basename(video_path).split(".")[0]
                            results_path = os.path.join(
                                results_dir, case_dir, video_name
                            )

                            # Check if key files exist
                            ik_file_path = os.path.join(results_path, "ik_results.pkl")
                            output_mono_json_path = os.path.join(
                                results_path, "mono.json"
                            )
                            output_video_path = os.path.join(
                                results_path, "viewer_mono.webm"
                            )

                            if os.path.exists(ik_file_path) and os.path.exists(
                                output_mono_json_path
                            ):
                                # Calculate processing time (should be very fast for cached results)
                                processing_duration = time.time() - start_time

                                # Get trial ID
                                trial_id = session_id or case_dir

                                # Format processing time nicely
                                if processing_duration < 60:
                                    processing_time_str = (
                                        f"{processing_duration:.2f} seconds"
                                    )
                                else:
                                    minutes = int(processing_duration // 60)
                                    seconds = processing_duration % 60
                                    processing_time_str = f"{minutes} minute{'s' if minutes != 1 else ''} {seconds:.2f} seconds"

                                # Send Slack notification for cached result
                                if SLACK_ENABLED and slack_notifier:
                                    try:
                                        slack_notifier.send_notification(
                                            title="Mono API - Cached Result",
                                            message=f"Returned cached result for trial",
                                            level=AlertLevel.SUCCESS,
                                            fields=[
                                                {
                                                    "title": "Trial ID",
                                                    "value": f"`{trial_id}`",
                                                    "short": True,
                                                },
                                                {
                                                    "title": "Processing Time",
                                                    "value": processing_time_str,
                                                    "short": True,
                                                },
                                                {
                                                    "title": "Endpoint",
                                                    "value": "`/run_mono`",
                                                    "short": True,
                                                },
                                                {
                                                    "title": "Status",
                                                    "value": "Cached",
                                                    "short": True,
                                                },
                                            ],
                                        )
                                    except Exception as e:
                                        logger.warning(
                                            f"Failed to send Slack notification for cached result: {e}"
                                        )

                                return {
                                    "message": "Mono pipeline completed successfully (cached result)!",
                                    "ik_file_path": ik_file_path,
                                    "case_id": case_dir,
                                    "case_dir": os.path.join(results_dir, case_dir),
                                    "visualization": {
                                        "created": True,
                                        "json_path": output_mono_json_path,
                                        "video_path": (
                                            output_video_path
                                            if os.path.exists(output_video_path)
                                            else None
                                        ),
                                    },
                                }

        # Create a case directory based on timestamp and request parameters
        if session_id:
            case_num = session_id
            logger.info(f"Using provided session_id as case_id: {case_num}")
        else:
            timestamp = int(time.time())
            # Hash includes parameters that might affect processing but aren't part of the input 'identity' for caching
            case_hash = md5(f"{timestamp}_{estimate_local_only}".encode()).hexdigest()[
                :8
            ]  # rerun is handled by the cache check logic
            case_num = f"{timestamp}_{case_hash}"
            logger.info(f"Generated new case_id: {case_num}")

        # Create case directory structure
        case_dir = os.path.join(repo_path, "results", case_num)
        os.makedirs(case_dir, exist_ok=True)

        logger.info(f"Created case directory: {case_dir}")

        # Create output directory
        video_name = os.path.basename(video_path).split(".")[0]  # Remove file extension
        # trial_path = os.path.join(case_dir, video_name)
        trial_path = case_dir
        logger.info(f"trial_path: {trial_path}")
        if not os.path.exists(trial_path):
            os.makedirs(trial_path)

        logger.info(f"video_path: {video_path}")
        logger.info(f"metadata_path: {metadata_path}")
        logger.info(f"calib_path: {calib_path}")
        logger.info(f"intrinsics_path: {intrinsics_path}")

        static_camera_checked = False

        if ENABLE_STATIC_CAMERA_CHECK:
            logger.info("Running static camera check on input video...")
            try:
                _run_static_camera_check_or_raise(video_path)
                static_camera_checked = True
            except HTTPException:
                raise
            except Exception as e:
                logger.warning(
                    f"Static camera check could not evaluate input video {video_path}: {e}"
                )

        # Check video rotation with error handling
        logger.info("Checking video orientation...")
        try:
            rotation = getVideoRotation(video_path)
            if rotation is not None:
                logger.info(f"Detected video rotation: {rotation}")
            else:
                logger.info("No video rotation metadata detected")
        except Exception as e:
            logger.warning(
                f"Failed to detect video rotation: {str(e)}. Assuming no rotation needed."
            )
            rotation = None

        # Convert MOV to AVI if necessary
        if video_path.lower().endswith(".mov"):
            logger.info(f"Converting MOV file to AVI: {video_path}")
            try:
                video_path = convert_to_avi(video_path, rotation=rotation)
                logger.info(f"Conversion complete. New video path: {video_path}")
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "Video conversion failed",
                        "error_type": type(e).__name__,
                        "message": f"Failed to convert MOV to AVI: {str(e)}",
                        "original_video_path": request.video_path,
                        "suggestion": "Check if ffmpeg is installed and the video file is not corrupted",
                        "traceback": traceback.format_exc(),
                    },
                )

        if ENABLE_STATIC_CAMERA_CHECK and not static_camera_checked:
            logger.info("Running static camera check on converted/normalized video...")
            try:
                _run_static_camera_check_or_raise(video_path)
            except HTTPException:
                raise
            except Exception as e:
                logger.warning(
                    f"Static camera check could not evaluate normalized video {video_path}: {e}"
                )

        # Run WHAM with comprehensive error handling
        logger.info("Starting WHAM processing...")
        inputs_wham = {
            "calib_path": calib_path,
            "intrinsics_path": intrinsics_path,
            "video_path": video_path,
            "output_path": trial_path,
            "visualize": False,
            "estimate_local_only": estimate_local_only,
            "save_pkl": True,
            "run_smplify": True,
            "rerun": rerun,
        }

        try:
            results = main_wham(**inputs_wham)
            logger.info("WHAM processing completed successfully")
            logger.info(f"Time taken for WHAM: {time.time() - start_time:.2f} seconds")
        except InsufficientFullBodyKeypointsError as e:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Invalid recording: body not fully visible or insufficient 2D keypoint confidence",
                    "error_type": "InsufficientFullBodyKeypointsError",
                    "message": str(e),
                    "stage": "WHAM_preprocess",
                    "suggestion": "Make sure your recording is contains a full body and is not heavily occluded.",
                },
            )
        except FileNotFoundError as e:
            error_detail = {
                "error": "WHAM processing failed - File not found",
                "error_type": "FileNotFoundError",
                "message": str(e),
                "stage": "WHAM",
                "inputs": inputs_wham,
                "suggestion": "Check if all WHAM dependencies and model files are present",
                "traceback": traceback.format_exc(),
            }
            # Send Slack notification
            send_trial_failure_notification(
                error=e,
                stage="WHAM",
                session_id=session_id or case_num,
                video_path=video_path,
                metadata_path=metadata_path,
                error_detail=error_detail,
                start_time=start_time,
            )
            raise HTTPException(status_code=500, detail=error_detail)
        except RuntimeError as e:
            error_detail = {
                "error": "WHAM processing failed - Runtime error",
                "error_type": "RuntimeError",
                "message": str(e),
                "stage": "WHAM",
                "inputs": inputs_wham,
                "suggestion": "This might be a GPU/CUDA error. Check if GPU is available and CUDA is properly installed",
                "traceback": traceback.format_exc(),
            }
            # Send Slack notification
            send_trial_failure_notification(
                error=e,
                stage="WHAM",
                session_id=session_id or case_num,
                video_path=video_path,
                metadata_path=metadata_path,
                error_detail=error_detail,
                start_time=start_time,
            )
            raise HTTPException(status_code=500, detail=error_detail)
        except ValueError as e:
            msg = str(e)
            if msg in _WHAM_USER_FRIENDLY_VALUE_ERRORS:
                user_err, suggestion = _WHAM_USER_FRIENDLY_VALUE_ERRORS[msg]
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": user_err,
                        "error_type": "ValueError",
                        "message": msg,
                        "stage": "WHAM",
                        "suggestion": suggestion,
                    },
                )
            error_detail = {
                "error": "WHAM processing failed",
                "error_type": "ValueError",
                "message": msg,
                "stage": "WHAM",
                "inputs": inputs_wham,
                "video_path": video_path,
                "output_path": trial_path,
                "traceback": traceback.format_exc(),
            }
            send_trial_failure_notification(
                error=e,
                stage="WHAM",
                session_id=session_id or case_num,
                video_path=video_path,
                metadata_path=metadata_path,
                error_detail=error_detail,
                start_time=start_time,
            )
            raise HTTPException(status_code=500, detail=error_detail)
        except Exception as e:
            error_detail = {
                "error": "WHAM processing failed",
                "error_type": type(e).__name__,
                "message": str(e),
                "stage": "WHAM",
                "inputs": inputs_wham,
                "video_path": video_path,
                "output_path": trial_path,
                "traceback": traceback.format_exc(),
            }
            # Send Slack notification
            send_trial_failure_notification(
                error=e,
                stage="WHAM",
                session_id=session_id or case_num,
                video_path=video_path,
                metadata_path=metadata_path,
                error_detail=error_detail,
                start_time=start_time,
            )
            raise HTTPException(status_code=500, detail=error_detail)

        # extract video name from video path. it's the last part of the path and without the extension
        video_name = os.path.basename(video_path).split(".")[0]

        # Run optimization
        results_path = (
            trial_path  # Use trial_path directly instead of joining with video basename
        )
        logger.info(f"results_path: {results_path}")

        # the result path is result path + the video name
        results_path = os.path.join(results_path, video_name)

        # Run the optimization
        logger.info("Running optimization...")

        inputs_optimization = {
            "data_dir": results_path,
            "trial_name": os.path.basename(
                results_path
            ),  # Use folder name instead of video filename
            "height_m": height_m,
            "mass_kg": mass_kg,
            "sex": sex,
            "intrinsics_pth": intrinsics_path,
            "run_opensim_original_wham": False,
            "run_opensim_opt2": True,
            "use_gpu": True,
            "static_cam": False,
            "n_iter_opt2": 75,
            "print_loss_terms": False,
            "plotting": False,
            "save_video_debug": False,
            "output_path": results_path,
            "video_path": video_path,
            "activity": activity,
        }

        # Keep compatibility across optimization.py variants by only passing
        # kwargs that the current run_optimization signature accepts.
        valid_opt_keys = set(inspect.signature(run_optimization).parameters.keys())
        inputs_optimization = {
            k: v for k, v in inputs_optimization.items() if k in valid_opt_keys
        }

        try:
            output_paths = run_optimization(**inputs_optimization)
            logger.info("Optimization completed successfully")
            logger.info(
                f"Time taken for optimization: {time.time() - start_time:.2f} seconds"
            )
        except FileNotFoundError as e:
            error_detail = {
                "error": "Optimization failed - File not found",
                "error_type": "FileNotFoundError",
                "message": str(e),
                "stage": "Optimization",
                "inputs": inputs_optimization,
                "suggestion": "Check if WHAM output files exist and OpenSim models are available",
                "traceback": traceback.format_exc(),
            }
            # Send Slack notification
            send_trial_failure_notification(
                error=e,
                stage="Optimization",
                session_id=session_id or case_num,
                video_path=video_path,
                metadata_path=metadata_path,
                error_detail=error_detail,
                start_time=start_time,
            )
            raise HTTPException(status_code=500, detail=error_detail)
        except RuntimeError as e:
            error_detail = {
                "error": "Optimization failed - Runtime error",
                "error_type": "RuntimeError",
                "message": str(e),
                "stage": "Optimization",
                "inputs": inputs_optimization,
                "suggestion": "This might be a GPU/numerical optimization error. Check GPU availability and input data validity",
                "traceback": traceback.format_exc(),
            }
            # Send Slack notification
            send_trial_failure_notification(
                error=e,
                stage="Optimization",
                session_id=session_id or case_num,
                video_path=video_path,
                metadata_path=metadata_path,
                error_detail=error_detail,
                start_time=start_time,
            )
            raise HTTPException(status_code=500, detail=error_detail)
        except ValueError as e:
            error_detail = {
                "error": "Optimization failed - Invalid input",
                "error_type": "ValueError",
                "message": str(e),
                "stage": "Optimization",
                "inputs": inputs_optimization,
                "suggestion": "Check if height, mass, and sex values are valid",
                "traceback": traceback.format_exc(),
            }
            # Send Slack notification
            send_trial_failure_notification(
                error=e,
                stage="Optimization",
                session_id=session_id or case_num,
                video_path=video_path,
                metadata_path=metadata_path,
                error_detail=error_detail,
                start_time=start_time,
            )
            raise HTTPException(status_code=400, detail=error_detail)
        except Exception as e:
            error_detail = {
                "error": "Optimization failed",
                "error_type": type(e).__name__,
                "message": str(e),
                "stage": "Optimization",
                "inputs": inputs_optimization,
                "traceback": traceback.format_exc(),
            }
            # Send Slack notification
            send_trial_failure_notification(
                error=e,
                stage="Optimization",
                session_id=session_id or case_num,
                video_path=video_path,
                metadata_path=metadata_path,
                error_detail=error_detail,
                start_time=start_time,
            )
            raise HTTPException(status_code=500, detail=error_detail)
        finally:
            # Always reset the default tensor type after optimization so that a
            # failed run never leaves the process-wide default pointing at CUDA,
            # which causes CUDAGuardImpl assertion failures on subsequent requests.
            torch.set_default_tensor_type("torch.FloatTensor")

        logger.info(f"output_paths: {output_paths}")

        # Generate visualization
        logger.info("Generating visualization...")
        visualization_created = False
        jsonOutputPath = None

        try:
            # Get paths for visualization
            model_mono_sub_folder = os.path.join(results_path, "OpenSim", "Model")

            # find the folder which does not contain 'wham' in the name
            try:
                non_wham_folders = [
                    x
                    for x in os.listdir(model_mono_sub_folder)
                    if "wham" not in x
                    and os.path.isdir(os.path.join(model_mono_sub_folder, x))
                ]
                if not non_wham_folders:
                    raise FileNotFoundError("No non-WHAM model folder found")
                model_mono_folder = os.path.join(
                    model_mono_sub_folder, non_wham_folders[0]
                )
            except Exception as e:
                raise FileNotFoundError(f"Failed to locate model folder: {str(e)}")

            model_mono_file = os.path.join(
                model_mono_folder, "LaiUhlrich2022_scaled_no_patella.osim"
            )

            if not os.path.exists(model_mono_file):
                raise FileNotFoundError(f"Model file not found: {model_mono_file}")

            ik_motion_sub_folder = os.path.join(results_path, "OpenSim", "IK")

            # find the folder which does not contain 'wham' in the name
            try:
                non_wham_ik_folders = [
                    x
                    for x in os.listdir(ik_motion_sub_folder)
                    if "wham" not in x
                    and os.path.isdir(os.path.join(ik_motion_sub_folder, x))
                ]
                if not non_wham_ik_folders:
                    raise FileNotFoundError("No non-WHAM IK folder found")
                ik_motion_folder = os.path.join(
                    ik_motion_sub_folder, non_wham_ik_folders[0]
                )
            except Exception as e:
                raise FileNotFoundError(f"Failed to locate IK folder: {str(e)}")

            # Find .mot file
            try:
                mot_files = [
                    x for x in os.listdir(ik_motion_folder) if x.endswith(".mot")
                ]
                if not mot_files:
                    raise FileNotFoundError("No .mot file found in IK folder")
                ik_motion_file = os.path.join(ik_motion_folder, mot_files[0])
            except Exception as e:
                raise FileNotFoundError(f"Failed to locate IK motion file: {str(e)}")

            output_mono_json_path = os.path.join(results_path, "mono.json")
            output_video_path = os.path.join(results_path, "viewer_mono.webm")

            logger.info(f"model_mono_file: {model_mono_file}")
            logger.info(f"ik_motion_file: {ik_motion_file}")
            logger.info(f"output_mono_json_path: {output_mono_json_path}")
            logger.info(f"output_video_path: {output_video_path}")

            # Generate JSON for visualization
            try:
                jsonOutputPath = generateVisualizerJson(
                    modelPath=model_mono_file,
                    ikPath=ik_motion_file,
                    jsonOutputPath=output_mono_json_path,
                    vertical_offset=0,
                )
                logger.info(f"Generated visualization JSON file at {jsonOutputPath}")

                viz = False
                if viz:
                    # Create visualization video
                    automate_recording(
                        json_paths=[output_mono_json_path],
                        output_video_path=output_video_path,
                        num_loops=1,
                    )
                    logger.info("Generated visualization video")

                visualization_created = True
            except Exception as e:
                logger.error(f"Error in visualization JSON generation: {str(e)}")
                logger.error(traceback.format_exc())
                visualization_created = False

        except FileNotFoundError as e:
            logger.error(f"Visualization file setup failed: {str(e)}")
            logger.error(traceback.format_exc())
            visualization_created = False
        except Exception as e:
            logger.error(f"Unexpected error in visualization generation: {str(e)}")
            logger.error(traceback.format_exc())
            visualization_created = False

        ik_file_path = os.path.join(results_path, "ik_results.pkl")

        # Store the request hash before returning
        request_hash = generate_request_hash(video_path, metadata)
        hash_file_path = os.path.join(case_dir, "request_hash.txt")
        with open(hash_file_path, "w") as f:
            f.write(request_hash)
        logger.info(f"Stored request hash {request_hash} for future reference")

        # Calculate total processing time
        processing_duration = time.time() - start_time

        # Get trial ID (use session_id if provided, otherwise use case_num)
        trial_id = session_id or case_num

        # Format processing time nicely
        if processing_duration < 60:
            processing_time_str = f"{processing_duration:.2f} seconds"
        else:
            minutes = int(processing_duration // 60)
            seconds = processing_duration % 60
            processing_time_str = (
                f"{minutes} minute{'s' if minutes != 1 else ''} {seconds:.2f} seconds"
            )

        # Send Slack notification for successful completion
        if SLACK_ENABLED and slack_notifier:
            try:
                slack_notifier.send_notification(
                    title="Mono API - Processing Complete",
                    message=f"Successfully processed trial",
                    level=AlertLevel.SUCCESS,
                    fields=[
                        {"title": "Trial ID", "value": f"`{trial_id}`", "short": True},
                        {
                            "title": "Processing Time",
                            "value": processing_time_str,
                            "short": True,
                        },
                        {"title": "Endpoint", "value": "`/run_mono`", "short": True},
                    ],
                )
            except Exception as e:
                logger.warning(
                    f"Failed to send Slack notification for successful processing: {e}"
                )

        return {
            "message": "Mono pipeline completed successfully!",
            "ik_file_path": output_paths["ik_results_file"],
            "json_file_path": jsonOutputPath,
            "video_file_path": output_paths["trimmed_video"],
            "trc_file_path": output_paths["trc_file"],
            "scaled_model_file_path": output_paths["scaled_model_file"],
            "neutral_trc_file_path": output_paths.get("neutral_trc_file"),
            "case_id": case_num,
            "case_dir": case_dir,
            "metadata_path": metadata_path,
            "pose_pickle_path": output_paths["optimized_pkl"],
            "keypoints_3d_cam_path": output_paths.get("keypoints_3d_cam_pkl"),
            "vertices_3d_cam_path": output_paths.get("vertices_3d_cam_pkl"),
            "predicted_activity": output_paths.get("predicted_activity"),
            "activity_detection_method": output_paths.get("activity_detection_method"),
            "visualization": {
                "created": visualization_created,
                "json_path": jsonOutputPath if visualization_created else None,
                "video_path": (
                    os.path.join(results_path, "viewer_mono.webm")
                    if visualization_created
                    and os.path.exists(os.path.join(results_path, "viewer_mono.webm"))
                    else None
                ),
            },
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Catch-all for any unexpected errors
        logger.error(f"Unexpected error in run_mono: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")

        error_detail = {
            "error": "Mono pipeline failed with unexpected error",
            "error_type": type(e).__name__,
            "message": str(e),
            "timestamp": datetime.now().isoformat(),
            "request_params": {
                "video_path": request.video_path,
                "metadata_path": request.metadata_path,
                "calib_path": request.calib_path,
                "intrinsics_path": request.intrinsics_path,
                "estimate_local_only": request.estimate_local_only,
                "rerun": request.rerun,
                "session_id": request.session_id,
                "activity": request.activity,
            },
            "traceback": traceback.format_exc(),
        }

        # Send Slack notification
        send_trial_failure_notification(
            error=e,
            stage="Unknown",
            session_id=request.session_id,
            video_path=request.video_path,
            metadata_path=request.metadata_path,
            error_detail=error_detail,
            start_time=None,
        )

        raise HTTPException(status_code=500, detail=error_detail)


# Add this function to generate a unique hash for a video request
def generate_request_hash(video_path, metadata):
    """Generate a unique hash to identify a processing request based on its parameters."""
    # Create a string with all relevant parameters
    # Use 'sex' key consistently, providing defaults if missing
    sex = metadata.get("sex", "unknown")
    height = metadata.get("height_m", 0.0)
    mass = metadata.get("mass_kg", 0.0)
    key_string = f"{video_path}_{height}_{mass}_{sex}"
    # Generate a hash
    return hashlib.md5(key_string.encode()).hexdigest()


def send_trial_failure_notification(
    error: Exception,
    stage: str,
    session_id: str = None,
    video_path: str = None,
    metadata_path: str = None,
    error_detail: dict = None,
    start_time: float = None,
):
    """
    Send Slack notification for trial failure with comprehensive details

    Args:
        error: The exception that occurred
        stage: Processing stage where error occurred (e.g., "WHAM", "Optimization")
        session_id: Trial/session ID
        video_path: Path to video file
        metadata_path: Path to metadata file
        error_detail: Detailed error information dictionary
        start_time: Processing start time for duration calculation
    """
    if not SLACK_ENABLED or not slack_notifier:
        return

    try:
        # Calculate duration if start_time provided
        duration = None
        if start_time:
            duration = time.time() - start_time

        # Build error message
        error_msg = str(error)
        if error_detail and isinstance(error_detail, dict):
            error_msg = error_detail.get("message", str(error))

        # Get traceback
        tb_str = traceback.format_exc()

        # Prepare trial information
        trial_info = {
            "session_id": session_id or "N/A",
            "video_path": video_path or "N/A",
            "metadata_path": metadata_path or "N/A",
            "stage": stage,
            "error_type": type(error).__name__,
            "error_message": error_msg,
        }

        # Send notification
        slack_notifier.send_notification(
            title=f"Trial Failed - {stage}",
            message=f"❌ Trial processing failed at {stage} stage",
            level=AlertLevel.ERROR if AlertLevel else "error",
            fields=[
                {
                    "title": "Session/Trial ID",
                    "value": f"`{session_id or 'Unknown'}`",
                    "short": True,
                },
                {"title": "Failed Stage", "value": f"`{stage}`", "short": True},
                {
                    "title": "Error Type",
                    "value": f"`{type(error).__name__}`",
                    "short": True,
                },
                {
                    "title": "Duration Before Failure",
                    "value": f"{duration:.2f}s" if duration else "N/A",
                    "short": True,
                },
                {
                    "title": "Video File",
                    "value": f"`{os.path.basename(video_path) if video_path else 'N/A'}`",
                    "short": False,
                },
                {
                    "title": "Error Message",
                    "value": f"```{error_msg[:300]}```",
                    "short": False,
                },
                {
                    "title": "Traceback (preview)",
                    "value": f"```{tb_str[:400]}```" if tb_str else "N/A",
                    "short": False,
                },
            ],
        )

        logger.info(
            f"Sent Slack notification for trial failure: {session_id or 'unknown'}"
        )

    except Exception as e:
        logger.warning(f"Failed to send Slack notification for trial failure: {e}")


def discover_video_files(folder_path, supported_extensions=None):
    """Discover all video files in a folder with supported extensions."""
    if supported_extensions is None:
        supported_extensions = [".avi", ".mov", ".mp4", ".mkv", ".webm"]

    video_files = []

    if not os.path.exists(folder_path):
        raise ValueError(f"Folder path does not exist: {folder_path}")

    if not os.path.isdir(folder_path):
        raise ValueError(f"Path is not a directory: {folder_path}")

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            _, ext = os.path.splitext(file.lower())
            if ext in [e.lower() for e in supported_extensions]:
                video_files.append(file_path)

    if not video_files:
        raise ValueError(f"No video files found in folder: {folder_path}")

    logger.info(f"Found {len(video_files)} video files in {folder_path}")
    return sorted(video_files)  # Return sorted for consistent ordering


@app.post("/run_mono_batch")
def run_mono_batch(
    request: BatchRunMonoRequest, api_key: bool = Depends(verify_api_key)
):
    """Process multiple videos from a folder with shared metadata and intrinsics."""
    try:
        # Extract request parameters
        video_folder_path = request.video_folder_path
        metadata_path = _resolve_path(request.metadata_path)
        calib_path = _resolve_path(request.calib_path)
        intrinsics_path = _resolve_path(request.intrinsics_path)
        estimate_local_only = request.estimate_local_only
        rerun = request.rerun
        session_id = request.session_id
        activity = request.activity
        supported_extensions = request.supported_extensions

        # Validate input paths
        if not os.path.exists(metadata_path):
            raise HTTPException(status_code=400, detail="Metadata file does not exist.")
        if not os.path.exists(calib_path):
            raise HTTPException(
                status_code=400, detail="Calibration file does not exist."
            )
        if not os.path.exists(intrinsics_path):
            raise HTTPException(
                status_code=400, detail="Intrinsics file does not exist."
            )

        # Load metadata once (shared across all videos)
        with open(metadata_path, "r") as f:
            metadata = yaml.safe_load(f)

        height_m = metadata.get("height_m", 1.70)
        mass_kg = metadata.get("mass_kg", 70.0)
        sex = metadata.get("sex", "male")
        logger.info(
            f"Shared metadata - Height: {height_m} m, Mass: {mass_kg} kg, Sex: {sex}"
        )

        # Discover video files
        try:
            video_files = discover_video_files(video_folder_path, supported_extensions)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Generate batch session ID if not provided
        if session_id:
            batch_case_id = session_id
            logger.info(f"Using provided session_id as batch case_id: {batch_case_id}")
        else:
            timestamp = int(time.time())
            batch_hash = md5(
                f"{timestamp}_{estimate_local_only}_batch".encode()
            ).hexdigest()[:8]
            batch_case_id = f"batch_{timestamp}_{batch_hash}"
            logger.info(f"Generated new batch case_id: {batch_case_id}")

        # Create batch results directory
        batch_case_dir = os.path.join(repo_path, "results", batch_case_id)
        os.makedirs(batch_case_dir, exist_ok=True)
        logger.info(f"Created batch case directory: {batch_case_dir}")

        # Initialize results tracking
        batch_results = {
            "batch_case_id": batch_case_id,
            "batch_case_dir": batch_case_dir,
            "total_videos": len(video_files),
            "processed_videos": 0,
            "failed_videos": 0,
            "results": [],
            "shared_metadata": {"height_m": height_m, "mass_kg": mass_kg, "sex": sex},
            "shared_paths": {
                "metadata_path": metadata_path,
                "calib_path": calib_path,
                "intrinsics_path": intrinsics_path,
            },
        }

        # Process each video
        for i, video_path in enumerate(video_files, 1):
            video_name = os.path.basename(video_path).split(".")[0]
            logger.info(f"Processing video {i}/{len(video_files)}: {video_name}")

            try:
                # Check for cached results first (if rerun is False)
                cached_result = None
                if not rerun:
                    request_hash = generate_request_hash(video_path, metadata)
                    # Check if this exact video with these parameters was processed before
                    results_dir = os.path.join(repo_path, "results")
                    if os.path.exists(results_dir):
                        for case_dir in os.listdir(results_dir):
                            cache_file = os.path.join(
                                results_dir, case_dir, "request_hash.txt"
                            )
                            if os.path.exists(cache_file):
                                with open(cache_file, "r") as f:
                                    stored_hash = f.read().strip()
                                if stored_hash == request_hash:
                                    logger.info(
                                        f"Found cached results for {video_name}"
                                    )
                                    results_path = os.path.join(
                                        results_dir, case_dir, video_name
                                    )
                                    ik_file_path = os.path.join(
                                        results_path, "ik_results.pkl"
                                    )
                                    output_mono_json_path = os.path.join(
                                        results_path, "mono.json"
                                    )
                                    if os.path.exists(ik_file_path) and os.path.exists(
                                        output_mono_json_path
                                    ):
                                        cached_result = {
                                            "status": "success_cached",
                                            "video_name": video_name,
                                            "video_path": video_path,
                                            "case_id": case_dir,
                                            "case_dir": os.path.join(
                                                results_dir, case_dir
                                            ),
                                            "ik_file_path": ik_file_path,
                                            "json_file_path": output_mono_json_path,
                                            "visualization": {
                                                "created": True,
                                                "json_path": output_mono_json_path,
                                                "video_path": (
                                                    os.path.join(
                                                        results_path, "viewer_mono.webm"
                                                    )
                                                    if os.path.exists(
                                                        os.path.join(
                                                            results_path,
                                                            "viewer_mono.webm",
                                                        )
                                                    )
                                                    else None
                                                ),
                                            },
                                        }
                                        break

                if cached_result:
                    batch_results["results"].append(cached_result)
                    batch_results["processed_videos"] += 1
                    continue

                # Create individual video request
                video_request = RunMonoRequest(
                    video_path=video_path,
                    metadata_path=metadata_path,
                    calib_path=calib_path,
                    intrinsics_path=intrinsics_path,
                    estimate_local_only=estimate_local_only,
                    rerun=rerun,
                    session_id=f"{batch_case_id}_{video_name}",  # Unique session ID for each video
                    activity=activity,
                )

                # Process the video
                video_result = run_mono(video_request)

                # Store request hash for future caching
                request_hash = generate_request_hash(video_path, metadata)
                hash_file_path = os.path.join(
                    video_result["case_dir"], "request_hash.txt"
                )
                with open(hash_file_path, "w") as f:
                    f.write(request_hash)

                # Format result for batch response
                batch_video_result = {
                    "status": "success",
                    "video_name": video_name,
                    "video_path": video_path,
                    "case_id": video_result["case_id"],
                    "case_dir": video_result["case_dir"],
                    "ik_file_path": video_result["ik_file_path"],
                    "json_file_path": video_result.get("json_file_path"),
                    "visualization": video_result.get("visualization", {}),
                }

                batch_results["results"].append(batch_video_result)
                batch_results["processed_videos"] += 1
                logger.info(f"Successfully processed {video_name}")

            except HTTPException as e:
                logger.error(f"Failed to process {video_name}: {e.detail}")
                error_detail = (
                    e.detail if isinstance(e.detail, dict) else {"message": str(e.detail)}
                )
                error_result = {
                    "status": "failed",
                    "video_name": video_name,
                    "video_path": video_path,
                    "error": _user_facing_http_error_msg(error_detail),
                    "error_type": error_detail.get("error_type", "HTTPException"),
                    "status_code": e.status_code,
                    "detail": error_detail,
                }
                batch_results["results"].append(error_result)
                batch_results["failed_videos"] += 1
            except Exception as e:
                logger.error(f"Failed to process {video_name}: {str(e)}")
                error_result = {
                    "status": "failed",
                    "video_name": video_name,
                    "video_path": video_path,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
                batch_results["results"].append(error_result)
                batch_results["failed_videos"] += 1

        # Store batch request hash
        batch_hash_file_path = os.path.join(batch_case_dir, "batch_request_hash.txt")
        batch_request_info = f"{video_folder_path}_{metadata_path}_{height_m}_{mass_kg}_{sex}_{estimate_local_only}"
        batch_request_hash = hashlib.md5(batch_request_info.encode()).hexdigest()
        with open(batch_hash_file_path, "w") as f:
            f.write(batch_request_hash)

        logger.info(
            f"Batch processing completed. Processed: {batch_results['processed_videos']}, Failed: {batch_results['failed_videos']}"
        )

        return {
            "message": f"Batch mono pipeline completed! Processed {batch_results['processed_videos']}/{batch_results['total_videos']} videos successfully.",
            "batch_results": batch_results,
        }

    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, detail=f"Batch processing failed: {str(e)}"
        )


@app.get("/batch_status/{batch_case_id}")
def get_batch_status(batch_case_id: str):
    """Get the status and results of a batch processing job."""
    try:
        batch_case_dir = os.path.join(repo_path, "results", batch_case_id)

        if not os.path.exists(batch_case_dir):
            raise HTTPException(
                status_code=404, detail=f"Batch case {batch_case_id} not found"
            )

        # Look for batch request hash file
        batch_hash_file = os.path.join(batch_case_dir, "batch_request_hash.txt")
        if os.path.exists(batch_hash_file):
            with open(batch_hash_file, "r") as f:
                batch_hash = f.read().strip()
        else:
            batch_hash = None

        # Count processed videos by looking for subdirectories
        processed_videos = []
        failed_videos = []

        for item in os.listdir(batch_case_dir):
            item_path = os.path.join(batch_case_dir, item)
            if os.path.isdir(item_path) and item.startswith(batch_case_id + "_"):
                # This is a video result directory
                video_name = item.replace(batch_case_id + "_", "")

                # Check if processing was successful
                ik_file = os.path.join(item_path, "ik_results.pkl")
                json_file = os.path.join(item_path, "mono.json")

                if os.path.exists(ik_file) and os.path.exists(json_file):
                    processed_videos.append(
                        {
                            "video_name": video_name,
                            "case_dir": item_path,
                            "status": "completed",
                        }
                    )
                else:
                    failed_videos.append(
                        {
                            "video_name": video_name,
                            "case_dir": item_path,
                            "status": "failed",
                        }
                    )

        return {
            "batch_case_id": batch_case_id,
            "batch_case_dir": batch_case_dir,
            "batch_hash": batch_hash,
            "total_videos": len(processed_videos) + len(failed_videos),
            "processed_videos": len(processed_videos),
            "failed_videos": len(failed_videos),
            "processed_list": processed_videos,
            "failed_list": failed_videos,
        }

    except Exception as e:
        logger.error(f"Failed to get batch status: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get batch status: {str(e)}"
        )


@app.post("/download_last_video_and_run_mono")
async def download_last_video_and_run_mono(
    request: DownloadLastVideoRequest, api_key: bool = Depends(verify_api_key)
):
    try:
        session_id = request.session_id
        height_m = request.height_m
        mass_kg = request.mass_kg
        sex = request.sex

        # Validate inputs
        if not session_id:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid session ID",
                    "error_type": "ValueError",
                    "message": "Session ID cannot be empty",
                    "suggestion": "Please provide a valid session ID",
                },
            )

        # Construct command to run the download script
        download_script = os.path.join(
            repo_path, "dynamics/opencap-processing-grf/batchDownload.py"
        )

        # Check if download script exists
        if not os.path.exists(download_script):
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Download script not found",
                    "error_type": "FileNotFoundError",
                    "script_path": download_script,
                    "message": f"Download script does not exist at: {download_script}",
                    "suggestion": "Check if the dynamics/opencap-processing-grf module is properly installed",
                },
            )

        cmd = ["python", download_script, session_id]

        logger.info(f"Running download command: {' '.join(cmd)}")

        # Execute the download command with comprehensive error handling
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=300,  # 5 minute timeout
            )
            logger.info(f"Download completed successfully for session ID: {session_id}")
            logger.debug(f"Command output: {result.stdout}")
        except subprocess.TimeoutExpired:
            raise HTTPException(
                status_code=504,
                detail={
                    "error": "Download timeout",
                    "error_type": "TimeoutError",
                    "message": "Download script exceeded 5 minute timeout",
                    "session_id": session_id,
                    "command": " ".join(cmd),
                    "suggestion": "The server might be slow or the session has a lot of data. Try again or check the server status",
                },
            )
        except subprocess.CalledProcessError as e:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Download script failed",
                    "error_type": "SubprocessError",
                    "message": f"Download script exited with code {e.returncode}",
                    "session_id": session_id,
                    "command": " ".join(cmd),
                    "stdout": e.stdout,
                    "stderr": e.stderr,
                    "suggestion": "Check if the session ID is valid and the OpenCap server is accessible",
                },
            )
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Python executable not found",
                    "error_type": "FileNotFoundError",
                    "message": str(e),
                    "command": " ".join(cmd),
                    "suggestion": "Ensure Python is installed and in the system PATH",
                },
            )

        # Parse the output to get video_path and metadata_path
        video_path = None
        metadata_path = None

        try:
            # Try to parse as JSON first
            download_result = json.loads(result.stdout)
            video_path = download_result.get("video_path")
            metadata_path = download_result.get("metadata_path")
            logger.info("Successfully parsed download output as JSON")
        except json.JSONDecodeError:
            # Fallback: try to extract paths from the script output
            logger.info("Output is not JSON, parsing as text")
            stdout_lines = result.stdout.strip().split("\n")

            for line in stdout_lines:
                if "Video path:" in line:
                    video_path = line.split("Video path:")[1].strip()
                if "Metadata path:" in line:
                    metadata_path = line.split("Metadata path:")[1].strip()

        if not video_path or not metadata_path:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Failed to parse download output",
                    "error_type": "ParsingError",
                    "message": "Could not extract video_path or metadata_path from the download script output",
                    "stdout": result.stdout,
                    "suggestion": "Check the download script output format. It should include 'Video path:' and 'Metadata path:' or be valid JSON",
                },
            )

        logger.info(f"Extracted video_path: {video_path}")
        logger.info(f"Extracted metadata_path: {metadata_path}")

        # Verify downloaded files exist
        if not os.path.exists(video_path):
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Downloaded video not found",
                    "error_type": "FileNotFoundError",
                    "video_path": video_path,
                    "message": f"Video file was not downloaded or does not exist at: {video_path}",
                    "suggestion": "Check download script logs and server connectivity",
                },
            )

        if not os.path.exists(metadata_path):
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Downloaded metadata not found",
                    "error_type": "FileNotFoundError",
                    "metadata_path": metadata_path,
                    "message": f"Metadata file was not downloaded or does not exist at: {metadata_path}",
                    "suggestion": "Check download script logs and server connectivity",
                },
            )

        # Update the metadata file with user-provided information
        try:
            # Read the existing metadata file
            with open(metadata_path, "r") as f:
                metadata = yaml.safe_load(f)

            # Update the subject info
            metadata["height_m"] = height_m
            metadata["mass_kg"] = mass_kg

            # Always use 'sex' key and remove 'gender_mf' if it exists
            metadata["sex"] = sex
            if "gender_mf" in metadata:
                del metadata["gender_mf"]  # Remove gender_mf key if it exists

            # Write back to the metadata file
            with open(metadata_path, "w") as f:
                yaml.dump(metadata, f, default_flow_style=False)

            logger.info(
                f"Updated metadata file with height: {height_m}m, mass: {mass_kg}kg, sex: {sex}"
            )

        except yaml.YAMLError as e:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Failed to parse/update metadata YAML",
                    "error_type": "YAMLError",
                    "metadata_path": metadata_path,
                    "message": str(e),
                    "suggestion": "The downloaded metadata file might be corrupted or invalid YAML format",
                    "traceback": traceback.format_exc(),
                },
            )
        except PermissionError as e:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Permission denied updating metadata",
                    "error_type": "PermissionError",
                    "metadata_path": metadata_path,
                    "message": str(e),
                    "suggestion": "Check file permissions on the metadata file",
                },
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Failed to update metadata file",
                    "error_type": type(e).__name__,
                    "metadata_path": metadata_path,
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                },
            )

        # Generate a unique hash for this request
        request_hash = generate_request_hash(video_path, metadata)

        # Check if we've processed this exact video with these parameters before
        results_dir = os.path.join(repo_path, "results")
        processed_before = False
        cached_result = None

        # Look for existing results with this hash
        if os.path.exists(results_dir):
            for case_dir in os.listdir(results_dir):
                cache_file = os.path.join(results_dir, case_dir, "request_hash.txt")
                if os.path.exists(cache_file):
                    with open(cache_file, "r") as f:
                        stored_hash = f.read().strip()

                    if stored_hash == request_hash:
                        processed_before = True
                        logger.info(f"Found cached results in {case_dir}")

                        # Try to reconstruct the mono_result
                        video_name = os.path.basename(video_path).split(".")[0]
                        results_path = os.path.join(results_dir, case_dir, video_name)

                        # Check if key files exist
                        ik_file_path = os.path.join(results_path, "ik_results.pkl")
                        output_mono_json_path = os.path.join(results_path, "mono.json")
                        output_video_path = os.path.join(
                            results_path, "viewer_mono.webm"
                        )

                        if os.path.exists(ik_file_path) and os.path.exists(
                            output_mono_json_path
                        ):
                            cached_result = {
                                "message": "Mono pipeline completed successfully (cached result)!",
                                "ik_file_path": ik_file_path,
                                "case_id": case_dir,
                                "case_dir": os.path.join(results_dir, case_dir),
                                "visualization": {
                                    "created": True,
                                    "json_path": output_mono_json_path,
                                    "video_path": (
                                        output_video_path
                                        if os.path.exists(output_video_path)
                                        else None
                                    ),
                                },
                            }
                            break

        # If we found cached results, use them
        if processed_before and cached_result:
            logger.info("Using cached results instead of reprocessing")
            mono_result = cached_result
        else:
            # No cached results found, process the video.
            # Resolve device-specific intrinsics from downloaded metadata.
            calib_path = os.path.join(repo_path, "examples/walking4/calib.txt")

            device_model = None
            try:
                device_model = metadata.get("iphoneModel", {}).get("Cam0", "")
                device_model = device_model.replace("iphone", "iPhone").replace(
                    "ipad", "iPad"
                )
            except Exception:
                pass

            default_intrinsics = os.path.join(
                repo_path, "examples/Intrinsics/iphone12Pro_intrinsics.pickle"
            )
            if device_model:
                device_intrinsics = os.path.join(
                    repo_path,
                    f"camera_intrinsics/{device_model}/Deployed/cameraIntrinsics.pickle",
                )
                if os.path.exists(device_intrinsics):
                    intrinsics_path_for_run = device_intrinsics
                    logger.info(
                        f"Using device-specific intrinsics for {device_model}: {device_intrinsics}"
                    )
                else:
                    intrinsics_path_for_run = default_intrinsics
                    logger.warning(
                        f"No intrinsics found for device '{device_model}' at {device_intrinsics}. "
                        f"Falling back to {default_intrinsics}"
                    )
            else:
                intrinsics_path_for_run = default_intrinsics
                logger.warning(
                    f"Could not determine device model from metadata. "
                    f"Falling back to {default_intrinsics}"
                )

            mono_request = RunMonoRequest(
                video_path=video_path,
                metadata_path=metadata_path,
                calib_path=calib_path,
                intrinsics_path=intrinsics_path_for_run,
                estimate_local_only=False,
                rerun=False,
            )

            # Call run_mono with the prepared request
            mono_result = run_mono(mono_request)

            # Store the request hash for future reference
            hash_file_path = os.path.join(mono_result["case_dir"], "request_hash.txt")
            with open(hash_file_path, "w") as f:
                f.write(request_hash)
            logger.info(f"Stored request hash {request_hash} for future reference")

        # Get paths for downloadable files
        video_download_path = video_path
        json_download_path = mono_result.get("visualization", {}).get("json_path")

        # Return combined results with download links (Removed leading slash)
        response_data = {
            "download": {
                "message": f"Successfully downloaded video for session ID: {session_id}",
                "status": "success",
                "video_path": video_path,
                "metadata_path": metadata_path,
                # Remove leading slash from URL path
                "video_download_url": f"download_file?file_path={video_download_path}",
                # Remove leading slash from URL path
                "json_download_url": (
                    f"download_file?file_path={json_download_path}"
                    if json_download_path
                    else None
                ),
                "cached_result": processed_before,
            },
            "subject_info": {"height_m": height_m, "mass_kg": mass_kg, "sex": sex},
            "mono_processing": mono_result,
        }
        return JSONResponse(content=response_data)
    except Exception as e:
        # Log detailed error information
        error_context = {
            "session_id": session_id,
            "height_m": height_m,
            "mass_kg": mass_kg,
            "sex": sex,
            "error_type": type(e).__name__,
        }

        log_error_with_context(e, error_context)

        # Return detailed error response
        error_detail = {
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now().isoformat(),
        }

        logger.error(f"Unexpected error in download_last_video_and_run_mono: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")

        raise HTTPException(status_code=500, detail=error_detail)


@app.get("/download_file")
async def download_file(
    file_path: str = Query(..., description="Path to the file to download"),
    api_key: bool = Depends(verify_api_key),
):
    logger.info(f"Attempting to download file at path: {file_path}")

    # Resolve and sandbox: the requested path must resolve to within the project root.
    # This prevents arbitrary file reads (e.g. /etc/passwd, ~/.ssh/id_rsa).
    try:
        resolved = os.path.realpath(os.path.join(_API_ROOT, file_path) if not os.path.isabs(file_path) else file_path)
        api_root_real = os.path.realpath(_API_ROOT)
        if not resolved.startswith(api_root_real + os.sep) and resolved != api_root_real:
            logger.warning(f"Download path escape attempt blocked: {file_path!r} → {resolved!r}")
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "Access denied",
                    "message": "Requested file is outside the allowed directory",
                },
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Path resolution error for {file_path!r}: {exc}")
        raise HTTPException(status_code=400, detail="Invalid file path")

    if not os.path.exists(resolved):
        logger.error(f"File not found at path: {resolved}")
        raise HTTPException(
            status_code=404, detail=f"File not found at specified path: {file_path}"
        )
    if not os.access(resolved, os.R_OK):
        logger.error(f"Read permission denied for file: {resolved}")
        raise HTTPException(
            status_code=403, detail=f"Permission denied for file: {file_path}"
        )

    filename = os.path.basename(resolved)
    content_type = None
    if resolved.lower().endswith((".avi", ".mov")):
        content_type = "video/x-msvideo"
    elif resolved.lower().endswith(".json"):
        content_type = "application/json"

    return FileResponse(path=resolved, filename=filename, media_type=content_type)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
