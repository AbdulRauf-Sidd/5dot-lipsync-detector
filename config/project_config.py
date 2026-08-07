import os

from dotenv import load_dotenv

load_dotenv()

# --- Service identity ---
# Does NOT need to match the DB column prefix (lipsync_*) -- this is only used
# for /model-cache/{SERVICE_NAME}/ and the webhook's service_name field.
SERVICE_NAME = os.environ["SERVICE_NAME"]

# --- Paths ---
SHARED_JOBS_DIR = os.getenv("SHARED_JOBS_DIR", "/tmp/shared_jobs")
MODEL_CACHE_DIR = f"/model-cache/{SERVICE_NAME}"
SYNCNET_CHECKPOINT = os.path.join(MODEL_CACHE_DIR, "syncnet_v2.model")
S3FD_CHECKPOINT = os.path.join(MODEL_CACHE_DIR, "sfd_face.pth")

# --- ML ---
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

# Chunk duration is a parameter (not hardcoded) because the pre-created
# detection_chunks rows assume a shared grid across all 3 detection services --
# if that grid convention ever changes, only this needs to move.
CHUNK_LENGTH_SECONDS = int(os.getenv("CHUNK_LENGTH_SECONDS", "5"))

# --- SQS ---
SQS_QUEUE_URL = os.environ["SQS_QUEUE_URL"]
IDLE_TIMEOUT_SECONDS = int(os.getenv("IDLE_TIMEOUT_SECONDS", "60"))

# --- Postgres ---
DB_HOST = os.environ["DB_HOST"]
DB_NAME = os.environ["DB_NAME"]
DB_USER = os.environ["DB_USER"]
DB_PASSWORD = os.environ["DB_PASSWORD"]
DB_PORT = int(os.getenv("DB_PORT", "5432"))

# --- Webhook ---
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "https://core-service.placeholder/webhook")
WEBHOOK_MAX_RETRIES = 3
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")

# --- AWS S3 (source download) ---
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME", "5dot-storage")
