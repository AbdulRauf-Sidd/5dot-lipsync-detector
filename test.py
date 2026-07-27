"""
Smoke test for lip_sync (standalone SQS worker, no FastAPI/Celery).

Run from inside lipsync-detector/:

    python test.py

Checks are split into two tiers:
  - HARD checks: pure code/wiring correctness (imports, config loading).
    No network/GPU/model-weights/DB required -- these must pass.
  - SOFT checks: anything that needs the actual model weights on disk
    (/model-cache/{SERVICE_NAME}/) or a live Postgres connection. Reported
    but don't fail the run, since missing weights/DB isn't a code bug.
"""

import os
import sys
import traceback

# Required env vars so config/project_config.py can be imported without a
# real deployment environment. Only used for the HARD (wiring) checks below.
os.environ.setdefault("SERVICE_NAME", "lipsync")
os.environ.setdefault("SQS_QUEUE_URL", "https://sqs.us-east-1.amazonaws.com/000000000000/lipsync-test")
os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_NAME", "postgres")
os.environ.setdefault("DB_USER", "postgres")
os.environ.setdefault("DB_PASSWORD", "postgres")

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

hard_failures = []
soft_warnings = []


def _run(label, fn, hard=True):
    try:
        fn()
        print(f"{PASS} {label}")
        return True
    except Exception as exc:
        bucket = hard_failures if hard else soft_warnings
        bucket.append(f"{label}: {exc}")
        print(f"{FAIL if hard else WARN} {label}: {exc}")
        if hard:
            traceback.print_exc(limit=3)
        return False


# ---------------------------------------------------------------------------
# HARD checks -- no network, no GPU, no model weights required
# ---------------------------------------------------------------------------

def check_config_imports():
    from config.project_config import (  # noqa: F401
        SQS_QUEUE_URL, DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, WEBHOOK_URL,
        SERVICE_NAME, IDLE_TIMEOUT_SECONDS, MODEL_CACHE_DIR, SYNCNET_CHECKPOINT,
        S3FD_CHECKPOINT, DEVICE, CHUNK_LENGTH_SECONDS,
    )


def check_db_module_imports():
    from db import connect, fetch_job, mark_processing, save_result, mark_failed, update_chunk, other_services_done  # noqa: F401


def check_shared_storage_imports():
    from shared_storage import get_source_file, cleanup_if_last, SharedDownloadError  # noqa: F401


def check_webhook_module_imports():
    from webhook import notify  # noqa: F401


def check_utils_import():
    from utils.s3 import upload_file, download_file, delete_file, presigned_url  # noqa: F401


def check_infer_imports():
    """
    Importing must succeed without touching disk/network -- it only defines
    the SyncNet/S3FD pipeline functions. Actual weight loading only happens
    inside load_models().
    """
    from infer import load_models, split_video_into_chunks, infer_chunk, compute_chunk_score  # noqa: F401


def check_worker_module_imports():
    """Importing worker.py must not start the poll loop (guarded by __main__)."""
    import worker
    assert hasattr(worker, "main")
    assert hasattr(worker, "process_job")
    assert hasattr(worker, "extract_job_id")


def check_compute_chunk_score():
    from infer import compute_chunk_score, GOOD_DIST, BAD_DIST
    assert compute_chunk_score([]) == 0.0
    assert compute_chunk_score([GOOD_DIST]) == 0.0
    assert compute_chunk_score([BAD_DIST]) == 1.0


# ---------------------------------------------------------------------------
# SOFT checks -- need real model weights on disk / a live database
# ---------------------------------------------------------------------------

def check_database_connection():
    import db
    conn = db.connect()
    with conn.cursor() as cur:
        cur.execute("SELECT 1")
    conn.close()


def check_models_load():
    """
    Loads SyncNet + S3FD for real. Needs:
      - /model-cache/{SERVICE_NAME}/syncnet_v2.model
      - /model-cache/{SERVICE_NAME}/sfd_face.pth
    """
    from config.project_config import DEVICE, S3FD_CHECKPOINT, SYNCNET_CHECKPOINT
    from infer import load_models
    import infer
    load_models(device=DEVICE, syncnet_checkpoint=SYNCNET_CHECKPOINT, s3fd_checkpoint=S3FD_CHECKPOINT)
    assert infer.SYNCNET_MODEL is not None
    assert infer.FACE_DETECTOR is not None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main_test():
    print("=== lip_sync smoke test ===\n")

    print("-- hard checks (imports & wiring, no weights/network/DB required) --")
    _run("config.project_config imports", check_config_imports)
    _run("db module imports", check_db_module_imports)
    _run("shared_storage module imports", check_shared_storage_imports)
    _run("webhook module imports", check_webhook_module_imports)
    _run("utils.s3 imports", check_utils_import)
    _run("infer module imports", check_infer_imports)
    _run("worker module imports (no loop started)", check_worker_module_imports)
    _run("compute_chunk_score mapping", check_compute_chunk_score)

    print("\n-- soft checks (model weights on disk / database required) --")
    _run("database connection (SELECT 1)", check_database_connection, hard=False)
    _run("load_models() -- SyncNet + S3FD", check_models_load, hard=False)

    print("\n=== summary ===")
    if hard_failures:
        print(f"{FAIL} {len(hard_failures)} hard check(s) failed:")
        for f in hard_failures:
            print(f"  - {f}")
    else:
        print(f"{PASS} all hard checks passed")

    if soft_warnings:
        print(f"{WARN} {len(soft_warnings)} soft check(s) skipped/failed (weights/network/DB not available):")
        for w in soft_warnings:
            print(f"  - {w}")

    return 1 if hard_failures else 0


if __name__ == "__main__":
    sys.exit(main_test())
