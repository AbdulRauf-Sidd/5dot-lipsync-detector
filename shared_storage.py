"""Shared-download coordination for /tmp/shared_jobs.

All 3 detection services (video_ai_service, lip_sync, scene_detection) run as
separate containers on the same EC2 instance, bind-mounting the same host
path at SHARED_JOBS_DIR. Whichever service processes a job_id first
downloads the source file into a per-job folder there; the other two wait
for a `.done` marker instead of downloading it again.
"""

import logging
import os
import shutil
import time

from config.project_config import AWS_BUCKET_NAME, SHARED_JOBS_DIR
from utils.s3 import download_file

logger = logging.getLogger(__name__)

DOWNLOAD_WAIT_TIMEOUT_SECONDS = int(os.getenv("DOWNLOAD_WAIT_TIMEOUT_SECONDS", "600"))
DOWNLOAD_WAIT_POLL_SECONDS = 2


class SharedDownloadError(Exception):
    pass


def _job_dir(job_id: str) -> str:
    path = os.path.join(SHARED_JOBS_DIR, str(job_id))
    os.makedirs(path, exist_ok=True)
    return path


def _lock_path(job_id: str) -> str:
    return os.path.join(_job_dir(job_id), "download.lock")


def _done_path(job_id: str) -> str:
    return os.path.join(_job_dir(job_id), "download.done")


def _source_ext(job: dict) -> str:
    if job.get("file_key"):
        return os.path.splitext(job["file_key"])[1] or ".mp4"
    return ".mp4"


def _download_url(url: str, dest_path: str) -> None:
    import yt_dlp

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": dest_path,
        "quiet": True,
        "no_warnings": True,
        "merge_output_format": "mp4",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def _download_source(job: dict, dest_path: str) -> None:
    if job.get("file_key"):
        logger.info("Downloading s3://%s/%s -> %s", AWS_BUCKET_NAME, job["file_key"], dest_path)
        download_file(job["file_key"], dest_path)
    elif job.get("url_source"):
        logger.info("Downloading %s -> %s", job["url_source"], dest_path)
        _download_url(job["url_source"], dest_path)
    else:
        raise SharedDownloadError(f"job {job['id']} has neither file_key nor url_source")


def get_source_file(job: dict) -> str:
    """Return the local path to the job's source file, downloading it if
    no other service already has (or is currently doing so)."""
    job_id = str(job["id"])
    job_dir = _job_dir(job_id)
    lock_path = _lock_path(job_id)
    done_path = _done_path(job_id)
    source_path = os.path.join(job_dir, f"source{_source_ext(job)}")

    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return _wait_for_download(job_id, done_path, source_path)

    # We won the race: we own the download.
    try:
        os.write(fd, str(os.getpid()).encode())
    finally:
        os.close(fd)

    _download_source(job, source_path)
    with open(done_path, "w") as f:
        f.write(os.path.basename(source_path))
    return source_path


def _wait_for_download(job_id: str, done_path: str, source_path: str) -> str:
    deadline = time.time() + DOWNLOAD_WAIT_TIMEOUT_SECONDS
    while time.time() < deadline:
        if os.path.exists(done_path):
            return source_path
        time.sleep(DOWNLOAD_WAIT_POLL_SECONDS)
    raise SharedDownloadError(
        f"job {job_id}: timed out after {DOWNLOAD_WAIT_TIMEOUT_SECONDS}s waiting for "
        f"another service to finish downloading the source file"
    )


def cleanup_if_last(conn, job_id: str) -> None:
    """Delete the shared job folder once every requested service has reached
    a terminal state, so /tmp/shared_jobs doesn't grow unbounded."""
    from db import other_services_done

    if not other_services_done(conn, job_id):
        return
    job_dir = _job_dir(job_id)
    try:
        shutil.rmtree(job_dir)
        logger.info("Cleaned up shared job dir for %s (last service to finish)", job_id)
    except FileNotFoundError:
        pass
