"""lip_sync worker.

Standalone script (no FastAPI/Celery): long-polls its own SQS queue in an
infinite loop, does raw-SQL Postgres reads/writes, and reports completion to
the core service via webhook. See config/project_config.py for env vars.
"""

from __future__ import annotations

import json
import logging
import os
import time

import boto3

import db
import shared_storage
import webhook
from config.project_config import (
    AWS_REGION,
    CHUNK_LENGTH_SECONDS,
    DEVICE,
    IDLE_TIMEOUT_SECONDS,
    S3FD_CHECKPOINT,
    SERVICE_NAME,
    SQS_QUEUE_URL,
    SYNCNET_CHECKPOINT,
)
from infer import chunk_boundaries, infer_chunk, label_for_score, load_models
from timing import format_timings, merge_timings, timed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(SERVICE_NAME)

INFERENCE_MAX_ATTEMPTS = 3  # per chunk: 1 initial attempt + 2 retries, for transient errors like CUDA OOM


def parse_sqs_message(message: dict) -> dict:
    body = message.get("Body")
    if isinstance(body, str):
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        return {"job_id": body.strip()}
    return {"job_id": str(body)}


def extract_job_id(message: dict) -> str:
    parsed = parse_sqs_message(message)
    return str(parsed.get("job_id", "")).strip()


def _process_chunks(job_id: str, source_path: str, timings: dict) -> list[dict]:
    with timed(timings, "plan_chunks"):
        chunks = chunk_boundaries(source_path, CHUNK_LENGTH_SECONDS)
    if not chunks:
        raise RuntimeError("No video chunks could be extracted.")

    results = []
    for i, (start, end) in enumerate(chunks):
        t0 = time.perf_counter()
        r = infer_chunk(source_path, start, end, max_attempts=INFERENCE_MAX_ATTEMPTS)
        merge_timings(timings, r["timings"])
        logger.info("Job %s chunk %d/%d [%.0f-%.0fs] took %.2fs: %s",
                    job_id, i + 1, len(chunks), start, end, time.perf_counter() - t0, r["result"])
        results.append(r)
    return results


def process_job(conn, job_id: str, message_meta: dict | None = None) -> None:
    job = db.fetch_job(conn, job_id)
    if not job:
        logger.error("Job %s not found in detection_requests", job_id)
        return

    if message_meta:
        if message_meta.get("s3_key"):
            if job.get("file_key") and job["file_key"] != message_meta["s3_key"]:
                logger.warning(
                    "Job %s: DB file_key %s differs from SQS s3_key %s; using SQS s3_key",
                    job_id,
                    job.get("file_key"),
                    message_meta["s3_key"],
                )
            job["file_key"] = message_meta["s3_key"]
        if message_meta.get("url_source") and not job.get("url_source"):
            job["url_source"] = message_meta["url_source"]

    if not job.get("detect_lipsync"):
        logger.info("Job %s did not request lipsync detection, skipping", job_id)
        return

    db.mark_processing(conn, job_id)

    timings = {}
    job_t0 = time.perf_counter()
    video_seconds = 0.0
    try:
        with timed(timings, "download_source"):
            source_path = shared_storage.get_source_file(job)
        chunk_results = _process_chunks(job_id, source_path, timings)
        video_seconds = chunk_results[-1]["end"]

        with timed(timings, "db_writes"):
            for i, r in enumerate(chunk_results):
                db.update_chunk(conn, job_id, i, r["result"]["score"], r["start"], r["end"])

            overall_score = sum(r["result"]["score"] for r in chunk_results) / len(chunk_results)
            db.save_result(conn, job_id, overall_score)

        with timed(timings, "webhook"):
            webhook.notify(job_id, "complete", {"score": overall_score, "label": label_for_score(overall_score)})

    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        db.mark_failed(conn, job_id, str(exc))
        webhook.notify(job_id, "failed", {"error": str(exc)})

    finally:
        shared_storage.cleanup_if_last(conn, job_id)
        wall = time.perf_counter() - job_t0
        speed = f", {wall / video_seconds:.2f}s per video-second" if video_seconds else ""
        logger.info("Job %s timing breakdown (%.1fs of video%s):\n%s",
                    job_id, video_seconds, speed, format_timings(timings, wall))


def main():
    logger.info("Loading %s models from /model-cache/%s ...", SERVICE_NAME, SERVICE_NAME)
    load_models(device=DEVICE, syncnet_checkpoint=SYNCNET_CHECKPOINT, s3fd_checkpoint=S3FD_CHECKPOINT)
    logger.info("Models loaded, connecting to Postgres...")
    conn = db.connect()

    sqs = boto3.client("sqs", region_name=AWS_REGION)
    last_activity_at = time.time()

    logger.info("Polling %s", SQS_QUEUE_URL)
    while True:
        resp = sqs.receive_message(
            QueueUrl=SQS_QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20,
        )
        messages = resp.get("Messages", [])

        if not messages:
            if time.time() - last_activity_at >= IDLE_TIMEOUT_SECONDS:
                logger.info("Idle: no messages in the last %ss", IDLE_TIMEOUT_SECONDS)
                last_activity_at = time.time()
            time.sleep(1)
            continue

        last_activity_at = time.time()
        message = messages[0]
        message_meta = parse_sqs_message(message)
        job_id = str(message_meta.get("job_id", "")).strip()

        logger.info("Received job %s", job_id)
        try:
            process_job(conn, job_id, message_meta=message_meta)
        except Exception:
            logger.exception("Unhandled error processing job %s", job_id)
        finally:
            sqs.delete_message(QueueUrl=SQS_QUEUE_URL, ReceiptHandle=message["ReceiptHandle"])
            logger.info("Deleted SQS message for job %s", job_id)


if __name__ == "__main__":
    main()
