"""Raw-SQL Postgres access for the lip_sync worker.

No ORM: a single psycopg2 connection is opened at startup by the caller
(worker.py) and reused across the whole poll loop. Autocommit is on so a
crash mid-job never leaves a stray transaction open on the shared connection.
"""

from __future__ import annotations

import logging

import psycopg2
import psycopg2.extras

from config.project_config import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = ("complete", "failed")


def connect():
    conn = psycopg2.connect(
        host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT
    )
    conn.autocommit = True
    return conn


def fetch_job(conn, job_id: str) -> dict | None:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id, filename, file_key, url_source, duration, detect_lipsync
            FROM detection_requests
            WHERE id = %s
            """,
            (job_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def mark_processing(conn, job_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE detection_requests SET lipsync_status = 'processing' WHERE id = %s",
            (job_id,),
        )


def save_result(conn, job_id: str, score: float) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE detection_requests
            SET lipsync_status = 'complete', overall_lipsync_score = %s
            WHERE id = %s
            """,
            (score, job_id),
        )


def mark_failed(conn, job_id: str, error_message: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE detection_requests
            SET lipsync_status = 'failed', error_message = %s
            WHERE id = %s
            """,
            (error_message[:2000], job_id),
        )


def update_chunk(conn, job_id: str, chunk_index: int, score: float, start: float, end: float) -> None:
    """Chunk rows are pre-created (by whatever created the job) keyed on
    (detection_request_id, chunk_index); this only ever UPDATEs them."""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE detection_chunks
            SET lipsync_score = %s, lipsync_start = %s, lipsync_end = %s
            WHERE detection_request_id = %s AND chunk_index = %s
            """,
            (score, start, end, job_id, chunk_index),
        )
        if cur.rowcount == 0:
            logger.warning(
                "No detection_chunks row for job %s chunk_index %s; expected it to pre-exist",
                job_id, chunk_index,
            )


def other_services_done(conn, job_id: str) -> bool:
    """True if every other requested detection (audio/video/changes) on this
    job has already reached a terminal status. Used to decide whether this
    service is the last one still touching the shared downloaded source file."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT detect_ai_audio, ai_audio_status,
                   detect_ai_video, ai_video_status,
                   detect_changes, changes_status
            FROM detection_requests
            WHERE id = %s
            """,
            (job_id,),
        )
        row = cur.fetchone()
        if not row:
            return True

        def done(requested, status):
            return (not requested) or (status in TERMINAL_STATUSES)

        return (
            done(row["detect_ai_audio"], row["ai_audio_status"])
            and done(row["detect_ai_video"], row["ai_video_status"])
            and done(row["detect_changes"], row["changes_status"])
        )
