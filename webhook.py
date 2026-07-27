"""POST job completion back to the core service.

Best-effort: the result is already durably saved in Postgres by the time
this runs, so a webhook failure is logged and swallowed rather than raised
-- the caller deletes the SQS message either way.
"""

from __future__ import annotations

import logging
import time

import requests

from config.project_config import SERVICE_NAME, WEBHOOK_MAX_RETRIES, WEBHOOK_URL

logger = logging.getLogger(__name__)

BACKOFF_SECONDS = (1, 2, 4)


def notify(job_id: str, status: str, result: dict | None = None) -> None:
    payload = {
        "job_id": job_id,
        "service_name": SERVICE_NAME,
        "status": status,
        "result": result or {},
    }

    for attempt in range(1, WEBHOOK_MAX_RETRIES + 1):
        try:
            resp = requests.post(WEBHOOK_URL, json=payload, timeout=10)
            resp.raise_for_status()
            return
        except Exception as exc:
            logger.error(
                "Webhook attempt %s/%s failed for job %s: %s",
                attempt, WEBHOOK_MAX_RETRIES, job_id, exc,
            )
            if attempt < WEBHOOK_MAX_RETRIES:
                time.sleep(BACKOFF_SECONDS[min(attempt - 1, len(BACKOFF_SECONDS) - 1)])

    logger.error("Webhook permanently failed for job %s after %s attempts", job_id, WEBHOOK_MAX_RETRIES)
