"""Profile the lipsync pipeline on a video in S3 -- no SQS/Postgres needed.

Downloads the video, then runs the same split + per-chunk inference the
worker does and prints a per-stage timing breakdown:

    python bench.py s3://5dot-storage/clips/video.mp4
    python bench.py https://5dot-storage.s3.us-east-1.amazonaws.com/clips/video.mp4
    python bench.py clips/video.mp4            # bare key -> AWS_BUCKET_NAME
    python bench.py '<presigned url>'          # downloaded over plain HTTPS
    python bench.py s3://bucket/key.mp4 --max-chunks 6

AWS credentials come from the usual boto3 chain (AWS_* env vars / .env,
instance role). Model weights default to /model-cache/{SERVICE_NAME}/.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time
import urllib.request
from urllib.parse import parse_qs, unquote, urlparse

import boto3
import torch
from dotenv import load_dotenv

from infer import infer_chunk, load_models, split_video_into_chunks
from timing import format_timings, merge_timings, timed

load_dotenv()


def parse_s3_location(location: str) -> tuple[str, str]:
    """Return (bucket, key) for s3://, virtual-hosted / path-style https, or a bare key."""
    parsed = urlparse(location)
    if parsed.scheme == "s3":
        return parsed.netloc, parsed.path.lstrip("/")
    if parsed.scheme in ("http", "https"):
        host, path = parsed.netloc, unquote(parsed.path.lstrip("/"))
        if host.startswith("s3.") or host.startswith("s3-"):
            bucket, _, key = path.partition("/")
            return bucket, key
        if ".s3" in host:
            return host.split(".s3", 1)[0], path
        raise ValueError(f"Not an S3 URL: {location}")
    return os.getenv("AWS_BUCKET_NAME", "5dot-storage"), location.lstrip("/")


def download_video(location: str, dest_dir: str) -> str:
    parsed = urlparse(location)
    ext = os.path.splitext(parsed.path)[1] or ".mp4"
    dest = os.path.join(dest_dir, f"source{ext}")

    if parsed.scheme in ("http", "https") and "X-Amz-Signature" in parse_qs(parsed.query):
        print(f"[INFO] Downloading presigned URL -> {dest}")
        urllib.request.urlretrieve(location, dest)
        return dest

    bucket, key = parse_s3_location(location)
    print(f"[INFO] Downloading s3://{bucket}/{key} -> {dest}")
    boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1")).download_file(bucket, key, dest)
    return dest


def main():
    model_dir = f"/model-cache/{os.getenv('SERVICE_NAME', 'lipsync')}"

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", help="s3://bucket/key, S3 https URL, presigned URL, or bare key")
    parser.add_argument("--chunk-length", type=int, default=int(os.getenv("CHUNK_LENGTH_SECONDS", "5")))
    parser.add_argument("--max-chunks", type=int, default=None, help="only process the first N chunks")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--syncnet", default=os.path.join(model_dir, "syncnet_v2.model"))
    parser.add_argument("--s3fd", default=os.path.join(model_dir, "sfd_face.pth"))
    args = parser.parse_args()

    load_models(device=args.device, syncnet_checkpoint=args.syncnet, s3fd_checkpoint=args.s3fd)

    timings = {}
    work_dir = tempfile.mkdtemp(prefix="lipsync_bench_")
    t0 = time.perf_counter()
    try:
        with timed(timings, "download_source"):
            source_path = download_video(args.video, work_dir)
        with timed(timings, "split_chunks"):
            chunks = split_video_into_chunks(source_path, os.path.join(work_dir, "chunks"), args.chunk_length)
        if args.max_chunks:
            chunks = chunks[:args.max_chunks]

        results = []
        for path, start, end in chunks:
            c0 = time.perf_counter()
            r = infer_chunk(path, start, end, data_dir=work_dir)
            merge_timings(timings, r["timings"])
            results.append((r, time.perf_counter() - c0))
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    wall = time.perf_counter() - t0

    print("\n=== per chunk ===")
    for r, secs in results:
        res = r["result"]
        extra = f"  ERROR: {res['error']}" if "error" in res else (
            f"  frames={res['num_frames']} tracks={res['num_tracks']}")
        print(f"  [{r['start']:6.1f}-{r['end']:6.1f}s] {secs:7.2f}s  score={res['score']:6.1f} {res['label']:<11}{extra}")

    video_seconds = chunks[-1][2] - chunks[0][1] if chunks else 0.0
    per_sec = wall / video_seconds if video_seconds else 0.0
    print(f"\n=== timing breakdown: {len(chunks)} chunks, {video_seconds:.1f}s of video, "
          f"{per_sec:.2f}s per video-second ===")
    print(format_timings(timings, wall))


if __name__ == "__main__":
    main()
