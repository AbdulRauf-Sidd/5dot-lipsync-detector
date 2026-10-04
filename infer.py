from __future__ import annotations
import os, json, pickle, subprocess, cv2, numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from scipy.interpolate import interp1d
from scenedetect.detectors import ContentDetector
from detectors import S3FD
from detectors.s3fd.box_utils import nms_
from SyncNetInstance import SyncNetInstance
from scipy import signal
from timing import timed


SYNCNET_MODEL = None
FACE_DETECTOR = None

# SyncNet's own per-track confidence (median dist - min dist: how much
# sharper the best av-offset match is than a typical one) turned into a
# 0-100 lipsync authenticity score (higher = more likely in sync/authentic).
# NOTE: this is the inverse of the "higher = more suspicious" polarity used
# by the sibling ai_audio/ai_video/changes detectors that feed
# overall_ai_video_score -- whatever combines them needs to account for that.
#
# Confidence -> score is piecewise linear across bands calibrated from
# testing, each with a gap from its neighbor so a chunk never lands in the
# ambiguous zone between two confidence tiers (e.g. never scores 35, which
# would be unclear as between the 0-4 and 4-6 confidence bands). Bounds are
# (conf_lo, conf_hi, score_lo, score_hi).
CONF_SCORE_BANDS = [
    (0.0, 4.0, 0.0, 30.0),
    (4.0, 6.0, 40.0, 60.0),
    (6.0, 9.0, 70.0, 90.0),
    (9.0, 12.0, 95.0, 100.0),
]

# Score used when there's no SyncNet evidence at all (no face track found,
# or evaluation errored) -- defaults to "authentic" rather than "desynced"
# so missing data doesn't get misread as a confirmed manipulation signal.
NO_EVIDENCE_SCORE = 100.0

# The demo default (100) assumes a whole video is being tracked; chunks are
# only CHUNK_LENGTH_SECONDS long (~125 frames at 25fps for the default 5s),
# so a lower floor is needed or almost no track ever qualifies.
CHUNK_MIN_TRACK_FRAMES = 15

AUDIO_SAMPLE_RATE = 16000  # what SyncNet's MFCC front end expects


def load_models(device="cuda", syncnet_checkpoint="data/syncnet_v2.model", s3fd_checkpoint=None):
    """
    Load SyncNet and face detector once at worker startup. In production,
    syncnet_checkpoint/s3fd_checkpoint point at /model-cache/{SERVICE_NAME}/
    (baked into the AMI, loaded offline); the defaults here only support
    local dev via download_model.sh.
    """
    global SYNCNET_MODEL, FACE_DETECTOR
    if SYNCNET_MODEL is None:
        SYNCNET_MODEL = SyncNetInstance()
        SYNCNET_MODEL.loadParameters(syncnet_checkpoint)
    if FACE_DETECTOR is None:
        FACE_DETECTOR = S3FD(device=device, weight_path=s3fd_checkpoint)
    print("[INFO] Models loaded")

class Config:
    def __init__(self, video_path, reference, data_dir="data/work", min_track=100):
        self.videofile = video_path
        self.reference = reference
        self.data_dir = data_dir

        self.frame_rate = 25
        self.facedet_scale = 0.25
        self.crop_scale = 0.4
        self.min_track = min_track
        self.num_failed_det = 25
        self.min_face_size = 100
        self.facedet_conf_th = 0.9
        # Frames per S3FD forward pass. Halved automatically on CUDA OOM.
        self.facedet_batch_size = 16
        self.batch_size = 20
        self.vshift = 15

        self.avi_dir = os.path.join(data_dir, "pyavi")
        self.tmp_dir = os.path.join(data_dir, "pytmp")
        self.work_dir = os.path.join(data_dir, "pywork")
        self.crop_dir = os.path.join(data_dir, "pycrop")
        self.frames_dir = os.path.join(data_dir, "pyframes")

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])

    return interArea/(boxAArea + boxBArea - interArea + 1e-6)

def _run_ffmpeg(cmd: list) -> bytes:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (exit {proc.returncode}): {proc.stderr.decode(errors='replace')[-500:]}")
    return proc.stdout

def _seek_args(start, end):
    # Input-side -ss/-to while decoding is frame accurate (unlike a -c copy cut,
    # which snaps to keyframes and makes neighbouring chunks overlap).
    args = []
    if start is not None:
        args += ["-ss", str(start)]
    if end is not None:
        args += ["-to", str(end)]
    return args

def _probe_frame_size(input_path: str) -> tuple[int, int]:
    """Displayed (width, height): ffmpeg auto-rotates on decode, so a 90/270
    rotation tag (old ffmpeg: tags.rotate, new: side_data_list) swaps them."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json", input_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    width, height = int(stream["width"]), int(stream["height"])
    rotation = stream.get("tags", {}).get("rotate")
    for side_data in stream.get("side_data_list", []):
        if "rotation" in side_data:
            rotation = side_data["rotation"]
    if rotation is not None and abs(int(float(rotation))) % 180 == 90:
        width, height = height, width
    return width, height

def decode_video(input_path: str, start=None, end=None, frame_rate: int = 25) -> np.ndarray:
    """Decode [start, end) straight to an (N, H, W, 3) BGR uint8 array at frame_rate.
    Held in memory for the whole chunk: ~6MB/frame at 1080p, so ~0.8GB per 5s chunk."""
    width, height = _probe_frame_size(input_path)
    raw = _run_ffmpeg(
        ["ffmpeg", "-v", "error"] + _seek_args(start, end) + ["-i", input_path,
         "-an", "-r", str(frame_rate), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    )
    frame_bytes = width * height * 3
    if not raw or len(raw) % frame_bytes:
        raise RuntimeError(f"Decoded {len(raw)} bytes, not a whole number of {width}x{height} frames: {input_path}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3)

def decode_audio(input_path: str, start=None, end=None, sample_rate: int = AUDIO_SAMPLE_RATE) -> np.ndarray:
    """Decode [start, end) to mono int16 PCM at sample_rate."""
    raw = _run_ffmpeg(
        ["ffmpeg", "-v", "error"] + _seek_args(start, end) + ["-i", input_path,
         "-vn", "-async", "1", "-ac", "1", "-ar", str(sample_rate), "-acodec", "pcm_s16le", "-f", "s16le", "-"]
    )
    return np.frombuffer(raw, dtype=np.int16)

def scene_detect(frames: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) frame ranges per shot, via PySceneDetect's ContentDetector
    run directly on the decoded frames (same auto-downscale to ~256px wide
    that VideoManager.set_downscale_factor() applied)."""
    detector = ContentDetector()
    width = frames.shape[2]
    downscale = 1 if width < 256 else width // 256

    cuts = []
    for frame_num, frame in enumerate(frames):
        if downscale > 1:
            frame = np.ascontiguousarray(frame[::downscale, ::downscale, :])
        cuts.extend(detector.process_frame(frame_num, frame) or [])
    cuts.extend(detector.post_process(len(frames) - 1) or [])

    bounds = [0] + sorted(c for c in set(cuts) if 0 < c < len(frames)) + [len(frames)]
    return list(zip(bounds[:-1], bounds[1:]))

def preprocess_frame(frame, scale):
    """BGR frame -> S3FD input. Runs in worker threads (cv2 releases the GIL),
    so it overlaps with the GPU forward of the prior batch."""
    image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return S3FD.preprocess(image_np, scale), image_np.shape[1], image_np.shape[0]

def detect_batch_safe(DET, batch, w, h, conf_th):
    try:
        return DET.detect_batch(batch, w, h, conf_th)
    except torch.cuda.OutOfMemoryError:
        if len(batch) == 1:
            raise
        torch.cuda.empty_cache()
        mid = len(batch) // 2
        print(f"[WARN] CUDA OOM on face detection batch of {len(batch)}, splitting")
        return (detect_batch_safe(DET, batch[:mid], w, h, conf_th)
                + detect_batch_safe(DET, batch[mid:], w, h, conf_th))

def inference_video(cfg, frames, timings=None):
    """
    Use preloaded FACE_DETECTOR instead of creating a new one every call.
    """
    DET = FACE_DETECTOR
    if not len(frames):
        raise ValueError(f"No frames decoded from {cfg.videofile}")
    dets = [[] for _ in range(len(frames))]

    bs = cfg.facedet_batch_size
    batches = [range(i, min(i + bs, len(frames))) for i in range(0, len(frames), bs)]

    with ThreadPoolExecutor(max_workers=8) as executor:
        def submit(batch):
            return [executor.submit(preprocess_frame, frames[idx], cfg.facedet_scale) for idx in batch]

        pending = submit(batches[0])
        for bi, batch in enumerate(batches):
            with timed(timings, "face_detection/load_wait"):
                loaded = [f.result() for f in pending]
            pending = submit(batches[bi + 1]) if bi + 1 < len(batches) else None

            # Group by frame size so each forward pass gets a uniform stack
            # (frames of one video always share a size; this is just a guard).
            groups = {}
            for idx, item in zip(batch, loaded):
                groups.setdefault((item[1], item[2]), []).append((idx, item[0]))

            for (w, h), items in groups.items():
                with timed(timings, "face_detection/forward"):
                    results = detect_batch_safe(DET, np.stack([img for _, img in items]), w, h, cfg.facedet_conf_th)
                for (idx, _), bboxes in zip(items, results):
                    for bbox in bboxes[nms_(bboxes, 0.1)]:
                        dets[idx].append({"frame": idx, "bbox": bbox[:-1].tolist(), "conf": bbox[-1]})

    return dets

def track_shot(cfg, scenefaces):
    iouThres = 0.5
    tracks = []

    faces_array = [list(f) for f in scenefaces]  # copy
    while True:
        track = []
        for framefaces in faces_array:
            for face in framefaces:
                if not track:
                    track.append(face)
                    framefaces.remove(face)
                elif face["frame"] - track[-1]["frame"] <= cfg.num_failed_det:
                    iou = bb_intersection_over_union(face["bbox"], track[-1]["bbox"])
                    if iou > iouThres:
                        track.append(face)
                        framefaces.remove(face)
                        continue
                else:
                    break
        if not track:
            break
        if len(track) > cfg.min_track:
            framenum = np.array([f["frame"] for f in track])
            bboxes = np.array([f["bbox"] for f in track])
            frame_i = np.arange(framenum[0], framenum[-1]+1)
            bboxes_i = np.stack([interp1d(framenum, bboxes[:, ij])(frame_i) for ij in range(4)], axis=1)
            if max(np.mean(bboxes_i[:, 2]-bboxes_i[:, 0]), np.mean(bboxes_i[:, 3]-bboxes_i[:, 1])) > cfg.min_face_size:
                tracks.append({"frame": frame_i, "bbox": bboxes_i})

    return tracks

        
def crop_track(opt, frames, audio, track):
    """Return the track's 224x224 BGR face crops and its matching audio slice,
    in memory -- what used to round-trip through an XVID .avi for SyncNet."""

    dets = {'x':[], 'y':[], 's':[]}

    for det in track['bbox']:

        dets['s'].append(max((det[3]-det[1]),(det[2]-det[0]))/2) 
        dets['y'].append((det[1]+det[3])/2) 
        dets['x'].append((det[0]+det[2])/2) 

    dets['s'] = signal.medfilt(dets['s'],kernel_size=13)   
    dets['x'] = signal.medfilt(dets['x'],kernel_size=13)
    dets['y'] = signal.medfilt(dets['y'],kernel_size=13)

    crops = np.empty((len(track['frame']), 224, 224, 3), dtype=np.uint8)

    for fidx, frame in enumerate(track['frame']):

        cs  = opt.crop_scale

        bs  = dets['s'][fidx]  
        bsi = int(bs*(1+2*cs)) 

        # Same result as np.pad(..., constant_values=110), several times faster.
        padded = cv2.copyMakeBorder(frames[frame], bsi, bsi, bsi, bsi, cv2.BORDER_CONSTANT, value=(110, 110, 110))
        my  = dets['y'][fidx]+bsi  
        mx  = dets['x'][fidx]+bsi 

        face = padded[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
        
        crops[fidx] = cv2.resize(face,(224,224))

    audiostart  = int(round(track['frame'][0] / opt.frame_rate * AUDIO_SAMPLE_RATE))
    audioend    = int(round((track['frame'][-1]+1) / opt.frame_rate * AUDIO_SAMPLE_RATE))

    print('Mean pos: x %.2f y %.2f s %.2f'%(np.mean(dets['x']),np.mean(dets['y']),np.mean(dets['s'])))

    return crops, audio[audiostart:audioend]


def crop_faces(cfg, frames, audio, tracks):
    print(f"[INFO] Cropping {len(tracks)} track(s)...")
    return [crop_track(cfg, frames, audio, track) for track in tracks]


def run_syncnet(cfg, crops, timings=None):
    s = SYNCNET_MODEL

    if len(crops) == 0:
        print(f"[WARN] No face tracks for syncnet evaluation")
        return []

    confs = []
    for idx, (images, audio) in enumerate(crops):
        try:
            result = s.evaluate_arrays(cfg, images, audio, sample_rate=AUDIO_SAMPLE_RATE, timings=timings)
            if result is not None and len(result) > 2:
                _, conf, _ = result
                confs.append(float(np.array(conf)))
        except Exception as e:
            print(f"[WARN] Error evaluating syncnet for track {idx}: {e}")

    print(f"[INFO] Computed {len(confs)} sync confidence scores")
    return confs

def run_inference(video_path: str, reference: str, skip_persistent_save: bool = False,
                   data_dir: str = "data/work", min_track: int = 100, timings: dict | None = None,
                   start: float | None = None, end: float | None = None):
    """Analyse [start, end) of video_path (the whole file if both are None)."""

    cfg = Config(video_path, reference, data_dir=data_dir, min_track=min_track)

    print(f"[INFO] Decoding video...")
    with timed(timings, "decode_video"):
        frames = decode_video(video_path, start, end, cfg.frame_rate)
    print(f"[INFO] Decoding audio...")
    with timed(timings, "decode_audio"):
        audio = decode_audio(video_path, start, end)

    # FACE DETECTION
    print(f"[INFO] Running face detection...")
    with timed(timings, "face_detection"):
        faces = inference_video(cfg, frames, timings=timings)
    print(f"[INFO] Detected faces in {len([f for f in faces if f])} frames")

    # SCENE DETECTION
    print(f"[INFO] Running scene detection...")
    with timed(timings, "scene_detection"):
        scenes = scene_detect(frames)
    print(f"[INFO] Found {len(scenes)} scenes")

    # FACE TRACKING
    print(f"[INFO] Running face tracking...")
    tracks = []
    with timed(timings, "face_tracking"):
        for shot_idx, (shot_start, shot_end) in enumerate(scenes):
            if shot_end - shot_start >= cfg.min_track:
                shot_tracks = track_shot(cfg, faces[shot_start:shot_end])
                tracks.extend(shot_tracks)
                print(f"[INFO] Shot {shot_idx}: Found {len(shot_tracks)} tracks")
    
    print(f"[INFO] Total tracks: {len(tracks)}")

    with timed(timings, "crop_faces"):
        crops = crop_faces(cfg, frames, audio, tracks)
    
    print(f"[INFO] Running syncnet evaluation...")
    with timed(timings, "syncnet"):
        confs = run_syncnet(cfg, crops, timings=timings)

    result = {"tracks": tracks, "confs": confs, "num_frames": len(frames)}

    if not skip_persistent_save:
        out_dir = os.path.join(cfg.work_dir, reference)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "results.pkl"), "wb") as f:
            pickle.dump(result, f)

    return result


def _confidence_to_score(conf: float) -> float:
    """Map a single track's SyncNet confidence to a 0-100 score via CONF_SCORE_BANDS."""
    if conf <= 0.0:
        return 0.0
    for conf_lo, conf_hi, score_lo, score_hi in CONF_SCORE_BANDS:
        if conf_lo <= conf < conf_hi:
            t = (conf - conf_lo) / (conf_hi - conf_lo)
            return score_lo + t * (score_hi - score_lo)
    return CONF_SCORE_BANDS[-1][3]


def compute_chunk_score(confs: list) -> float:
    """Average per-track scores; no evidence defaults to NO_EVIDENCE_SCORE."""
    if not confs:
        return NO_EVIDENCE_SCORE
    return float(np.mean([_confidence_to_score(c) for c in confs]))


def label_for_score(score: float) -> str:
    if score >= 70:
        return "authentic"
    if score >= 31:
        return "uncertain"
    return "manipulated"


def _probe_duration(input_path: str) -> float:
    """Container-level format=duration can be stale/inflated -- e.g. after a
    stream-copy trim, or when a corrupted secondary stream drags the container
    tag out -- reporting a duration far longer than the video actually plays.
    Prefer the video stream's own duration and only fall back to the
    container-level tag if the stream field is unavailable."""
    stream_result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", input_path],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    stream_duration = stream_result.stdout.strip()
    if stream_duration and stream_duration != "N/A":
        return float(stream_duration)

    format_result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", input_path],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    return float(format_result.stdout.strip())


def chunk_boundaries(input_path: str, chunk_length: int = 5) -> list[tuple[float, float]]:
    """Returns (start, end) tuples -- callers must use these real,
    rounded-to-nearest-second boundaries rather than re-deriving them from
    chunk_length, since the last chunk is almost always shorter than the rest."""
    duration = _probe_duration(input_path)
    return [
        (float(start), min(start + chunk_length, duration))
        for start in range(0, max(int(duration), 1), chunk_length)
    ]


def infer_chunk(source_path: str, start: float, end: float, max_attempts: int = 1) -> dict:
    """
    Analyse [start, end) of source_path, decoded directly from the source (no
    chunk files on disk).

    Never raises: after max_attempts failures the chunk gets NO_EVIDENCE_SCORE
    plus an "error" field, consistent with how a chunk with no face track is
    scored. Retries are per chunk so a transient error (e.g. CUDA OOM) doesn't
    force the whole video to be reprocessed. The returned "timings" accumulate
    across attempts.
    """
    chunk_name = f"chunk_{start:g}-{end:g}"
    reference = f"{chunk_name}_{os.getpid()}"
    timings = {}
    base = {"chunk": chunk_name, "start": start, "end": end, "timings": timings}

    for attempt in range(1, max_attempts + 1):
        try:
            result = run_inference(
                source_path, reference, skip_persistent_save=True,
                min_track=CHUNK_MIN_TRACK_FRAMES, timings=timings, start=start, end=end,
            )
            break
        except Exception as e:
            print(f"[ERROR] Chunk inference failed (attempt {attempt}/{max_attempts}): {chunk_name}, error: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if attempt == max_attempts:
                return {**base, "result": {
                    "score": NO_EVIDENCE_SCORE,
                    "label": label_for_score(NO_EVIDENCE_SCORE),
                    "error": str(e),
                }}

    score = compute_chunk_score(result["confs"])
    return {**base, "result": {
        "score": score,
        "label": label_for_score(score),
        "num_tracks": len(result["tracks"]),
        "num_confs": len(result["confs"]),
        "num_frames": result["num_frames"],
    }}
