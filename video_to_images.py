"""Convert lossless FFV1 ``.avi`` videos in ``videos/`` into per-frame JPEGs.

For each video file ``videos/<name>.<ext>`` the script writes frames to
``datasets/guidewire/<name>/Images/<frame_index>.jpg`` (frame_index starts at 0).

Pipeline (per frame):
    FFV1 stream -> ffmpeg decode (lossless) -> uint8 BGR ndarray -> libjpeg encode -> .jpg on disk

Reproducibility contract
------------------------
The detection network will be trained on the JPEGs produced here, and at
inference time you will receive raw frames (e.g. from a camera) and need to
apply the *exact same* JPEG conversion before running the network.

To make that work, the JPEG encoding parameters are centralised in
``CANONICAL_JPEG_PARAMS`` / :func:`make_jpeg_params` and exposed through the
reusable helpers :func:`encode_frame_to_jpeg` and
:func:`encode_and_decode_frame`. Import those from your inference code:

    from video_to_images import encode_and_decode_frame
    processed = encode_and_decode_frame(raw_bgr_frame, quality=90)
    # `processed` is bit-identical to what cv2.imread(...) would return
    # for the corresponding training frame.

Notes on reproducibility:
- ``cv2.imencode(".jpg", frame, params)`` and ``cv2.imwrite(path, frame, params)``
  produce byte-identical JPEGs for the same input + params (verified on cv2 4.13).
- Frames are kept in OpenCV's native BGR order on both sides; do NOT convert
  to RGB before encoding or you'll silently swap channels relative to training.
- Chroma subsampling is pinned to 4:2:0 explicitly so the result does not
  depend on the libjpeg default of whichever OpenCV build is installed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional; fall back to a no-op wrapper.
    def tqdm(iterable: Iterable, **_kwargs):  # type: ignore[no-redef]
        return iterable


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VIDEOS_DIR = SCRIPT_DIR / "videos"
DEFAULT_DATASETS_DIR = SCRIPT_DIR / "datasets" / "guidewire"
VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv", ".webm"}

DEFAULT_JPEG_QUALITY = 90


def make_jpeg_params(quality: int = DEFAULT_JPEG_QUALITY) -> list[int]:
    """Build the canonical ``cv2.imwrite`` / ``cv2.imencode`` JPEG parameters.

    These parameters define the dataset's JPEG format. Use them everywhere
    (training data generation AND inference-time encoding) to guarantee that
    the network sees the same pixels in both settings.

    - ``IMWRITE_JPEG_QUALITY``: master quality 0-100.
    - ``IMWRITE_JPEG_OPTIMIZE=1``: optimised Huffman tables (smaller files at
      the same visual quality; output remains a standard JPEG).
    - ``IMWRITE_JPEG_PROGRESSIVE=0``: baseline JPEG, decodable by any reader.
    - ``IMWRITE_JPEG_SAMPLING_FACTOR=4:2:0``: pin chroma subsampling so the
      result doesn't depend on libjpeg's default in the installed OpenCV build.
    - We deliberately do NOT set ``IMWRITE_JPEG_LUMA_QUALITY`` /
      ``IMWRITE_JPEG_CHROMA_QUALITY`` — when those are provided, libjpeg uses
      them as per-channel quantisation scales and the master quality setting
      effectively becomes a no-op for file size.
    """
    quality = max(0, min(100, int(quality)))
    return [
        cv2.IMWRITE_JPEG_QUALITY, quality,
        cv2.IMWRITE_JPEG_OPTIMIZE, 1,
        cv2.IMWRITE_JPEG_PROGRESSIVE, 0,
        cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420,
    ]


def _validate_frame_bgr(frame: np.ndarray) -> None:
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(frame).__name__}.")
    if frame.dtype != np.uint8:
        raise ValueError(f"Expected uint8 frame, got dtype={frame.dtype}.")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(
            "Expected an 8-bit BGR frame with shape (H, W, 3); got shape "
            f"{frame.shape}. Convert YUV/RGB inputs to BGR before encoding."
        )


def encode_frame_to_jpeg(frame_bgr: np.ndarray, quality: int = DEFAULT_JPEG_QUALITY) -> bytes:
    """Encode an 8-bit BGR frame to JPEG bytes using the canonical params.

    The bytes returned are bit-identical to the on-disk files written by this
    script for the same ``frame_bgr`` and ``quality``.
    """
    _validate_frame_bgr(frame_bgr)
    ok, buf = cv2.imencode(".jpg", frame_bgr, make_jpeg_params(quality))
    if not ok:
        raise IOError("cv2.imencode failed to JPEG-encode frame.")
    return bytes(buf)


def encode_and_decode_frame(frame_bgr: np.ndarray, quality: int = DEFAULT_JPEG_QUALITY) -> np.ndarray:
    """Round-trip a raw BGR frame through the canonical JPEG codec.

    Equivalent to ``cv2.imread(path, IMREAD_COLOR)`` where ``path`` is one of
    the training JPEGs. Use this at inference time to apply the same lossy
    JPEG step the network was trained on.
    """
    jpeg_bytes = encode_frame_to_jpeg(frame_bgr, quality=quality)
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if decoded is None:
        raise IOError("cv2.imdecode failed on freshly encoded JPEG bytes.")
    return decoded


def list_videos(videos_dir: Path) -> list[Path]:
    if not videos_dir.is_dir():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")
    videos = sorted(
        p for p in videos_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    return videos


def extract_frames(
    video_path: Path,
    output_dir: Path,
    overwrite: bool,
    jpeg_params: list[int],
) -> tuple[int, int]:
    """Extract frames from ``video_path`` into ``output_dir``.

    Returns a ``(written, skipped)`` tuple.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # CAP_PROP_FRAME_COUNT can be unreliable for some containers; treat <=0 as unknown.
    progress_total = total_frames if total_frames > 0 else None

    written = 0
    skipped = 0
    frame_index = 0
    progress = tqdm(total=progress_total, desc=video_path.name, unit="frame", leave=False)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            out_path = output_dir / f"{frame_index}.jpg"
            if out_path.exists() and not overwrite:
                skipped += 1
            else:
                _validate_frame_bgr(frame)
                if not cv2.imwrite(str(out_path), frame, jpeg_params):
                    raise IOError(f"Failed to write {out_path}")
                written += 1

            frame_index += 1
            if hasattr(progress, "update"):
                progress.update(1)
    finally:
        capture.release()
        if hasattr(progress, "close"):
            progress.close()

    return written, skipped


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=DEFAULT_VIDEOS_DIR,
        help=f"Directory containing input videos (default: {DEFAULT_VIDEOS_DIR}).",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=DEFAULT_DATASETS_DIR,
        help=f"Root directory where per-video frame folders are written (default: {DEFAULT_DATASETS_DIR}).",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Skip frames that already exist on disk. By default existing frames are overwritten.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=DEFAULT_JPEG_QUALITY,
        help=(
            "JPEG quality 0-100 (default: %(default)s). Lower = smaller files. "
            "90-95 is visually lossless, 75-85 is solid for storage, <70 starts "
            "showing block artefacts."
        ),
    )
    parser.set_defaults(overwrite=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    videos_dir: Path = args.videos_dir
    datasets_dir: Path = args.datasets_dir

    videos = list_videos(videos_dir)
    if not videos:
        print(f"No videos found in {videos_dir}.", file=sys.stderr)
        return 1

    datasets_dir.mkdir(parents=True, exist_ok=True)
    jpeg_params = make_jpeg_params(args.quality)

    print(f"Found {len(videos)} video(s) in {videos_dir}.")
    print(f"Writing frames under {datasets_dir} (JPEG quality={args.quality}, 4:2:0).")

    total_written = 0
    total_skipped = 0
    for video_path in videos:
        out_dir = datasets_dir / video_path.stem / "Images"
        print(f"\n[{video_path.name}] -> {out_dir}")
        try:
            written, skipped = extract_frames(
                video_path, out_dir, overwrite=args.overwrite, jpeg_params=jpeg_params
            )
        except (RuntimeError, IOError, ValueError) as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            continue
        total_written += written
        total_skipped += skipped
        print(f"  wrote {written} frames, skipped {skipped} existing.")

    print(f"\nDone. Total frames written: {total_written}, skipped: {total_skipped}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
