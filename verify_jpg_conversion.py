"""Verify that the canonical JPEG conversion of a raw FFV1 frame matches the
on-disk training JPEG for every frame of a single video.

For ``VIDEO_NAME = "<name>"`` this script:
    1. Opens ``videos/<name>.avi`` (lossless FFV1) and decodes it frame by frame.
    2. Runs :func:`jpg_convert` on each raw BGR frame to obtain a numpy array
       representing the JPEG-converted image (encode -> decode round trip).
    3. Loads ``datasets/guidewire/<name>/Images/<i>.jpg`` with ``cv2.imread``.
    4. Compares the two arrays pixel-for-pixel.

A clean run (0 mismatches, 0 missing) means a real-time
``raw_frame -> jpg_convert -> network`` pipeline will feed the detection
network the exact same pixels it was trained on.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from video_to_images import DEFAULT_JPEG_QUALITY, make_jpeg_params

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional; fall back to a no-op wrapper.
    def tqdm(iterable, **_kwargs):
        return iterable


# Configure here.
VIDEO_NAME = "103"
JPEG_QUALITY = DEFAULT_JPEG_QUALITY

SCRIPT_DIR = Path(__file__).resolve().parent
VIDEO_PATH = SCRIPT_DIR / "videos" / f"{VIDEO_NAME}.avi"
IMAGES_DIR = SCRIPT_DIR / "datasets" / "guidewire" / VIDEO_NAME / "Images"


def jpg_convert(frame_bgr: np.ndarray, quality: int = JPEG_QUALITY) -> np.ndarray:
    """Apply the dataset's JPEG conversion to a raw BGR frame.

    Encodes ``frame_bgr`` with the canonical JPEG parameters from
    ``video_to_images.make_jpeg_params`` and immediately decodes it back to a
    uint8 BGR numpy array. The returned array is bit-identical to what
    ``cv2.imread(path, IMREAD_COLOR)`` returns for the matching file under
    ``datasets/guidewire/<video>/Images/``.
    """
    if (
        not isinstance(frame_bgr, np.ndarray)
        or frame_bgr.dtype != np.uint8
        or frame_bgr.ndim != 3
        or frame_bgr.shape[2] != 3
    ):
        raise ValueError(
            "jpg_convert expects an 8-bit BGR ndarray with shape (H, W, 3); "
            f"got dtype={getattr(frame_bgr, 'dtype', type(frame_bgr).__name__)}, "
            f"shape={getattr(frame_bgr, 'shape', None)}."
        )

    ok, buf = cv2.imencode(".jpg", frame_bgr, make_jpeg_params(quality))
    if not ok:
        raise IOError("cv2.imencode failed to JPEG-encode frame.")
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if decoded is None:
        raise IOError("cv2.imdecode failed on freshly encoded JPEG bytes.")
    return decoded


def main() -> int:
    if not VIDEO_PATH.is_file():
        print(f"Video not found: {VIDEO_PATH}", file=sys.stderr)
        return 1
    if not IMAGES_DIR.is_dir():
        print(f"Images directory not found: {IMAGES_DIR}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(str(VIDEO_PATH), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"Could not open video: {VIDEO_PATH}", file=sys.stderr)
        return 1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_total = total_frames if total_frames > 0 else None

    print(f"Video  : {VIDEO_PATH}")
    print(f"Images : {IMAGES_DIR}")
    print(f"Quality: {JPEG_QUALITY}")

    matched = 0
    mismatched = 0
    missing = 0
    frame_index = 0
    first_problem: tuple[int, dict] | None = None

    # jpg_convert timings, in seconds, one entry per frame we actually convert.
    convert_times: list[float] = []

    progress = tqdm(total=progress_total, unit="frame", desc=VIDEO_NAME, leave=True)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            jpg_path = IMAGES_DIR / f"{frame_index}.jpg"

            if not jpg_path.is_file():
                missing += 1
                if first_problem is None:
                    first_problem = (frame_index, {"reason": "jpg missing", "path": str(jpg_path)})
            else:
                t0 = time.perf_counter()
                converted = jpg_convert(frame)
                convert_times.append(time.perf_counter() - t0)

                disk = cv2.imread(str(jpg_path), cv2.IMREAD_COLOR)
                if disk is None:
                    mismatched += 1
                    if first_problem is None:
                        first_problem = (frame_index, {"reason": "imread returned None", "path": str(jpg_path)})
                elif disk.shape != converted.shape:
                    mismatched += 1
                    if first_problem is None:
                        first_problem = (
                            frame_index,
                            {"reason": "shape mismatch", "disk": disk.shape, "converted": converted.shape},
                        )
                elif not np.array_equal(disk, converted):
                    mismatched += 1
                    if first_problem is None:
                        diff = cv2.absdiff(disk, converted)
                        first_problem = (
                            frame_index,
                            {
                                "reason": "pixels differ",
                                "max_abs_diff": int(diff.max()),
                                "mean_abs_diff": float(diff.mean()),
                                "n_diff_pixels": int(np.count_nonzero(diff.any(axis=-1))),
                            },
                        )
                else:
                    matched += 1

            frame_index += 1
            if hasattr(progress, "update"):
                progress.update(1)
    finally:
        cap.release()
        if hasattr(progress, "close"):
            progress.close()

    print()
    print(f"Frames decoded from video: {frame_index}")
    print(f"  matched   : {matched}")
    print(f"  mismatched: {mismatched}")
    print(f"  missing   : {missing}")
    if first_problem is not None:
        idx, info = first_problem
        print(f"First problem at frame {idx}: {info}")

    if convert_times:
        times_ms = np.asarray(convert_times) * 1000.0
        print()
        print(f"jpg_convert() timing over {len(times_ms)} frames:")
        print(f"  avg : {times_ms.mean():.3f} ms  ({1000.0 / times_ms.mean():.1f} fps)")
        print(f"  min : {times_ms.min():.3f} ms")
        print(f"  max : {times_ms.max():.3f} ms")
        print(f"  std : {times_ms.std():.3f} ms")

    if mismatched == 0 and missing == 0 and frame_index > 0:
        print("OK: every frame's jpg_convert(frame) matches the on-disk JPEG.")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
