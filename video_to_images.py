"""Convert all videos in ``videos/`` into per-frame JPG images.

For each video file ``videos/<name>.<ext>`` the script writes frames to
``datasets/<name>/Images/<frame_index>.jpg`` (frame_index starts at 0).

Frames are saved as JPEGs with optimised Huffman tables. The compression
quality is controlled by ``--quality`` (default 90 = visually lossless and
much smaller than 100).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import cv2

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


def make_jpeg_params(quality: int) -> list[int]:
    """Build cv2.imwrite JPEG parameters for the requested quality (0-100).

    Notes:
    - We deliberately do NOT set ``IMWRITE_JPEG_LUMA_QUALITY`` /
      ``IMWRITE_JPEG_CHROMA_QUALITY``. When those are provided, libjpeg uses
      them as per-channel quantisation scales and the master
      ``IMWRITE_JPEG_QUALITY`` setting effectively becomes a no-op for file
      size, which makes tuning ``quality`` look broken.
    - ``IMWRITE_JPEG_OPTIMIZE=1`` enables optimised Huffman tables (smaller
      file at the same visual quality).
    - ``IMWRITE_JPEG_PROGRESSIVE=0`` keeps files baseline so any reader can
      decode them.
    """
    quality = max(0, min(100, int(quality)))
    return [
        cv2.IMWRITE_JPEG_QUALITY, quality,
        cv2.IMWRITE_JPEG_OPTIMIZE, 1,
        cv2.IMWRITE_JPEG_PROGRESSIVE, 0,
    ]


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

    capture = cv2.VideoCapture(str(video_path))
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
    print(f"Writing frames under {datasets_dir} (JPEG quality={args.quality}).")

    total_written = 0
    total_skipped = 0
    for video_path in videos:
        out_dir = datasets_dir / video_path.stem / "Images"
        print(f"\n[{video_path.name}] -> {out_dir}")
        try:
            written, skipped = extract_frames(
                video_path, out_dir, overwrite=args.overwrite, jpeg_params=jpeg_params
            )
        except (RuntimeError, IOError) as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            continue
        total_written += written
        total_skipped += skipped
        print(f"  wrote {written} frames, skipped {skipped} existing.")

    print(f"\nDone. Total frames written: {total_written}, skipped: {total_skipped}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
