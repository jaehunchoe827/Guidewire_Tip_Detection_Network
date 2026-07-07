"""Organize the guidewire dataset by removing images that have no matching label.

The dataset lives under ``datasets/guidewire`` and is laid out as::

    datasets/guidewire/<sequence>/Images/<stem>.jpg
    datasets/guidewire/<sequence>/Labels/<stem>.txt

Not every image has a corresponding label. This script removes the label-less
images, but only after a sanity check confirms that the number of images that
would remain exactly matches the total number of labels. A post-deletion sanity
check then verifies every label still has its matching image.

Usage::

    python3 -m engine.organize_dataset                # perform the cleanup
    python3 -m engine.organize_dataset --dry-run      # report only, delete nothing
"""

import argparse
from pathlib import Path

IMAGE_EXT = ".jpg"
LABEL_EXT = ".txt"
IMAGES_DIR = "Images"
LABELS_DIR = "Labels"


def default_dataset_root() -> Path:
    # engine/ -> gwtd_src/ -> datasets/guidewire
    return Path(__file__).resolve().parent.parent / "datasets" / "guidewire"


def collect_sequences(root: Path):
    sequences = []
    for seq_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        images_dir = seq_dir / IMAGES_DIR
        labels_dir = seq_dir / LABELS_DIR
        if not images_dir.is_dir() or not labels_dir.is_dir():
            print(f"  [skip] {seq_dir.name}: missing Images/ or Labels/ folder")
            continue

        label_stems = {p.stem for p in labels_dir.glob(f"*{LABEL_EXT}")}
        image_paths = sorted(images_dir.glob(f"*{IMAGE_EXT}"))
        keep = [p for p in image_paths if p.stem in label_stems]
        remove = [p for p in image_paths if p.stem not in label_stems]

        sequences.append(
            {
                "name": seq_dir.name,
                "labels": label_stems,
                "images": image_paths,
                "keep": keep,
                "remove": remove,
            }
        )
    return sequences


def pre_sanity_check(sequences) -> bool:
    total_labels = sum(len(s["labels"]) for s in sequences)
    total_images = sum(len(s["images"]) for s in sequences)
    total_keep = sum(len(s["keep"]) for s in sequences)
    total_remove = sum(len(s["remove"]) for s in sequences)

    print("=" * 60)
    print("PRE-DELETION SANITY CHECK")
    print("=" * 60)
    print(f"{'sequence':<12}{'labels':>10}{'images':>10}{'keep':>10}{'delete':>10}")
    print("-" * 52)
    for s in sequences:
        print(
            f"{s['name']:<12}{len(s['labels']):>10}{len(s['images']):>10}"
            f"{len(s['keep']):>10}{len(s['remove']):>10}"
        )
    print("-" * 52)
    print(
        f"{'TOTAL':<12}{total_labels:>10}{total_images:>10}"
        f"{total_keep:>10}{total_remove:>10}"
    )
    print()
    print(f"Total labels               : {total_labels}")
    print(f"Total images               : {total_images}")
    print(f"Images to remain (have label): {total_keep}")
    print(f"Images to delete (no label) : {total_remove}")
    print()

    # Labels without a matching image would make keep < labels.
    labels_without_image = []
    for s in sequences:
        kept_stems = {p.stem for p in s["keep"]}
        missing = s["labels"] - kept_stems
        if missing:
            labels_without_image.append((s["name"], sorted(missing)))

    if labels_without_image:
        print("WARNING: some labels have no matching image:")
        for name, stems in labels_without_image:
            preview = ", ".join(stems[:10]) + (" ..." if len(stems) > 10 else "")
            print(f"  {name}: {len(stems)} missing -> {preview}")
        print()

    if total_keep != total_labels:
        print(
            f"CHECK FAILED: images to remain ({total_keep}) does NOT match "
            f"total labels ({total_labels}). Aborting, no files deleted."
        )
        return False

    print(
        f"CHECK PASSED: images to remain ({total_keep}) matches "
        f"total labels ({total_labels})."
    )
    return True


def delete_images(sequences, dry_run: bool) -> int:
    deleted = 0
    for s in sequences:
        for path in s["remove"]:
            if dry_run:
                deleted += 1
            else:
                path.unlink()
                deleted += 1
    return deleted


def post_sanity_check(root: Path) -> bool:
    print()
    print("=" * 60)
    print("POST-DELETION SANITY CHECK")
    print("=" * 60)

    ok = True
    total_labels = 0
    total_images = 0
    for seq_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        images_dir = seq_dir / IMAGES_DIR
        labels_dir = seq_dir / LABELS_DIR
        if not images_dir.is_dir() or not labels_dir.is_dir():
            continue

        label_stems = {p.stem for p in labels_dir.glob(f"*{LABEL_EXT}")}
        image_stems = {p.stem for p in images_dir.glob(f"*{IMAGE_EXT}")}
        total_labels += len(label_stems)
        total_images += len(image_stems)

        labels_no_image = label_stems - image_stems
        images_no_label = image_stems - label_stems
        if labels_no_image or images_no_label:
            ok = False
            print(f"  [FAIL] {seq_dir.name}: "
                  f"{len(labels_no_image)} labels w/o image, "
                  f"{len(images_no_label)} images w/o label")

    print(f"\nTotal labels: {total_labels}, total images: {total_images}")
    if ok and total_labels == total_images:
        print("CHECK PASSED: every label has exactly one matching image.")
    else:
        ok = False
        print("CHECK FAILED: mismatch between labels and images.")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=default_dataset_root(),
        help="Path to datasets/guidewire (default: auto-detected).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report counts and run checks without deleting any files.",
    )
    args = parser.parse_args()

    root = args.root
    if not root.is_dir():
        raise SystemExit(f"Dataset root not found: {root}")

    print(f"Dataset root: {root}\n")
    sequences = collect_sequences(root)

    if not pre_sanity_check(sequences):
        raise SystemExit(1)

    if args.dry_run:
        would_delete = sum(len(s["remove"]) for s in sequences)
        print(f"\n[dry-run] Would delete {would_delete} images. No files changed.")
        return

    deleted = delete_images(sequences, dry_run=False)
    print(f"\nDeleted {deleted} images without a matching label.")

    if not post_sanity_check(root):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
