"""
setup_data.py .

This script:
  1. Re-uses kagglehub to resolve (or re-download, if already cached — instant)
     the dataset path.
  2. Finds the recipe JSON file inside it (largest *.json file).
  3. Finds the images folder inside it (folder with the most .jpg/.png/.webp files).
  4. Copies the JSON to data/recipes_images.json.
  5. Copies (or symlinks, to save disk space) the images into data/images/.

USAGE
-----
    python setup_data.py
"""

from __future__ import annotations

import os
import shutil
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"Running setup_data.py from {ROOT_DIR}")

DEST_DATA_DIR = "data"
DEST_JSON = os.path.join(DEST_DATA_DIR, "recipes_images.json")
DEST_IMAGES_DIR = os.path.join(DEST_DATA_DIR, "images")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def get_kaggle_dataset_path() -> str:
    import kagglehub
    path = kagglehub.dataset_download(
        "seungyeonhan1/recipe-dataset-with-images-tags-and-ratings"
    )
    print(f"Kaggle dataset resolved at: {path}")
    return path


def find_recipe_json(root: str) -> str:
    """Return the path to the largest .json file under root (assumed to be
    the recipes dataset, since Kaggle dataset dumps sometimes also include
    small metadata/license json files)."""
    candidates = []
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".json"):
                full = os.path.join(dirpath, f)
                candidates.append((os.path.getsize(full), full))
    if not candidates:
        raise FileNotFoundError(f"No .json files found under {root}")
    candidates.sort(reverse=True)
    largest_size, largest_path = candidates[0]
    print(f"Found recipe JSON: {largest_path} ({largest_size / 1e6:.1f} MB)")
    if len(candidates) > 1:
        print(f"  (found {len(candidates)} .json files total; picked the largest)")
    return largest_path


def find_images_dir(root: str) -> str | None:
    """Return the folder under root containing the most image files."""
    best_dir = None
    best_count = 0
    for dirpath, _dirs, files in os.walk(root):
        count = sum(1 for f in files if os.path.splitext(f)[1].lower() in IMAGE_EXTS)
        if count > best_count:
            best_count = count
            best_dir = dirpath
    if best_dir:
        print(f"Found images folder: {best_dir} ({best_count:,} images)")
    else:
        print("No images folder found under the dataset — CLIP index will be skipped.")
    return best_dir


def copy_json(src: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        print(f"{dest} already exists — leaving as-is (delete it first to re-copy).")
        return
    print(f"Copying {src} -> {dest} …")
    shutil.copy2(src, dest) # this preserves timestamps, which is nice for reproducibility
    print("Done.")


def copy_images(src_dir: str, dest_dir: str) -> None:
    if os.path.isdir(dest_dir) and os.listdir(dest_dir):
        print(f"{dest_dir} already has files — leaving as-is (delete it first to re-copy).")
        return
    os.makedirs(dest_dir, exist_ok=True)
    print(f"Copying images from {src_dir} -> {dest_dir} … (this may take a while)")
    count = 0
    for f in os.listdir(src_dir):
        ext = os.path.splitext(f)[1].lower()
        if ext in IMAGE_EXTS:
            shutil.copy2(os.path.join(src_dir, f), os.path.join(dest_dir, f))
            count += 1
            if count % 500 == 0:
                print(f"  …{count:,} images copied so far")
    print(f"Done. {count:,} images copied to {dest_dir}")


def main():
    dataset_root = get_kaggle_dataset_path() # downloads data in cache and returns the path to it

    json_path = find_recipe_json(dataset_root) # gets the path to the largest json file in the dataset
    copy_json(json_path, DEST_JSON) # copies the json file to data/recipes_images.json

    images_dir = find_images_dir(dataset_root) # gets the path to the folder with the most images in the dataset
    if images_dir:
        copy_images(images_dir, DEST_IMAGES_DIR) # copies the images to data/images/

    print("\nAll set. Next steps:")
    print("  python build_indexes.py")
    print("  python app.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
