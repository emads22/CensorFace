from pathlib import Path


ASSETS_DIR = Path("./assets")
VIDEOS_DIR = ASSETS_DIR / "videos"
IMAGES_DIR = ASSETS_DIR / "images"
OUTPUT_DIR = ASSETS_DIR / "output"

LINE_SEPARATOR = "=" * 80
VIDEOS_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]  # Common video extensions
CENSORING_METHODS = ["blur", "box", "cat"]
BOX_COLOR = (64, 64, 64)  # Dark Gray
CAT_FACE = IMAGES_DIR / "cat_face.png"
