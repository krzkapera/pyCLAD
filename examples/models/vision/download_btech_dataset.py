from __future__ import annotations

import hashlib
import logging
import shutil
import tempfile
import urllib.request
from pathlib import Path
from zipfile import ZipFile

DOWNLOAD_URL = "https://avires.dimi.uniud.it/papers/btad/btad.zip"
EXPECTED_SHA256 = "461c9387e515bfed41ecaae07c50cf6b10def647b36c9e31d239ab2736b10d2a"

SCRIPT_DIR = Path(__file__).resolve().parent
VISION_RESOURCES_DIR = SCRIPT_DIR.parents[1] / "resources" / "vision"
TARGET_DIR = VISION_RESOURCES_DIR / "BTech_Dataset_transformed"
ARCHIVE_PATH = VISION_RESOURCES_DIR / "btad.zip"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return

    logger.info("Downloading %s", url)
    with urllib.request.urlopen(url) as response, destination.open("wb") as output_file:
        shutil.copyfileobj(response, output_file)


def _safe_extract_zip(archive_path: Path, destination: Path) -> None:
    resolved_destination = destination.resolve()
    with ZipFile(archive_path, "r") as archive:
        for member in archive.infolist():
            member_path = (destination / member.filename).resolve()
            if not str(member_path).startswith(str(resolved_destination)):
                raise RuntimeError(f"Unsafe path in archive: {member.filename}")
        archive.extractall(destination)


def _find_dataset_root(base_dir: Path) -> Path | None:
    if (base_dir / "01").exists():
        return base_dir

    for candidate in base_dir.iterdir():
        if candidate.is_dir() and (candidate / "01").exists():
            return candidate
    return None


def main() -> None:
    if (TARGET_DIR / "01").exists():
        logger.info("Dataset already present at %s", TARGET_DIR)
        return

    _download(DOWNLOAD_URL, ARCHIVE_PATH)

    if _sha256(ARCHIVE_PATH) != EXPECTED_SHA256:
        raise RuntimeError("Downloaded archive hash does not match the expected value.")

    with tempfile.TemporaryDirectory(dir=VISION_RESOURCES_DIR) as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        _safe_extract_zip(ARCHIVE_PATH, tmp_dir)

        extracted_root = _find_dataset_root(tmp_dir)
        if extracted_root is None:
            raise RuntimeError("Could not find the extracted BTech dataset root.")

        TARGET_DIR.parent.mkdir(parents=True, exist_ok=True)
        if TARGET_DIR.exists():
            shutil.rmtree(TARGET_DIR)
        shutil.move(str(extracted_root), TARGET_DIR)

    logger.info("Dataset ready at %s", TARGET_DIR)


if __name__ == "__main__":
    main()
