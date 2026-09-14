"""Install optional dependencies and weights for Director Plus face refinement."""

import argparse
from pathlib import Path
import subprocess
import sys
import urllib.request


ASSETS = (
    ("https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt",
     "ultralytics/bbox/face_yolov8m.pt"),
    ("https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/"
     "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
     "loras/minimax/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"),
)


def install_asset(url, destination):
    if destination.is_file() and destination.stat().st_size:
        print(f"Exists: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    print(f"Downloading {destination.name}", flush=True)
    urllib.request.urlretrieve(url, partial)
    if not partial.stat().st_size:
        raise RuntimeError(f"Empty download: {destination.name}")
    partial.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy", type=Path, required=True)
    parser.add_argument("--skip-pip", action="store_true")
    args = parser.parse_args()
    root = args.comfy.resolve()
    if not (root / "main.py").is_file():
        parser.error(f"ComfyUI main.py not found in {root}")
    if not args.skip_pip:
        requirements = Path(__file__).resolve().parents[1] / "requirements-facerefine.txt"
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements)], check=True)
    for url, relative in ASSETS:
        install_asset(url, root / "models" / relative)
    print("FaceRefine dependencies ready. Restart ComfyUI to load the Director Plus nodes.")


if __name__ == "__main__":
    main()
