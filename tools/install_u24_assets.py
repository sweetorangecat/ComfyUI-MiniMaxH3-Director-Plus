"""Install verified optional U24/U22 H3 quality assets with hash reporting."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import urllib.request


ASSETS = (
    {
        "name": "Motion_Repair.safetensors",
        "category": "loras",
        "url": "https://huggingface.co/JOKER141/MiniMax-H3-General-Motion-Continuity-Repair/resolve/main/Motion_Repair.safetensors",
    },
    {
        "name": "H3_Combat_V2.safetensors",
        "category": "loras",
        "url": "https://huggingface.co/JOKER141/MiniMax-H3-Combat-Base-V2/resolve/main/H3_Combat_V2.safetensors",
    },
    {
        "name": "minimax_h3_turbo_4step_10ErosMax_test4_pruned_curveproj1025_exp_v001-T8.safetensors",
        "category": "loras",
        "url": "https://huggingface.co/10ErosMax/Minimax-H3-Turbo-4step/resolve/main/minimax_h3_turbo_4step_10ErosMax_test4_pruned_curveproj1025_exp_v001-T8.safetensors",
    },
    {
        "name": "minimax_h3_latent_upscaler_3d_fp16.safetensors",
        "category": "latent_upscale_models",
        "url": "https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/resolve/main/minimax_h3_latent_upscaler_3d_fp16.safetensors",
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install(asset: dict, root: Path, force: bool) -> tuple[Path, str]:
    destination = root / "models" / asset["category"] / asset["name"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if force or not destination.is_file() or destination.stat().st_size == 0:
        partial = destination.with_suffix(destination.suffix + ".partial")
        print(f"Downloading {asset['name']}", flush=True)
        urllib.request.urlretrieve(asset["url"], partial)
        if partial.stat().st_size == 0:
            raise RuntimeError(f"Empty download: {asset['name']}")
        partial.replace(destination)
    return destination, sha256(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true", help="Only print URLs and destinations")
    args = parser.parse_args()
    root = args.comfy.resolve()
    if not (root / "main.py").is_file():
        parser.error(f"ComfyUI main.py not found in {root}")
    if args.list:
        for asset in ASSETS:
            print(f"{asset['category']}/{asset['name']} <- {asset['url']}")
        return
    for asset in ASSETS:
        path, digest = install(asset, root, args.force)
        print(f"{path.relative_to(root)} sha256={digest}")
    print("U24 optional assets ready. Restart ComfyUI and check the capability panel.")


if __name__ == "__main__":
    main()
