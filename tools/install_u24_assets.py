"""Install verified optional U24/U22 H3 quality assets with hash reporting."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
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

# Fixed author file IDs and SHA256 from Civitai model-version 3416478.
# Opt-in: these are candidates, not a replacement for the default U22 recipe.
MYSTIC_V5_ASSETS = (
    {
        "name": "minimax/MysticXXX_MMH3-V5-ref2va.safetensors", "category": "loras",
        "url": "https://civitai.com/api/download/models/3416478?fileId=3311427",
        "mirrors": ("https://huggingface.co/Sentinel7/h3/resolve/78182982f18da3feb9274d6bdd5562355e7a57ca/2856467/3416478/MysticXXX_MMH3-V5-ref2va.safetensors",),
        "sha256": "F57C76E761F7A5113E541AD58E84C648A5FE4D3CFBC277D0C44AE9763DA29AFE",
    },
    {
        "name": "minimax/MysticXXX_MMH3-V5.safetensors", "category": "loras",
        "url": "https://civitai.com/api/download/models/3416478?fileId=3309543",
        "mirrors": ("https://huggingface.co/Sentinel7/h3/resolve/78182982f18da3feb9274d6bdd5562355e7a57ca/2856467/3416478/MysticXXX_MMH3-V5.safetensors",),
        "sha256": "4168917DAEB6E570A004BC6994F73D091B9F3E887F7E946BD00C0F2F7573E5F3",
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "MiniMaxH3-Director-Plus/asset-installer"})
    with urllib.request.urlopen(request, timeout=60) as source, path.open("wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)


def install(asset: dict, root: Path, force: bool) -> tuple[Path, str]:
    destination = root / "models" / asset["category"] / asset["name"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = asset.get("sha256", "").lower()
    if force or not destination.is_file() or destination.stat().st_size == 0:
        partial = destination.with_suffix(destination.suffix + ".partial")
        print(f"Downloading {asset['name']}", flush=True)
        try:
            urls = (asset["url"], *asset.get("mirrors", ()))
            for index, url in enumerate(urls):
                try:
                    _download(url, partial)
                    break
                except OSError:
                    partial.unlink(missing_ok=True)
                    if index == len(urls) - 1:
                        raise
                    print(f"Source unavailable; trying verified mirror for {asset['name']}", flush=True)
            if partial.stat().st_size == 0:
                raise RuntimeError(f"Empty download: {asset['name']}")
            actual = sha256(partial)
            if expected and actual.lower() != expected:
                raise RuntimeError(f"SHA256 mismatch: {asset['name']}")
            partial.replace(destination)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
    else:
        actual = sha256(destination)
        if expected and actual.lower() != expected:
            raise RuntimeError(f"Cached SHA256 mismatch: {asset['name']}; use --force to download again")
    return destination, actual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--mystic-v5", action="store_true", help="Only install the two verified Mystic V5 candidates")
    parser.add_argument("--list", action="store_true", help="Only print URLs and destinations")
    args = parser.parse_args()
    root = args.comfy.resolve()
    if not (root / "main.py").is_file():
        parser.error(f"ComfyUI main.py not found in {root}")
    assets = MYSTIC_V5_ASSETS if args.mystic_v5 else ASSETS
    if args.list:
        for asset in assets:
            print(f"{asset['category']}/{asset['name']} <- {asset['url']}")
        return
    for asset in assets:
        path, digest = install(asset, root, args.force)
        print(f"{path.relative_to(root)} sha256={digest}")
    label = "Mystic V5 candidates" if args.mystic_v5 else "U24 optional assets"
    print(f"{label} ready. Restart ComfyUI to refresh the model list.")


if __name__ == "__main__":
    main()
