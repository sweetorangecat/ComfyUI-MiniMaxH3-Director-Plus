"""Opt-in lossless video-latent snapshots for locating two-stage artifacts."""

import json
import logging
import os
from pathlib import Path
from uuid import uuid4


LOGGER = logging.getLogger("MiniMaxH3.DirectorPlus.Diagnostics")


class StageDiagnostics:
    def __init__(self, guide, seed):
        self.directory = None
        self.manifest = {}
        if os.environ.get("MINIMAX_H3_STAGE_DIAGNOSTICS") != "1":
            return
        try:
            import folder_paths

            directory = Path(folder_paths.get_output_directory()) / "h3_stage_diagnostics" / uuid4().hex
            directory.mkdir(parents=True, exist_ok=False)
            self.directory = directory
            # Whitelist scalar settings: never serialize images, voices, or prompts.
            self.manifest = {"seed": seed, "stages": {}, "settings": {
                key: value for key, value in guide.items()
                if key.startswith(("two_stage_", "split_", "second_stage_"))
                and isinstance(value, (str, int, float, bool, type(None)))
            }}
            LOGGER.info("[H3 diagnostics] 阶段 latent 留档已开启: %s", directory)
        except (OSError, ImportError) as error:
            LOGGER.warning("[H3 diagnostics] 无法开启留档: %s", error)

    def save(self, stage, samples):
        if self.directory is None:
            return
        try:
            import torch
            from safetensors.torch import save_file

            if getattr(samples, "is_nested", False):
                samples = samples.unbind()[0]
            # Save only the complete video stream. Temporal subsets change VAE
            # context and cannot establish whether a boundary artifact is real.
            tensor = samples.detach().to(device="cpu").contiguous()
            path = self.directory / f"{stage}.latent"
            save_file({"latent_tensor": tensor, "latent_format_version_0": torch.empty(0)}, str(path))
            self.manifest["stages"][stage] = {"file": path.name, "shape": list(tensor.shape),
                                               "dtype": str(tensor.dtype)}
            (self.directory / "manifest.json").write_text(
                json.dumps(self.manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            LOGGER.info("[H3 diagnostics] 已保存 %s: %s", stage, path)
        except (OSError, RuntimeError, ValueError, TypeError, ImportError) as error:
            LOGGER.warning("[H3 diagnostics] %s 留档失败，继续原生成流程: %s", stage, error)

