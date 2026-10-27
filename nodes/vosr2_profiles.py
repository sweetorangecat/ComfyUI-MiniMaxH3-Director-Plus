"""Explicit VOSR2 experiments; no unmeasured profile is selected automatically."""
from __future__ import annotations

import math

VOSR2_PROFILE_NAMES = ('legacy', 'manual_baseline', 'experimental_32gb')


def resolve_vosr2_profile(profile='legacy', *, free_vram_gb=None, total_vram_gb=None,
                         supports_tile_batch=False):
    """Return fresh upstream kwargs. Memory values are measured GiB, not card labels."""
    if profile not in VOSR2_PROFILE_NAMES:
        raise ValueError(f'未知 VOSR2 配置：{profile!r}；可选：{", ".join(VOSR2_PROFILE_NAMES)}')
    experimental = profile == 'experimental_32gb'
    values = (free_vram_gb, total_vram_gb)
    measured = all(isinstance(v, (int, float)) and not isinstance(v, bool)
                   and math.isfinite(v) for v in values)
    roomy = (measured and 0 < free_vram_gb <= total_vram_gb
             and total_vram_gb >= 30 and free_vram_gb >= 24)
    if experimental:
        if not roomy:
            raise ValueError('VOSR2 experimental_32gb 需要释放模型后实测总显存至少 30 GiB、'
                             '空闲显存至少 24 GiB；无法测量或不足时请显式选择 manual_baseline。')
    legacy = profile == 'legacy'
    # The Linux compatibility backend batches independent spatial DiT tiles.
    # A fully idle 32GB card has enough headroom for eight tiles; keeping four
    # for the lower-headroom case avoids trading an OOM for a small speed gain.
    tile_batch = (
        8 if legacy and supports_tile_batch is True and roomy
        and float(free_vram_gb) >= 29.0
        else 4 if legacy and supports_tile_batch is True and roomy else 1
    )
    frame_batch = 4 if experimental else 2
    policy = 'auto' if legacy else 'staged'
    return {
        'name': profile,
        'experimental': experimental,
        'loader': dict(model_bundle='VOSR2', precision='auto', memory_policy=policy,
                       torch_compile=False, vae_encode_amp=False),
        'settings': dict(quality_profile='speed' if legacy else 'manual',
                         tile_strategy='auto' if legacy else 'tiled',
                         tile_size=512, tile_overlap=32, vae_tile_size=1024,
                         vae_tile_overlap=32, image_batch=tile_batch, frame_batch=frame_batch,
                         dino_batch=frame_batch, temporal_cache=False,
                         cache_threshold=0.003, cache_refresh=4,
                         memory_policy=policy, color_alignment='wavelet'),
        'video': dict(frame_batch=frame_batch, temporal_cache=False,
                      cache_threshold=0.003, cache_refresh=4),
    }
