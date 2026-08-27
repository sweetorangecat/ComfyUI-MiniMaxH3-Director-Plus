# Route-isolated H3 quality presets

## Goal

Add explicit performance choices for the two H3 model contracts without changing existing saved workflow behavior:

- `高清快速（v4 8步）` / `fl_quality_fast_v4`: FL/T2VA-family only, official community v4 LoRA, 8 steps, simple/Euler, LoRA strength 1.0, no latent two-stage pass.
- `参考高清（原生20步）` / `ref_quality_native`: REF2VA or H3 voice-reference route only, native 20 steps with SageAttention, no Turbo LoRA and no EasyCache.
- `参考极速（官方4步）` / `ref_fast_4step`: REF2VA or H3 voice-reference route only, official Ref2VA Turbo LoRA, 4 steps, stock Euler, no EasyCache.

Existing `稳定质量`, `质量优先加速`, `质量优先二采样`, `极速4步`, `参考图加速`, `低显存`, and `低显存二采` retain their current contracts. A saved workflow containing a route-incompatible new key falls back to stable quality during normalization.

## Route contract

The schema and frontend expose FL-only v4 on non-reference routes and REF-only choices on reference routes. Fish remains subject to its existing restrictions and cannot use trained latent two-stage presets.

## Observability

The guide records the resolved preset, backend, LoRA name/strength, steps, scheduler and applied acceleration flags. UI and API documentation use Chinese labels and explain that these are selectable presets, not manual node connections.

