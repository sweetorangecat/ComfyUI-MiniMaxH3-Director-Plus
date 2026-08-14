# U11 Mode Performance Isolation

## Goal

Prevent performance presets from being offered or executed for incompatible MiniMax H3 modes. Voice-reference workflows in I2VA, FL2VA, L2VA, and T2VA must retain access to the REF2VA-compatible acceleration set.

## Preset Matrix

The matrix is selected from the requested mode and voice mode:

| Route | Allowed presets |
|---|---|
| T2VA without voice reference | stable quality, low VRAM |
| I2VA/FL2VA/L2VA without voice reference | stable quality, fast 4-step, low VRAM |
| REF2VA without voice reference | stable quality, reference fast, fast 4-step, low VRAM |
| Any mode with H3 or Fish voice reference | stable quality, reference fast, fast 4-step, low VRAM |

`custom` remains accepted for legacy/API compatibility but is hidden from the normal Director Plus controls. `quality` is the fallback for an invalid legacy combination.

## Behavior

- The schema exposes a pure helper that returns allowed presets and a safe default.
- Request normalization maps Chinese and internal preset names, then replaces an invalid combination with `quality` and appends a Chinese warning.
- The custom Director Plus UI derives its performance dropdown from the same route-aware matrix and automatically selects the safe default when mode or voice mode changes.
- Existing serialized workflows continue to load; invalid saved values are corrected at execution time instead of failing late in sampling.

## Testing

- Unit tests cover every matrix row, voice-reference routing for I2VA/FL2VA/L2VA/T2VA, invalid fallback warnings, and legacy `custom` acceptance.
- Existing performance, schema, API, and workflow validation tests must remain green.
