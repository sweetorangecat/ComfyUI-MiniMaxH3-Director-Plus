# Adaptive Video Quality

Goal: retain the existing 1080p / 2K / 4K output choices, rename the automatic
preset, and choose the backend without another user-facing quality tier.

- [x] Inspect U22, installed nodes, generation log, and current upstream documentation.
- [x] Reproduce the missing automatic-name alias and 24GB/15-second policy failures.
- [x] Implement an automatic QHD plan for 4-15 seconds. At >=28GB total and
  >=24GB free, use 1280x736 -> 2560x1472 and aspect-preserving QHD export.
  At >=20GB total and >=18GB free, use 960x544 -> 1920x1088 then SeedVR2.
  These are engineering budgets, not measured guarantees of GPU fit.
- [x] Require complete temporal/spatial sampling dependencies before starting
  any automatically planned tiled pass; never fall back to full-frame on absence.
- [x] Reduce tile/chunk concurrency on CUDA OOM with bounded retries, retaining
  resolution, seed, sigmas, and frame count. Fail clearly if the minimum fails.
- [x] Rename the public automatic preset, preserve old serialized values, update
  the template and builder. Keep three resolution options and 4-15 second input.
- [x] Fix aspect handling before SeedVR2 so output is not stretched from 1408 to 1440.
- [x] Verify policy, integration, fallback, frontend and workflow tests; document
  hardware validation limitations. CPU/ComfyUI-stub suite: 824 passed,
  1 CUDA-only test skipped; frontend syntax and existing Node contract pass.
- [x] Prepare the verified changes for delivery to main. The existing U11
  graph validates: 26 nodes, 25 links, one subgraph, no visible overlaps.

Delivery target: push main and fast-forward the clean installed plugin.
Git history and the final delivery report record the resulting revision.

Sources checked on 2026-09-06:

- https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler
  README: learned scales include 2x (40%) and 1.5x (10%). The 2026-08-28
  split sampler supplies temporal chunks, spatial tiles, blending and anchors.
  Latent upscaling alone does not reduce the refinement pass's VRAM requirement.
- https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director
  Refine supports an H3 latent target canvas and optional separate refine model.
  No independently verified YZ/5090/15-second QHD performance claim was found.
- User log: 192 frames, source 1920x1056, SeedVR2 2560x1408, export 2560x1440,
  32m14s total; SeedVR2 VAE decode alone 8m29s. Video file is unavailable locally.

The previous clarity-labelled workflow changed only labels/notes. This change
must include tested executable routing changes. Quality and speed still require
matched GPU A/B samples; unit tests cannot establish visual quality or VRAM fit.
