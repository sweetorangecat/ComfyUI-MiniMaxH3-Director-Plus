# H3 balanced 8-step and measured finishing implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans to implement these independently testable tasks. The user approved this design on 2026-10-18; no additional execution approval is needed.

**Goal:** Add an opt-in mode-specific 768p eight-step recipe, reproducible VOSR2 comparisons, and honest execution/timing reports without changing saved quality recipes.

**Architecture:** A small recipe module owns the new adapter/sampler/shift contract. Existing director, schema, API and UI select that recipe; acceleration and samplers enforce it. VOSR2 profiles and stage diagnostics remain separate, with existing defaults retained and no automatic adoption of unmeasured candidates.

**Tech Stack:** Python, ComfyUI native H3 nodes, pytest, vanilla JavaScript.

## Task 1: Explicit balanced recipe

- [x] Add failing tests in `tests/test_balanced_recipe.py`: all five modes and four voice modes route to the correct eight-step adapter; FL and REF names differ; sampler=Euler, scheduler=simple, strength=1, shifts=6/3; legacy 20-step remains unchanged; missing adapters reject before sampling; downstream manual shifts normalize on cloned scheduler and guider models, and mismatched samplers/sigmas reject.
- [x] Implement `nodes/sampling_recipes.py`; integrate `nodes/performance.py`, `nodes/schema.py`, `nodes/director.py` and `js/minimax_h3_director_plus_v10.js`. New key `balanced_8step`, label `均衡高质（768p 8步）`. Keep first/last hard endpoint limitations visible for existing reference-audio routes. Do not install T8 or sparse attention in this batch.
- [x] Verify preset menus, native conditioning, fixed sigma count and model/sampler validation. Add precise download instructions for both adapter files; do not silently use the older 544p or four-step adapter.

## Task 2: VOSR2 profiles and comparison harness

- [x] Test named profiles and forwarding into both settings and video calls, invalid profiles, output frame count, and minimum-resolution preservation.
- [x] Add `nodes/vosr2_profiles.py` with legacy, manual baseline and an explicitly experimental 32GB candidate. Wire an optional trailing director/widget/API field `vosr2_profile`, preserving old widget order and the legacy default. Require valid measured memory for the 32GB candidate; never auto-select it on total VRAM alone.
- [x] Add a short-clip comparison node/tool using identical input/seed/scale, writing configuration, wall time, memory and output paths for manual quality comparison. Do not call an untested candidate fastest or automatically persist it as a winner.
- [x] Keep temporal cache and compile disabled. Retain VOSR2 release barriers and output dimensions >= requested target.

## Task 3: Execution evidence

- [x] Test report construction with references, endpoints, recipe settings and completed stage measurements. Timing on failure must not appear successful.
- [x] Implement bounded metadata in `nodes/execution_report.py`; add conditioning, single-pass/two-stage, decode, reconstruction and output timings where the corresponding work actually executes. Distinguish combined postprocessing/encoding from isolated kernel timing; no estimated speed multiplier.
- [x] Show route and endpoint semantics before running, print selected settings in runtime logs, and include actual recipe/timing evidence in saved output metadata. Preserve existing audio and frame count behavior.

## Task 4: Verification and integration

- [x] Run focused red/green tests with `D:/ComfyUI_windows_portable-G313/python_embeded/python.exe -m pytest` (add project root to PYTHONPATH for the embedded runtime), then the relevant/full CPU suite. The user's pre-existing deletion of `examples/H3-VideoFinish-480p-768p-1080p.json` must not be restored; report or exclude its fixture-only test.
- [x] Validate JavaScript syntax and shipped workflow links; review diff for backward compatibility and unrelated changes.
- [ ] Commit/push only task-owned files, including the already-tested manual 32GB tile-budget fix. Do not include `.gitignore`, the deleted example or `%SystemDrive%/`.

GPU acceptance remains separate: use identical 5s then 15s inputs across all supported modes, with and without voice references; assess identity, endpoints, motion, detail, speech and lip sync. The current local CPU checks cannot establish performance on the user's AutoDL RTX 5090.

## Verification record (2026-10-18)

- Full CPU suite: 1341 passed, 1 fixture-only test deselected because the user deleted its example; existing pynvml deprecation warning only.
- JavaScript: 79 tests passed; syntax check passed; shipped U11 layout/link validation passed.
- Actual installed ComfyUI CPU integration: native Guider_Basic, MiniMaxH3SigmaShift, simple_scheduler and Euler agree for both FL/REF recipes after legacy 11/4 shift; this does not test GPU weights or image/voice quality.
- Spec review found and fixed native guider API mismatch and missing actual sampling-setting evidence; final spec review approved.
- Final independent quality review approved with no remaining findings. Integration is authorized directly to main; verify the pushed commit after publication.
