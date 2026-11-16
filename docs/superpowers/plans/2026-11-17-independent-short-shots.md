# Independent Short Shots Implementation Plan

> **For agentic workers:** Execute inline with `superpowers:executing-plans`. Existing user authorization covers implementation, tests and deployment; do not open a new approval or delegation flow.

**Goal:** Generate editorial long videos by repeating the original <=15-second route at authored cuts, avoiding recursive generation history.

**Architecture:** Add a pure shot planner/graph compiler, a disk-backed independent coordinator, and a short-output receipt adapter. Reuse the existing journal, queue backend, progress API and bounded-memory assembler; keep native AV continuation separate.

**Tech Stack:** Python, existing ComfyUI nodes, PyAV/PCM and pytest; original JavaScript progress panel.

## Task 1: Compile independent shots

- [x] Add `tests/test_long_video_shots.py` with a 45-second/three-shot prompt. Assert 360 delivered frames per clip, local timestamps, complete dialogue and unchanged source graph, model, sampling, LoRA and reference inputs.
- [x] Observe the missing-module failure before implementation; use the installed Comfy Python for integrated tests because a generic Python lacks Comfy's `folder_paths`.
- [x] Implement `nodes/long_video_shots.py`: parse the existing six-section syntax, select authored cuts, pack whole shots within 4–15 seconds, compile local prompts, preserve budgets, and clone the exact original short graph.
- [x] Add automatic selection before native-long settings are forced. Single-shot/<=15-second behavior remains as before.

## Task 2: Queue, review and assemble independent clips

- [x] Add focused lifecycle tests in `tests/test_long_video_shot_service.py` for the real journal with an injected queue and media fixture.
- [x] Implement `nodes/long_video_shot_service.py`, sharing the existing queue reconciler and publication identity. Revalidate frozen code/assets on resume. Keep completed independent clips reusable after another clip is retried.
- [x] Add `nodes/long_video_shot_output.py`. Slice delivery images and original PCM before the existing output postprocessing, save a technical receipt and publish only against the immutable request.
- [x] Select the independent coordinator in the existing driver; register its internal output adapter. Keep the original progress/review controls, label the route and hide unavailable native repair.

## Task 3: Verify real media

- [x] Run new tests, original entry tests, native service/store regression and JavaScript progress tests. Test exact media sample/frame counts with synthetic media, not only mocks. Final scoped run of all `test_long_video*.py` and `test_stream_output.py`: 864 passed; JavaScript progress:19 passed. This verifies code behavior, not generated photographic quality.
- [x] Snapshot remote files and queue state. Sync explicit changed files with hashes; restart only with an empty queue. Backup `h3-independent-shots-6455244d47bf`, Comfy PID141103; all11 runtime hashes and the internal node registration verified. Actual GPU:4080/32GB.
- [x] Generate45 seconds from three independent15-second clips using the original approved office/identity/scale assets and a new dialogue. Keep native probes and enhanced clips. Project `u11-shots-81fa05858f14c2f50d56` produced the45s cloned-voice film; `u11-shots-ce267fa769e0221a86b5` produced16s from two8s no-clone clips. Both are1920×1098/24fps with exact final frame and audio clocks.
- [x] Review actual frames, edit boundaries, audio evidence and stage timing. Actual per-clip and final review receipts document half-second whole-film inspection, frame-by-frame cuts and selected full-size probes. ASR and PCM/AAC checks were performed; direct listening, voice identity and exact lip sync are not certified. Color/delivery diagnostics pass; skin smoothness, paper/gesture changes and delayed framing changes remain. No speed improvement is claimed. See the research record for measured timing and acceptance limits.
- [x] Preserve upstream tensor execution metadata when trimming independent delivery frames. The actual first45s output exposed missing sampling/ColorGuard provenance; the real-output regression failed on that omission, then passed after preserving the source report across the slice. The45s project finished on its frozen initial code. With the queue idle, final runtime backup `h3-independent-shots-913def9ca5eb` was deployed and loaded as PID196366. Both actual no-clone clips confirm completed upstream sampling/decode/ColorGuard stages and original sampling settings in their sidecars, with voice_mode=none and zero audio references.

## Task 4: Deliver

- [x] Update long-video authoring guidance and the H3 skill's existing long-video reference to explain explicit authored cuts and fixed original anchors; retain single-prompt UI and original slot budget. Installed/source copies match; all33 skill-script checks pass with `H3_U11_PROJECT` set to this worktree. This is interface coverage, not a multi-agent behavior assessment.
- [x] Commit only scoped code/tests/docs and push the current branch. Runtime files have been synchronized with backups and hashes: all64 normalized Python runtime files match in the D: installation and E: project copy; the remote11-file runtime inventory and required node registration pass against the actual running service. Local/remote user documentation is also synchronized with preserved originals. The previous11/15 research draft and diagnostic outputs are excluded from the commit. Git publication evidence is recorded separately in the final delivery receipt.
- [x] Prepare the actual45s cloned-voice and16s no-clone films, technical evidence and honest findings for delivery. Preserve diagnostic-only acceptance and remaining texture/continuity limitations. Final remote health check confirms an empty queue and PID196366 still running; do not shut down the instance.
