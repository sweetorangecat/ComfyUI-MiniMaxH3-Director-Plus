# U11 Media Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add persistent image thumbnails and playable voice-sample audio controls to the existing U11 Director Plus upload slots without changing workflow layout or routing semantics.

**Architecture:** Extend the existing DOM upload control in one frontend extension file. A small URL builder converts the stored ComfyUI input filename into a same-origin `/view` URL; a media preview builder renders either a linked image or native audio player, and the upload completion path rebuilds that preview from the stored widget value.

**Tech Stack:** ComfyUI frontend extension API, browser DOM APIs, native `<img>` and `<audio>`, pytest source-contract tests, Node.js syntax check, ComfyUI browser verification.

---

### Task 1: Lock The Preview Contract With Failing Tests

**Files:**
- Modify: `tests/test_frontend_source.py`

- [ ] Add a source-contract test that requires `mediaViewUrl`, `URLSearchParams`, `mediaPreview`, an image link with `target = "_blank"`, native audio controls, `preload = "metadata"`, Chinese failure labels, and upload-success preview refresh.
- [ ] Change `SOURCE_PATH` from the v8 frontend filename to the v9 cache-buster filename.
- [ ] Run `python -m pytest tests/test_frontend_source.py -q` and confirm failure because v9/preview behavior does not exist yet.

### Task 2: Implement Persistent Media Previews

**Files:**
- Rename: `js/minimax_h3_director_plus_v8.js` to `js/minimax_h3_director_plus_v9.js`
- Modify: `js/minimax_h3_director_plus_v9.js`

- [ ] Add CSS that changes populated upload slots to a two-row grid, gives linked images a stable 132 px preview area with `object-fit: contain`, and constrains audio players to the slot width.
- [ ] Implement `mediaViewUrl(filename)` with `URLSearchParams({ filename, type: "input" })` and no request for an empty filename.
- [ ] Implement `mediaPreview(filename, accept)` that returns a same-origin linked `<img>` for `image/*` or `<audio controls preload="metadata">` for `audio/*`.
- [ ] Add media error handlers that preserve the widget value and show `图片预览不可用` or `音频预览不可用`.
- [ ] Make `uploadControl` render a preview immediately when its hidden widget already has a filename.
- [ ] Make `uploadFile` replace the existing preview after a successful upload.
- [ ] Run the focused test and `node --check js/minimax_h3_director_plus_v9.js`.

### Task 3: Verify Regression Safety And U11 Structure

**Files:**
- Modify only if generated output changes: `U11-MiniMaxH3-导演台Plus-中文增强版.json`

- [ ] Run `python -m pytest -q` and confirm the entire custom-node suite passes.
- [ ] Rebuild the same U11 path from U10 with `tools/build_u11_workflow.py`; do not create another workflow.
- [ ] Run `tools/validate_workflow.py` and confirm one U11 file, 24 main nodes, 22 main links, one subgraph, and zero visible overlaps.
- [ ] Confirm the live ComfyUI object-info endpoint still exposes `control_after_generate: "seed_mode"` and the workflow retains the Director-to-Settings seed link.

### Task 4: Browser Interaction Verification

**Files:**
- No source changes unless verification exposes a defect.

- [ ] Restart/reload ComfyUI only through the user's existing process; do not start or stop TE.
- [ ] Open a clean disk copy of U11 in a separate test tab so cached unsaved workflow state is not mistaken for the current file.
- [ ] Upload one image and verify the same slot contains a visible linked thumbnail whose `/view` URL contains the uploaded input filename.
- [ ] Enable H3 native voice reference, upload one audio sample, and verify the same slot contains a native player with controls and a playable `/view` source.
- [ ] Switch modes and verify both previews restore from widget values when their slots reappear.
- [ ] Confirm no Director Plus or `h3-director-plus` console errors, then close the test tab and preserve the user's original tab.
