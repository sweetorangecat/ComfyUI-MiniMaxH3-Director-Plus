# Keep the Accepted Short-Clip Clarity in Long Films

> **For agentic workers:** Use superpowers:executing-plans inline in the existing worktree. The user accepted the actual15-second picture and authorized adapting the original workflow. Preserve the unrelated handoff document and all existing low-memory/ordinary short routes.

**Goal:** Reuse the accepted1088p SelfLift geometry for eligible32GB editorial long-film clips, freeze the recipe across independent segments, and verify a genuinely new second15-second segment before claiming long-film picture acceptance.

**Architecture:** Keep the original Director, reference slots and authored4–15-second clip planner. Annotate only internal independent clip requests, and let the Director select the measured high grid for FHD SelfLift quality requests on32GB-class hardware. Keep public short/low-memory/combat settings unchanged. Record and check actual sampling evidence at publication; retain packet-preserving assembly. Original scene references remain soft references unless an actual explicit keyframe was supplied—do not turn a character portrait into every scene's hard opening frame.

**Tech Stack:** Existing Python/Comfy nodes, PyTorch and PyAV. No new UI option, weight download or alternate user workflow.

## Contract and limits

- Accepted evidence: `portrait15-clone-selflift-1088`, SHA256 `6b64af11606c904d1bd51cce8e12cb568a81ba63867cc48b1a1b31696cd552f3`,1632×1088,15s/360frames, original picture and voice, low grid1152×768, lowres_scale0.7, Euler/beta10(6+4), native Sol/global_stream/rho0, trained2D tier3. The accepted diagnostic also used a deliberately chosen original-image frame0 guide.
- Automatic high geometry is limited to independent REF2VA / quality_two_stage / SelfLift / continuity /1080p FHD, with28GB+ total VRAM and sufficient free memory. Smaller GPUs retain their existing plan; a qualified high-grid request must not silently fall back when available memory or its native sampler/lifter is insufficient.
- No change to the ordinary≤15s entry, low-memory presets, combat recipe, picture/audio counts, uploaded identities, authored cuts, dialogue, native AV or seed policy.
- High-grid geometry does not certify arbitrary action, framing or source assets. Do not automatically hard-anchor an identity image. Long-film content remains subject to actual frame/motion/color/voice review.
- Use the accepted raw native color/texture for this profile and bypass unnecessary SR once native dimensions meet FHD. Never feed an enhanced/generated tail into the next independent clip.

## Tasks

### 1. Establish red-capable recipe tests

Files: create `tests/test_short_quality_reference.py`; extend `tests/test_long_video_shots.py` and the existing sampling/color/Director tests.

- [ ] Assert the accepted3:2 dimensions and actual0.7 first grid; test landscape/portrait FHD rounding.
- [ ] Test ordinary short calls without the internal marker, low-memory/24GB, balanced/standard, combat, non-FHD and non-editorial requests retain their original behavior.
- [ ] Test below-budget high-grid requests fail explicitly, and publication rejects changed dimensions, lowres ratio, sparse/global route or lifter tier.
- [ ] Exercise real `build_shot_graph`: per-clip marker and existing16-way attention control, original reference/audio/prompt/seed policies preserved; no new keyframe or reference source invented.

### 2. Integrate only the internal high-grid profile

Files: new `nodes/short_quality_reference.py`; modify `nodes/long_video_shots.py`, `nodes/director.py`, `nodes/sampling_experiments.py`, `nodes/color_guard.py`, `nodes/long_video_shot_output.py`, `nodes/execution_report.py`.

- [ ] Add pure profile selection/planning/publication-validation helpers with a stable profile ID and explicit scope.
- [ ] Pass the internal marker using existing prompt metadata and hidden Comfy inputs; no visible controls. Director keeps all original public defaults and reference loading.
- [ ] Run actual0.7 SelfLift only for this marked high-grid plan; require native global_stream and the existing trained2D lifter. Preserve the old0.6 run and original author recipe elsewhere.
- [ ] Keep this native-size profile's pixels through the color/output path and record its actual evidence before publication.
- [ ] Run focused tests, then related Director/sampling/long-film regressions. Review any graph cache/cycle, reference or compatibility risk before deployment.

### 3. Review real30s and deploy verified behavior

Evidence: `outputs/diagnostics/accepted_long_reference_20261120/`.

- [x] Freeze the actual user-accepted15s recipe and original asset hashes, then submit a new15s paragraph with fresh dialogue/conditioning/seed, same original picture and voice. Private6197 only; production6006 untouched.
- [ ] Compare new raw frames, every PTS, rapid head/lip motion, foreground/background color and actual voice. Do not grant acceptance from successful execution alone.
- [ ] Assemble accepted15s+new15s; verify every decoded RGB frame and compressed video packet, audio clocks, and source integrity.
- [ ] Exercise the automatic production profile separately; keep the explicit-keyframe diagnostic boundary visible. Promote only supported evidence, or retain the old default if it fails.
- [ ] Back up and deploy only the verified files while services are idle, preserve dirty installed files, verify runtime/registration/database and push the authorized branch. Keep the instance on, clean only obsolete owned test tensors after evidence collection.
