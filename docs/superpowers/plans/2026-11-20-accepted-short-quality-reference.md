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

- [x] Assert the accepted3:2 dimensions and actual0.7 first grid; test landscape/portrait FHD rounding.
- [x] Test ordinary short calls without the internal marker, low-memory/24GB, balanced/standard, combat, non-FHD and non-editorial requests retain their original behavior.
- [x] Test below-budget high-grid requests fail explicitly, and publication rejects changed dimensions, lowres ratio, sparse/global route or lifter tier.
- [x] Exercise real `build_shot_graph`: per-clip marker and existing16-way attention control, original reference/audio/prompt/seed policies preserved; no new keyframe or reference source invented.

### 2. Integrate only the internal high-grid profile

Files: new `nodes/short_quality_reference.py`; modify `nodes/long_video_shots.py`, `nodes/director.py`, `nodes/sampling_experiments.py`, `nodes/color_guard.py`, `nodes/long_video_shot_output.py`, `nodes/execution_report.py`.

- [x] Add pure profile selection/planning/publication-validation helpers with a stable profile ID and explicit scope.
- [x] Use a separate internal Director class type with the identical public input schema. Hidden PROMPT does not reach the actual IS_CHANGED cache path; a distinct class type provides reliable cache identity without visible controls. Director keeps all original public defaults and reference loading.
- [x] Run actual0.7 SelfLift only for this marked high-grid plan; require native global_stream and the existing trained2D lifter. Preserve the old0.6 run and original author recipe elsewhere.
- [x] Keep this native-size profile's pixels through the color/output path and record its actual evidence before publication.
- [x] Run focused tests, then related Director/sampling/long-film regressions. Review any graph cache/cycle, reference or compatibility risk before deployment.

### 3. Review real30s and deploy verified behavior

Evidence: `outputs/diagnostics/accepted_long_reference_20261120/`.

- [x] Freeze the actual user-accepted15s recipe and original asset hashes, then submit a new15s paragraph with fresh dialogue/conditioning/seed, same original picture and voice. Private6197 only; production6006 untouched.
- [x] Compare selected new raw frames, every PTS, rapid head/lip-motion neighbours and scene color; retain ASR/voice evidence with its listening/lip-sync limits. Do not grant acceptance from successful execution alone.
- [x] Assemble accepted15s+new15s; verify every decoded RGB frame and compressed video packet, audio clocks, and source integrity.
- [x] Exercise the automatic profile separately on the actual selected1760×1184 canvas; inspect matching-time raw faces and rapid-motion neighbours. The explicit-keyframe portrait diagnostic boundary remains visible.
- [ ] Back up and deploy only the verified files while services are idle, preserve dirty installed files, verify runtime/registration/database and push the authorized branch. Keep the instance on, clean only obsolete owned test tensors after evidence collection.


## Reviewed implementation evidence

- The first graph-path regression was observed failing before integration. Real Director tests then exposed the existing FHD non16:9 pixel-budget behavior:3:2 is1760×1184, not1620×1080. The implementation preserves selected FHD dimensions and may use a larger canvas than the accepted1632×1088 diagnostic; this requires the separate real automatic15s test.
- Independent review reproduced three issues: hidden-PROMPT cache isolation, normal8:5 rounding silently returning to768p, and publication trusting a mutable guide instead of the image producer. The revised implementation uses a distinct registered internal class, explicit conventional-aspect eligibility/rounding budget, and the actual carried producer report with completed sampling evidence. The added regression set was observed12fail/28pass before these fixes.
- The final implementation passed635 related CPU tests; independent re-review passed409 relevant tests and found no remaining substantive issue. No production deployment or long-film visual acceptance is implied by these results.
- Existing keyframe/reference roles and image/audio counts remain intact. The portrait GPU comparisons explicitly reuse the already approved picture as frame0; this is not an automatic identity-image anchoring rule for other scenes.
- The actual automatic15s test and a cold full-process10s test completed without memory errors. The10s full-body framing did not satisfy the requested distant-face composition and is not accepted as that test. A separate follow-up is required. See `docs/research/2026-11-20-accepted-long-video-clarity.md`.
- Cohort publication/final-delivery checks cover later retained segments when redoing an earlier one, using the coordinator's unchanged seed/fingerprint policy. Final related CPU verification passed640 tests.
