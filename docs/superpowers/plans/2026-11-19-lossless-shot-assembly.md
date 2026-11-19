# Preserve Short-Clip Pictures in Long Films Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking. Continue inline in the existing isolated worktree under the user's authorization.

**Goal:** Concatenate compatible original H.264 clips without changing any decoded video frame, while retaining the existing cumulative PCM/AAC clock and automatic compatibility fallback.

**Architecture:** Add a small packet-copy module used only by long-film assembly. Preflight every compressed packet and require matching H.264 decoder parameters, exact target canvas, BT.709 limited video, closed starts and complete 24fps clocks. Preserve the original decode/crop/encode branch for incompatible clips. Audio continues through the existing bounded PCM reader and one AAC encoder. Publish only after the existing full movie validation succeeds.

**Tech Stack:** Python standard library, existing PyAV/libavformat, existing PyTorch/Pytest test fixtures. No new executable, package, model, UI option or GPU work.

---

### Task 1: Capture the actual quality regression

**Files:** Create `tests/test_long_video_packet_copy.py`.

- [x] Add real H.264 fixtures containing a translated fine random texture, deliberate B frames and stereo PCM. Each fixture uses `encode_preview` and `write_delivery_audio`, with global cumulative sample boundaries.
- [x] Test the public `assemble` function with multiple compatible clips. Compare SHA256 of **every decoded RGB frame** and **every compressed video packet**, in order, to the original clips. Check the complete PTS clock, audio length, BT.709 tags, source file hashes and assembly mode.

```python
before_frames = [value for part in parts for value in frame_hashes(part['video'])]
before_packets = [value for part in parts for value in packet_hashes(part['video'])]
report = media.assemble(parts, output, sum(p['frames'] for p in parts), (96, 64))
assert frame_hashes(output) == before_frames
assert packet_hashes(output) == before_packets
assert report['video_assembly_mode'] == 'packet_copy'
assert report['all_pts_exact']
```

- [x] Add real media tests for a required spatial crop, different H.264 reorder parameters and an incorrect source frame count. Crop/decoder mismatches must use the existing encoder. A failed count must leave a pre-existing output untouched and remove the temporary MP4.
- [x] Run the new tests and confirm the frame-preservation assertion fails on the current implementation.

Run: `D:\ComfyUI_windows_portable-G313\python_embeded\python.exe -X utf8 -m pytest tests/test_long_video_packet_copy.py -q`.

### Task 2: Implement the conservative packet path

**Files:** Create `nodes/long_video_packet_copy.py`; modify `nodes/long_video_media.py`.

- [x] Add these three interfaces. `plan_copy` returns `None` on a compatibility mismatch and a dictionary containing the common decoder signature and per-part immutable packet evidence on success. Invalid or missing media continues to fail explicitly.

Interfaces: `plan_copy(parts, target)` validates complete source packets; `create_stream(destination, plan)` copies H.264 stream parameters with `add_stream_from_template`; `mux_part(destination, stream, evidence, frame_offset)` copies payloads and offsets both timestamp clocks. Their behavior and rejection rules are specified in full below.

The complete validation contract is:

1. PyAV must expose `add_stream_from_template`. Every source must have one H.264 video stream, `yuv420p`, exact target size, average rate24 and BT.709 matrix/primaries/transfer plus limited range. Require an AVC configuration record and equal extradata, SAR and profile across parts.
2. Scan all video packets without decoding. Reject absent timestamps, zero durations, nonintegral frame positions, duplicate/out-of-range PTS, incomplete packet coverage or noncontiguous DTS. Require an integer frame tick in each source timebase. The first packet must be a keyframe whose VCL NAL units are IDR (NAL type5). All parts must have the same initial reorder delay.
3. Hash packet length, payload, original PTS/DTS/duration/timebase and packet flags during preflight. During copying, recheck decoder signature, packet count and both payload and timing digests. A source changed after preflight fails safely before publication.
4. For each packet preserve its source timebase, bytes, flags and duration. Add `Fraction(frame_offset, 24) / packet.time_base` to both PTS and DTS, then attach the destination stream and mux. Do not reset DTS to PTS, decode, crop, convert colour or encode the copied video.
5. Keep the existing PCM format/count checks and cumulative sample endpoint arithmetic. Flush only AAC on the packet path; flush both encoders on the fallback path.

- [x] Integrate before temporary output creation, then branch per source. The existing frame loop remains the fallback.

```python
from .long_video_packet_copy import plan_copy, create_stream, mux_part
copy_plan = plan_copy(parts, (width, height))
# Inside the already-existing destination context:
video = create_stream(dest, copy_plan) if copy_plan else _video_stream(dest, width, height, encoding)
# Inside the already-existing part loop:
if copy_plan:
    mux_part(dest, video, copy_plan['parts'][index], cursor)
    cursor += item['frames']
# Retain the existing PCM loop in both branches.
if not copy_plan:
    _mux(dest, video, None)
```

Place the existing video decode loop in the opposite branch, with its complete crop, color conversion and count checks unchanged. Record `video_assembly_mode=packet_copy|decode_reencode` and the actual video encoder (`copy|libx264`) separately from the requested encoding recipe, so a copied stream is not misreported as newly encoded at the requested CRF.

- [x] Run the new tests, then `tests/test_long_video_media.py` and the existing shot coordinator/graph tests. Inspect each failure and keep fallback behavior intact.

### Task 3: Validate with real generated clips and publish

**Files:** Diagnostic evidence under `outputs/diagnostics/portrait_controls_20261119/`; update `docs/research/2026-11-17-h3-generation-trials.md`.

- [x] Use at least two different completed original clips sharing decoder parameters. Run old and new assembly on the same parts; record wall time, output size, full frame hashes, video packet hashes, all PTS and audio sample clocks. Label this as assembly fidelity, not new skin detail.
- [x] Recheck the real15-second higher-grid GPU job independently. Do not infer15s/long picture acceptance from a successful packet test.
- [x] Deploy only the verified assembly files after checking production's queue is idle. Back up originals, verify uploaded hashes, restart only the production service if required and verify registration/database/queue. Do not stop the instance or interfere with private GPU generation.
- [x] Preserve the unrelated handoff document, commit only this change and its verified research notes, and push the existing authorized branch. Finish with the measured result and remaining picture-quality limits.


### Verification recorded before deployment

- TDD on the original assembler: the initial four tests produced 3 failures / 1 pass. The frame equality assertion exposed a real extra encoding generation.
- Independent review identified source timing changes after preflight and concealed corrupt IDR frames. Both regressions were reproduced before their fixes. The implementation now fingerprints original timing/flags and rejects corrupt decoded frames before publication. The reviewer reported no remaining substantive findings.
- The eight relevant media/shot/service/graph suites passed: **181 tests**.
- Actual two distinct15-second clips,1920×1098: old assembly37.3881s, new9.8041s including full validation. New output preserved all720 decoded RGB frames and all720 compressed video packets exactly; AAC payloads matched the old assembler, and source hashes remained unchanged. This establishes assembly fidelity, not acceptance of the source skin quality.
- Actual remote PyAV17.1.0 check: all240 RGB frames and240 video packets of two distinct5-second clips matched, exact audio/video clocks passed, sources unchanged.
- The real15-second1088p experiment completed separately and still has a smoothing/beautification limit; it was not promoted to a production default.


### Deployment and final45s check

- Feature commit `5418c3528f7e63c4cf84c6df3b2216c4a3a76d9c` pushed to `codex/u11-native-av-long-video`.
- Local installed plugin received only this feature through a tracked cherry-pick, `c4ccac1e061cbc28deb4732d78601ed061c2751c`. Hashes of all38 pre-existing dirty files remained unchanged.18 relevant media tests passed in that installed checkout. Local Comfy was not listening on8188/8189; files take effect on its next normal start.
- Remote6006 was idle; backed up the original module, atomically installed the two verified files and restarted the same command/cwd/environment. Director registration, queue and read-only database quick-check passed. All61 runtime Python files matched the current worktree on both installed machines. The completed owned6197 service was stopped, and the AutoDL instance stayed running.
- Actual three distinct15-second clips assembled into45s,1920×1098. All1080 decoded RGB frames and1080 compressed video packets matched their source segments exactly, all clocks passed, source hashes remained unchanged. Assembly plus full validation took14.8842s. This is a new assembly of retained sources, not new GPU generation or proof that those sources meet the skin/action target.
- Removed19 explicit obsolete private-test tensors,992,091,826bytes; actual free-space increase992,133,120bytes.174 delivered evidence files were re-hashed before deletion,3104 model symlinks checked, no active open handles depended on the targets. Kept all media, inputs, model assets and current15s conditioning/latent. Free space afterward29,809,733,632bytes.
