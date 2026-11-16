# H3 long-video drift: primary-source review, 2026-11-17

## Local observation

The reviewed 90-second test has progressive warmth, darker colors and smoother/stylized skin in later native frames. These changes exist before VOSR2. The short 15-second baseline already uses the matched 2D SelfLift lifter; replacing only the lifter did not resolve the long-chain issue. The automatic long entry additionally forces a native continuation recipe and adds the generated parent as a reference.

## Reviewed sources

| Primary source | Relevant finding | Limits |
|---|---|---|
| [MiniMax official H3 repository](https://github.com/MiniMax-AI/MiniMax-H3) and [shot fallback policy](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/3d-animation-short-generator/references/fallback-policy.md) | Treat individual clips as independently reviewable shots; reinforce the original reference anchors when a shot drifts. | The fallback file is a directing workflow, not a guarantee for photographic live-action or a request to switch models here. |
| [tritant H3 Extender](https://github.com/tritant/ComfyUI_MiniMax_H3_Extender), [independent Ref2VA implementation](https://github.com/tritant/ComfyUI_MiniMax_H3_Extender/blob/main/ref2va_independent.py) | Offers a separate noncausal Ref2VA path, without first/last/previous conditioning. Clips and cache entries are independent; Motion Context OFF trades automatic motion handoff for shot-level editing. The README currently lists v3.0.5. | Independent shots still require explicit continuity direction and actual review. Per-clip color controls alter decoded delivery, not latent history. |
| [joeygambino multishot settings](https://huggingface.co/joeygambino/MiniMax-H3-Multishot-Workflow/raw/main/SETTINGS.md) | Reports residual color/detail accumulation even with native latent pinning and fixed first-shot references. Decoded per-shot corrections do not repair the raw feedback state. | Author-reported experiments on their configuration, not measurements of our graph. Never transplant their percentages as local results. |
| [Stubelius Ultimate H3 author test](https://stuubzzz.studio/blog/minimax-h3-long-videos-retake-chunk-ultimate-h3/) | Distinguishes picture+sound continuation, last-frame continuation, and true new-shot reset. Describes partial retakes. Article 2026-10-18, updated 2026-11-01. | Published speed cases use 864x480 Speed, not our 1080p SelfLift+VOSR2 pipeline. Their timings are not comparable. |
| [seitanism Motion Context MultiRef](https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef) | Implements AV extension/bridge with exact H3 frame grids, native context and explicit trim semantics. | Useful for continuous motion; no general drift-free or infinite-quality claim is established. |
| [SatoDive latent continuation](https://github.com/SatoDive/Minimax-H3-Latent-Continuation) | Avoids repeated MP4/pixel reconstruction by retaining native latents. | Its README's absolute zero-degradation marketing claim is not demonstrated by our results; our current path already retains native latents. |

## Decision and falsifiable tests

Prefer independent <=15-second clips for authored multishot stories, using the exact original sampler/LoRA graph and original uploaded image/audio references. Reset history at real cuts, then assemble approved clips. Preserve native AV continuation when the user expressly needs a single continuous long take.

The leading hypothesis is recursive conditioning accumulating appearance errors. Compare the same scene, model, original anchors and short-sampler recipe across independently initialized clips; inspect native pixels before SR. Separate native sampling, SR and assembly evidence. A stable background/skin sample can support visible comparisons but cannot alone prove skin recovery or causal isolation when poses or lighting differ.

The new path bounds generative history, not total duration. It does not guarantee seamless action across cuts, perfect identity, exact reference pores, or zero color differences. Test and record those dimensions separately. Do not add user-facing controls or increase total reference/audio upload counts.

## Actual 45-second cloned-voice diagnostic

Project `u11-shots-81fa05858f14c2f50d56` completed three independent 15-second clips and final assembly. The actual remote GPU identifies as RTX 4080 with 32760 MiB, not the earlier reported 5090. All clips used the original hybrid INT8 model, SelfLift, continuity LoRAs, original three reviewed references and one H3 voice reference. No generated parent, latent history or AV mask was used.

| Clip | Take | Seed | Queue job wall seconds | VOSR2 reconstruction seconds |
|---|---|---:|---:|---:|
| 0–15s | `e7bbac86754540cca4d51461a001376c` | 111731100 | 1730.80 | 1028.90 |
| 15–30s | `e29512e406984eba9acaef11f48489d5` | 111732109 | 1733.69 | 1022.82 |
| 30–45s | `53119f69f2914532be2c746bf23da529` | 111733118 | 1730.79 | 1023.69 |

Each clip delivered 360 frames at 24fps: native 1344×768, enhanced 1920×1098. Final media has 1080 frames, exact PTS, 45 seconds of stereo audio at 32000Hz, and SHA256 `c8d88a7c754397e347c3afebe2e0b54254f544e338e428fea8689bba95c24ae1`. Stage times overlap; do not sum nested output/reconstruction stages. The three queue jobs sum to about 86.6 minutes, excluding preparation and review gaps. This is not a speed improvement claim; VOSR2 accounts for about 17.1 minutes per 15-second clip on this setup.

Actual native and enhanced overviews were reviewed at half-second spacing, with full-size first/middle/last probes. Final overviews cover the whole film; every frame within half a second of both assembly cuts was inspected. The old progressively warm/dark ratchet is not apparent in this sample. Nine source-to-final comparisons have mean absolute RGB error 1.22–1.80/255 and a roughly uniform signed channel change of -1.0 to -1.33/255. Assembly re-encodes once and has small differences; it is not lossless pixel identity.

Skin remains smooth and eyes somewhat stylized. A pencil is absent in the first clip, paper placement and gestures differ at the first edit, and the third clip starts close before moving to the wide view. Independent generation has not solved every prop, movement or photographic-texture defect. No clear white filament, snow or whole-frame ghost is apparent at the reviewed density. ASR candidates contain the intended complete sentences; direct listening, voice identity and exact lip sync were not certified. Diagnostic acceptance must not be presented as full quality acceptance.

Evidence is in `outputs/diagnostics/independent_shots_20261117/office45/`, including per-clip `content-review.json`, final `technical-audit.json` and final `content-review.json`. The first 45-second project ran on the initial adapter before its frame-slice provenance fix. Its execution sidecars therefore lack upstream sampler/ColorGuard records; stage logs and immutable graph/request records remain available. A focused failing regression established that Tensor slicing dropped attached execution metadata, and the adapter now reattaches that source report without modifying source pixels. Deployed runtime backup `h3-independent-shots-913def9ca5eb` and Comfy PID196366 include that fix. The following no-clone project verifies the final adapter on actual media. Keep the server running.

## Actual 16-second no-clone regression on final runtime

Project `u11-shots-ce267fa769e0221a86b5` completed two independently initialized 8-second clips, both actual reviews, and final assembly. The input deliberately retained inactive voice-upload fields; the compiled requests remove those fields. Both actual execution reports record `voice_mode=none`, three original image references and zero audio references, ten-step SelfLift, the matched `trained_2d_tier1` lifter and completed upstream sampling/decode/ColorGuard stages. The frame-slice provenance fix is therefore verified on real outputs, not only synthetic tensors.

| Clip | Take | Seed | Queue job wall seconds | Sampling seconds | VOSR2 reconstruction seconds |
|---|---|---:|---:|---:|---:|
| 0–8s | `4ab812c789d94a10aa1c62cacf84efdf` | 111731200 | 971.36 | 280.44 | 557.72 |
| 8–16s | `6bf37ec13d964886b73675b2fec6ffa8` | 111732209 | 971.52 | 287.86 | 553.04 |

The final file has 384 frames, exact 24fps PTS, 1920×1098 output, 16 seconds and 512000 source PCM samples at 32000Hz stereo. SHA256 is `a108529bd321109e942b91f4b8d237cad3ab38b7cadfd5d60e24294ca0ae24ef`. The two queue jobs sum to 32.38 minutes, excluding review and assembly gaps. This is not a speed improvement. Six source-to-final frame comparisons show mean absolute RGB error 1.37–1.72/255; mean signed channel changes stay about -1.1 to -1.34 rather than increasing with the clip index. AAC versus concatenated original delivery PCM correlation is 0.999989. Assembly re-encodes once and has small differences.

Actual native/enhanced overviews were reviewed every half second, with selected full-size probes. The final review covers the whole film at the same sampling interval, every delivered frame within half a second of the 8-second cut, and full-size frames 191/192. Daylight, pale paper and blue-gray clothing retain a similar neutral palette; small complexion/shot differences remain, but the previous progressively warm/dark trend is not apparent at this review density. Unprompted ASR detects each intended complete sentence and no additional words. Direct listening, ordinary-voice continuity and exact lip sync were not certified.

The first clip moves the paper despite the scripted resting hands. The assembly cut changes head/gaze, hand placement and paper/desk framing. The second clip starts wide and moves to the intended medium framing between its 0.5s and 1.0s samples. Skin remains smooth and eyes stylized. No clear white filament, snow or whole-frame ghost is apparent in reviewed frames; this does not certify unsampled frames. This diagnostic passes the color/delivery scope and does not constitute full photographic, action or prop-continuity approval. Evidence is in `outputs/diagnostics/independent_shots_20261117/no_clone16/`, including actual per-clip `provenance-check.json` and final technical/content reviews.

## Code and installation verification

The scoped Python run (`test_long_video*.py` and `test_stream_output.py`) completed with 864 passed and one warning; the JavaScript progress tests completed with 19 passed. The existing H3 skill interface checks completed with 33 passed. A static Bandit check of the three new runtime modules found no issues or scan errors. These results establish code behavior and recorded interfaces, not photographic quality, exact speech or motion fidelity.

The actual local portable installation is on D:, not the E: source copy. Thirteen runtime files were updated there after reviewing the installed changes, with backup `outputs/diagnostics/independent_shots_20261117/installed-backup-final/`; this includes the two older prompt/planner files required by prior fixes. All 64 Python runtime files in both the D: installation and E: project copy now match the tested worktree after line-ending normalization. The two unchanged older E: files were matched to their exact committed ancestor versions before replacement and retained in `project-tail-backup/`. No local Comfy process was running, so no local runtime restart was performed; the updated code loads at its next start. The two existing H3 skill files match between the installed and source copies. Remote runtime loading is separately verified by the running service and real-media tests, not inferred from these local file checks.
