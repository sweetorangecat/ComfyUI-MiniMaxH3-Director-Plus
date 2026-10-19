# Balanced attention and Linux VOSR2 validation

> Implementation: execute inline with superpowers:executing-plans. The user approved continuing the diagnosed optimization; preserve the existing workflow and UI.

**Goal:** Reduce measured runtime while preserving the complete 768p eight-step recipe, audio conditioning and output sizing.

**Architecture:** Apply the existing H3-specific memory-efficient SageAttention patch to the matched eight-step model. An unavailable optional accelerator must preserve the successfully loaded LoRA and its schedule. Measure the remote Linux VOSR2 compatibility implementation separately; change allocator housekeeping only after a deterministic output comparison. Do not change reconstruction tiling or add UI controls without quality evidence.

**Evidence:** `190028` sampled for 1568.568 s and reconstructed for 632.743 s. Eight-step attention was PyTorch; the previous 20-step Sage run sampled for 1587.393 s. Different runs are diagnostic evidence, not a controlled benchmark.

- [x] Add failing tests for FL/REF matched adapters retaining the recipe through Sage success and failure, and selecting the H3-specific patch.
- [x] Enable Sage for balanced presets and recipe metadata; preserve matched model on optional-patch failure, report actual backend and keep missing-adapter errors fatal.
- [x] Run balanced, performance, execution-report and complete Python regression tests (excluding the unrelated deleted example fixture): final run 1396 passed, 1 deselected. The excluded example test passed separately against the committed fixture in a temporary directory, without restoring the user's deleted file.
- [x] Wait for the user's remote generation to finish; never cancel it or restart while occupied. Another user job was submitted during the comparisons and is being preserved.
- [x] Run a short same-seed source comparison using the existing API graph, capture raw output and stage timings. Do not create an importable replacement workflow.
- [x] Inspect Linux compatibility housekeeping and independent DiT tile batching; verify unchanged geometry/conditioning against deterministic outputs and measured memory/time before deploying it. CPU equivalence and code review passed. RTX 5090 three-frame comparisons averaged 4.5817 s original versus 3.7958 s batch four (17.2% less reconstruction time), PSNR 49.67 dB. All three corresponding frames inspected without obvious additional blur; this is not a full-video quality acceptance.
- [x] Review diff, record measured limitations, commit/push the scoped changes, update installed local and remote plugin, and verify loaded service registration. Runtime commits: `fac84e1`, `f085273`, `54668a8`. Remote restarted only after the newly submitted user job finished; no held jobs required replay. Local files updated without starting the stopped local service. The existing local workflow hash is unchanged.

Acceptance: all automated recipe invariants pass; GPU comparison shows useful sampling speed improvement without obvious sampled-frame regression. Preserve source material, voice routing, 8 Euler/simple steps, shifts 6/3, no temporal cache, no second pass and adaptive output dimensions. If GPU validation is unavailable or inconclusive, do not present unmeasured quality or speed as established.

## Measured H3 comparison

RTX 5090 32 GB, REF2VA with H3 voice reference, four unchanged large image references using `max`, same prompt/seed/LoRA, 1344×768. The 4-second request aligns to 107 frames at 24 fps. Native output with CRF 16 isolates H3 from reconstruction. The baseline ran first, then the Sage candidate; the latter reused conditioning, so total job wall times are not an attention-only comparison.

| Actual attention implementation | Measured sampling stage |
|---|---:|
| PyTorch baseline | 485.493040 s |
| H3 memory-efficient Sage, head chunks 2 | 211.230830 s |

This sample's sampling speedup is 2.30× (56.5% less time); it is not a measured 15-second end-to-end speedup. Candidate kernel installation was confirmed in the server log. The controlled API graph inserted the existing KJ patch before the extra LoRA loader; it did not edit a saved UI workflow. Its old router metadata still describes the original preset, so it must not be used as evidence of the candidate's actual attention implementation.

First, middle and last frames were inspected: composition and visible texture remain comparable, with small pose/mouth differences. This is not pixel identity, a sharpness improvement, or a complete 15-second voice/lip-sync acceptance test.

The compatibility backend retains its own model cache outside Comfy's model manager. The patch advertises a cleanup hook; the H3 adapter calls it after the complete reconstruction, including failures, without masking the original exception. Tests and code review cover the lifecycle. Actual post-cleanup GPU memory remains part of deployment verification.

During deployment a new 15-second user job selected `ref_quality_native`; its log confirmed the generic `sageattn_qk_int8_pv_fp16_cuda` path. Extend the same H3-specific patch to this reference-quality preset and its Chinese label, retaining 20 steps and no Turbo/cache. Regression tests first reproduced both missed routes. Do not apply the eight-step measured speedup to this 20-step route.

## Installed runtime verification

After restarting the actual service, run the existing API graph through the updated router, without additional test attention nodes. Both 4-second requests align to 107 frames at 24 fps and reconstruct 1344×768 to 1920×1098. Both reports confirm `sage_applied=true`, `attention_backend=h3_sage`, two head chunks, the backend-matched eight-step LoRA, Euler/simple and shifts 6/3, VOSR2 image batch four, and successful backend cleanup. Free VRAM returned to 30.515 GiB, with 0.063 GiB PyTorch allocated, after each video.

| Validation | Sampling | Reconstruction including load/cleanup | Complete API execution |
|---|---:|---:|---:|
| REF2VA, unchanged four large references and H3 voice reference | 216.979 s | 152.659 s | 480.327 s |
| FL2VA, two endpoints and no voice reference | 66.611 s | 151.865 s | 257.073 s |

The second job has different conditioning and is a functional smoke test, not a voice-on/off performance comparison. Neither time predicts 15-second end-to-end runtime. The native reference 20-step routing fix has regression coverage but no new controlled 20-step GPU comparison.

The FL artifact was downloaded and probed: H.264, 1920×1098, 107 frames, 4.458 s video, 4.450 s AAC audio with nonzero waveform. First/middle/last frames were inspected. The post-deployment REF job completed and retained its execution report, but its video and frame files were absent by the time of download, as were several older outputs; the cause is not established. Do not claim a fresh visual/audio acceptance of that missing artifact. The earlier controlled REF native outputs and all three VOSR2 frame comparisons were downloaded and visually inspected.
