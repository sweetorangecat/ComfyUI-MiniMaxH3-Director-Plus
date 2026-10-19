# Balanced attention and Linux VOSR2 validation

> Implementation: execute inline with superpowers:executing-plans. The user approved continuing the diagnosed optimization; preserve the existing workflow and UI.

**Goal:** Reduce measured runtime while preserving the complete 768p eight-step recipe, audio conditioning and output sizing.

**Architecture:** Apply the existing H3-specific memory-efficient SageAttention patch to the matched eight-step model. An unavailable optional accelerator must preserve the successfully loaded LoRA and its schedule. Measure the remote Linux VOSR2 compatibility implementation separately; change allocator housekeeping only after a deterministic output comparison. Do not change reconstruction tiling or add UI controls without quality evidence.

**Evidence:** `190028` sampled for 1568.568 s and reconstructed for 632.743 s. Eight-step attention was PyTorch; the previous 20-step Sage run sampled for 1587.393 s. Different runs are diagnostic evidence, not a controlled benchmark.

- [x] Add failing tests for FL/REF matched adapters retaining the recipe through Sage success and failure, and selecting the H3-specific patch.
- [x] Enable Sage for balanced presets and recipe metadata; preserve matched model on optional-patch failure, report actual backend and keep missing-adapter errors fatal.
- [x] Run balanced, performance, execution-report and complete Python regression tests (excluding the unrelated deleted example fixture): 1393 passed, 1 deselected. The later reference-size wording correction also passed 138 director tests.
- [x] Wait for the user's remote generation to finish; never cancel it or restart while occupied. Another user job was submitted during the comparisons and is being preserved.
- [x] Run a short same-seed source comparison using the existing API graph, capture raw output and stage timings. Do not create an importable replacement workflow.
- [x] Inspect Linux compatibility housekeeping and independent DiT tile batching; verify unchanged geometry/conditioning against deterministic outputs and measured memory/time before deploying it. CPU equivalence and code review passed. RTX 5090 three-frame comparisons averaged 4.5817 s original versus 3.7958 s batch four (17.2% less reconstruction time), PSNR 49.67 dB. All three corresponding frames inspected without obvious additional blur; this is not a full-video quality acceptance.
- [ ] Review diff, record measured limitations, commit/push the scoped changes, update installed local and remote plugin, and verify loaded service registration.

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
