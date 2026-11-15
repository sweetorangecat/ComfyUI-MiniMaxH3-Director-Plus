# Native texture investigation, 2026-11-16

The office90 continuation reached a complete 2160-frame, 90-second,
1920x1080 export, but did **not** pass visual acceptance. Fine diagonal
scratches on paper, laptop, desk and background are visible in the native
samples before VOSR2; enhancement increases their visibility. Skin remains
smoother than the original reference. Queue completion and source-audio
preservation do not establish a visual fix. Keep the instance running.

The rejected final SHA256 is
`f219e8dee421fc4224a1acb4bce4a119164f97fa2c0d3c75d8f94113aec554d8`.
Its review is `completed90/content-review.json` under the diagnostic root
`outputs/diagnostics/long_video_quality_20261115`. The review explicitly
denies delivery and shutdown. All twelve overview pages, all eight complete
seam sheets and the stated original-resolution frame comparisons were
inspected. Audio checks establish preservation, not voice identity or exact
phoneme synchronization. The latter six windows use complete, prepared
same-reference utterances as H3 input; there is no final audio replacement.

## Bounded experiments

The 107-frame controls keep the actual three references, cloned-timbre
reference, seed111330200, hybrid base, DMAD0.8/Mystic0.45/MotionRepair0.25,
Sol-Attn, global-stream context, 12/3 shifts and beta/Euler10 steps (6+4).
They use the same direct upstream Avatar sampler. They are texture probes,
not approved narrative windows or full-length voice tests.

| Probe | Change from short LMS control | Actual result |
| --- | --- | --- |
| LMS | None | Fine diagonal surface scratches are visible. |
| Generic3D FP16 | External lifter checkpoint only | Laptop/paper are somewhat smoother, with slightly softer detail. Skin remains smooth. |
| LMS + rho0.6 | Consistency correction only | Scratches are reduced in inspected laptop/wood regions, with softened detail. Not a fix yet. |
| Generic3D FP16 full first-scene text | Lifter against the separately defined full graph | 345 frames at24fps. All five overview pages and original48/147/293 inspected. No accepted skin-texture improvement. |

The full probe has345 frames, whereas the existing production first window
has328. Its direct duration14 input includes a different physical safety
tail. Do not describe their comparison as a strict single-variable pair.
The actual short probes and their native PNGs avoid that confound.

The short LMS, short FP16, rho0.6 and full FP16 video SHA256 values are,
respectively:

```
2105eb6f3aefb198268e6653fa4fba7b3032972ccbfbded53c119602c3230c68
081ad3f2fe861befb1c10473ab988150ba057cce3663f1732f775a419681ad5d
2b00fb9592d3195698d84e115f5c4e1d5611a0d8690deac37b45150989540efc
eb7fb6e49b555a7bad62181f986b15fff0e365bfee7d46d84401cc286b1af725
```

## Isolated 2D checkpoint probe

Primary source: `LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler`, commit
`f34cadb19f10fee7e36091b690cac39b4eb40573`,
`nodes/minimax_h3_latent_upscaler_2d.py`, SHA256
`8c95ea2c09f474464640570b67b61565ac6d8aa38c579b0662a5e080e26157b8`.
Capture includes the upstream MIT license. The installed old 2D implementation
has a different architecture and cannot be assumed compatible with new tier
weights merely because its node name matches.

Actual canonical model: HF `LBH-123-AI/Minimax_h3_latent_Upscaler`, revision
`85a0187096046badc0a71ab6bd6be182ab06c0f4`,
`minimax_h3_latent_upscaler_2d_conv_v1/minimax_h3_latent_upscaler_2d_conv_v1_fp16.safetensors`.
364308928 bytes, SHA256
`6e0f35692784ba2376704a48a90fb07f90aef76ef21200d5525d6627a54fe909`.
All262 state keys and shapes match the pinned inference code; all floating
weights are finite. It has640 feature channels,12+12 blocks, dual-scale/tier
embedding and temporal kernel5. These are compatibility checks, not quality
evidence. The earlier speculative distributor filename was not confirmed by
the actual primary model API and is not used.

Remote direct download failed; the local download was hashed and transferred
by SSH, then independently hashed on the host. A private diagnostic callback
reuses the unchanged upstream `progressive_sample` with the same global AV
context and masks, replacing only the learned lifter. It uses the author's
normalization, inference architecture and tier1. At this private-probe stage,
production plugin sources, UI, defaults and original project state had not
changed. The private node
was activated with an idle queue and the original process command/environment.

The107-frame2D clip completed and decoded fully with exact24fps timestamps;
both overview pages and original PNG12/48/84 were inspected. Its SHA256 is
`eb8e8fb22b4ddc9a861f01ca386136bcef32268a7f0570a11ca7f490745134b8`.
Native-pixel face/hair, laptop, paper and desk crops indicate less diagonal
microtexture on some surfaces, without an obvious loss of hair-edge detail.
Skin is still smooth. This limited observation does not establish a skin
quality fix, motion regression result or long-film acceptance.

The345-frame2D counterpart to the full FP16 graph completed as
`9fbe3f0f-3a56-4e1a-959a-4d56e5e957ac`. It has exact24fps PTS, fully decodes,
and has SHA256 `a2e88fef9d848476215c136d9886c685c46b8da346737db9efcd2b7554c91777`.
All five overview pages and original48/147/293 were inspected. Broad motion
has no obvious new anomaly at that inspection density; paper microtexture is
somewhat reduced and skin remains smooth. This is not final motion acceptance.

Both107-frame VOSR2 comparisons also completed, using the actual hashed
encoded short native clips as inputs and the same manual-baseline settings.
Their output is1890x1080, preserving the native1344x768 aspect ratio. This
is encoded-texture diagnosis, not an exact unencoded production replay.
Both clips fully decode with exact PTS. Both pairs of overview pages and
original12/48/84 PNGs have now been inspected.

| Encoded-input export | SHA256 | Observation at inspected native pixels |
| --- | --- | --- |
| LMS, `64ab4eb2-81b7-47ab-ae20-d7c11bbfec30` | `a2e1ae398bb529a54265af8f583f6cfb53620a4a88426e794b8bd888bbb15ede` | Dense oblique scratches on paper, laptop, desk and rear screen. |
| Reviewed2D tier1, `e369f33d-0a2e-4432-b60c-b5cdbdd042fe` | `a239480f67666c37ca6977087bc2067b8d74ad5263fafa1854cdfb41c76208b5` | Those surfaces are visibly cleaner; hair edges remain visible, but skin is still smooth. |

## Original-chain integration in progress

`nodes/selflift_lifter.py` uses the pinned MIT inference source and exact
checkpoint through Avatar's explicit progressive-sampler callback. Ordinary
SelfLift selects it only when the compatible API, verified native sparse
attention and global-stream route are present. The author combat recipe
retains its exact LMS lifter. No workflow node or UI option is added.
The AV tensors, masks, schedule and global context stay with the existing
progressive sampler. Checkpoint/source identities are recorded in bounded
execution evidence. Long-film provenance also freezes this optional weight
and both inference/adapter source files.

The integration and declared fractional-repair-mask fix were backed up and
deployed by file hashes, without touching the host's parent Git repository.
Deployment identity: `2d283b29fe1ec61e4d2032fe72eb936ef99e3a7db4e135554c81a7325d84d799`.
ComfyUI restarted from its original command/environment as PID155685 and
passed required-node/file-hash health checks. No private grid or attention
probe was activated. The original90s project is unchanged and unapproved.

The original Director/TwoStage/ColorGuard/U11 production path completed a
107-frame trial, prompt `e998b06e-6a3a-48c0-b9d1-771a004a3161`. The actual
unencoded chain produced1920x1098 at24fps, not the1890x1080 of the earlier
encoded-input comparison. Its SHA256 is
`cdaf9eaaa4d88f92cf5b62f2869b38ad2de05169da6789a685f768b0ddeef8dd`.
Full decode and all video PTS pass. Both overview pages, original output
frames12/48/84 and native48 were inspected. Paper, laptop and desk are
substantially cleaner, with visible hair edges and ribbed clothing. Broad
motion shows no obvious new anomaly at that inspection density; this is
not full motion approval, skin-texture restoration or voice certification.

Measured sampling was198.81s, including the4.93s transition. VOSR2 took
295.99s; its296.12s reconstruction measurement is nested in the310.33s
output measurement. Do not add overlapping timings. This host is an
RTX4080 SUPER with32GB; these are not5090 timing claims.

The original15-second chain is separately testing as
`1aea381c-64a2-4db9-bfe8-d2794d52451b`, with362 physical frames. Sampling
has completed and VOSR2 is running. The nine-window90-second retained
contract passed the running server's static preflight but has not been
resubmitted. Its actual content, later-frame texture and AV continuity
remain to be validated.

Fresh verification:441 related checks passed across the lifter, sampling,
author recipe, native attention, execution evidence, two-stage integration,
native AV/runtime, asset provenance and long-video coordinator suites.
These checks and short-clip observations do not establish full90s acceptance.
Keep the instance running; shutdown is not authorized.
