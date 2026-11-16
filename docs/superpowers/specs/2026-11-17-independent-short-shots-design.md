# Reuse the original short-video graph for editorial long videos

The user reports progressive warming, darkening and loss of photographic texture in the reviewed 90-second continuation. They explicitly authorize a different script, community research, changes, GPU testing and deployment. The server must remain running. The original <=15-second route must keep its behavior.

## Evidence and alternatives

The current automatic long entry forces REF2VA, quality two-stage, SelfLift and manual VOSR2. Each continuation consumes native AV and, by default, an additional generated parent picture. Native frames from the later 90-second windows already show degradation before SR. This establishes where the defect exists, not a single-factor causal result.

1. Preferred for editorial stories: independently generate bounded clips from the original short-video graph, resetting generative history at explicit authored cuts. Each clip uses the original uploaded references and voice files. Motion continuity is authored and reviewed across a real cut.
2. Preserve the existing native AV continuation for explicitly continuous shots. It has useful motion history but remains subject to observed cumulative drift.
3. Global color matching may help delivery but cannot recover lost skin/hair detail; it is not the primary repair.

Primary-source research is recorded in `docs/research/2026-11-17-long-video-independent-shots.md`. No new third-party node pack or weights are needed to test the preferred route.

## Selection and data flow

Keep the original duration, prompt and reference controls. A globally timed H3 prompt with two or more explicit `[Shot N]` sections selects independent clips when the total exceeds 15 seconds. Pack adjacent authored shots into 4–15-second generation requests without splitting dialogue or inserting unrequested cuts. A shot too long to fit receives actionable prompt guidance; a single continuous shot retains the native adapter.

Compile the six H3 sections into a self-contained local prompt per clip. Retain stable Subject, Picture and Audio identities; map Sx by each independent request's actual speaking order without touching literal dialogue. Preserve the selected model, sampler, performance preset, LoRAs, postprocessing and voice route. Do not add a generated parent reference, context mask or continuation latent. First/last inputs retain their original per-request semantics; this initial automatic path is limited to REF2VA so global hard endpoints are not silently repurposed.

Use the existing disk journal, queue driver and progress API. A separate independent coordinator submits the original graph with an internal short output adapter. It saves exact delivery frames, the original PCM audio, final clip and a receipt, then waits for real content review. Accepted clips assemble in timeline order with bounded memory. The existing stop, resume and retry controls continue to work. A retry of an independent clip retains other clips and their reviews when their inputs are unchanged.

## Safety of existing behavior

No change to <=15-second execution or native continuation sampling. The independent path never calls native context or repair code and never grants a visual review from numeric validation. Reject unsupported output timing (interpolation, pingpong, audio cropping) before GPU work. Do not add reference files or UI options. Persist immutable request/provenance and verify content hashes before reuse and final publication.

## Acceptance

CPU tests cover route selection, short settings retained, local clocks, complete dialogue, fixed image/audio budgets, isolated retry, queue ownership, recovery and exact assembly ranges. Real GPU tests use original references with 3 x 15-second independent clips and both cloned and ordinary generated voice cases. Inspect native and enhanced frames, motion, skin/hair, color, actual edit boundaries, audio and total duration. Report measured time on the actual hardware. Do not claim unlimited quality, perfect identity, or speed improvement from structural tests. Keep the machine running after tests.
