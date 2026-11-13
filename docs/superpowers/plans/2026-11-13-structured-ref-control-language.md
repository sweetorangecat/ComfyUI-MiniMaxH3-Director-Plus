# Structured REF Control Language Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development for implementation, independent spec and quality review. Preserve running jobs; no GPU or deployment during preparation.

**Goal:** Prevent the helper from appending Chinese control prose to an English structured REF2VA prompt, while preserving dialogue, voice references and the no-visible-writing constraint.

**Architecture:** Keep the existing original Director and prompt-helper boundaries. Default visual policy and automatically generated structured audio notes use clear English control descriptions; supplied user text and transcripts remain verbatim, numbered waveform bindings remain stable. No new workflow, sampling change or audio postprocessing.

**Tech Stack:** Python, pytest, existing H3 prompt/Director integration.

## Evidence and scope

The concurrent single-scene 90s trial rejected `window02-take01-forward01`: its independent ASR includes the helper's literal default sentence beginning `除非用户明确要求画面文字`. Its submitted English six-section prompt does not contain that sentence. Source artifacts are under `E:/AIVideo/ComfyUI-MiniMaxH3-Director-Plus/outputs/diagnostics/continuous90_1080_20261113/nangongwan_office/`. This establishes an injected-text risk; it does not prove that all unwanted speech has this cause or that an English change has passed GPU review.

The user authorized author-recipe plot preparation and later tests after the other task ends. This prompt defect would contaminate those tests, so fix it offline first. Do not edit the concurrent experiment, saved prompt, latent masks, assets, timing or service. Do not mute or trim unwanted generated speech.

## Task 1: Reproduce and fix at the actual helper boundary

**Files:** `nodes/prompting.py`, `tests/test_prompting.py`; add a focused Director integration assertion only if the existing test harness can capture its actual resolved prompt without new dependencies.

- [x] Read the ASR and submitted English prompt, identify which text comes from the helper, and save only concise non-secret observations.
- [x] Write and run a failing test through `build_reference_prompt` and `ensure_visual_text_policy`: default English structured REF prompt plus Chinese text only inside `<d>` must not acquire the Chinese visual-policy sentence. Assert the dialogue is byte-for-byte unchanged and the English no-subtitle semantics remain present.
- [x] Test missing structured Audio binding and optional transcript notes: automatic explanatory prose is English, provided subject label/transcript and Audio numbering remain exact. A Chinese literal transcript is data and must not be silently translated or deleted.
- [x] Test supplied custom `visual_text_policy`, repeated helper calls and Chinese user-authored prompts remain preserved. No automatic rewrite of user-owned prose; no audio index/order changes.
- [x] Implement the narrow default/control-language change. Existing non-structured prompt semantics and voice-mode selection must remain intact.
- [x] Run the affected pytest set. No Git operations or production deployment by implementer; root performs independent review and release.

Example new contract:

```python
resolved = build_reference_prompt('REF2VA', english_six_sections, 15,
    extra_reference_count=2, has_audio=True, audio_count=1)
assert '<d>先松手。</d>' in resolved
assert '除非用户明确要求画面文字' not in resolved
assert 'subtitles' in resolved.lower()
assert ensure_visual_text_policy(resolved) == resolved
```

## Task 2: Verify and integrate for the next test

- [x] Independent spec review then quality review; fix concrete P1/P2 findings.
- [ ] Capture the full final resolved prompts for both H3-reference and no-reference benchmark cases, not only the user input. Compare the dialogue and binding manifests against the single frozen story contract.
- [x] Run relevant regressions and scoped diff checks. Preserve the untracked PyAV sidecar and unrelated local edits.
- [ ] Commit and push scoped source changes. Synchronize only idle local files with backup; remote update/restart/GPU tests wait for the other task's actual completion and an idle queue.

## Status

Source fix and independent reviews complete; benchmark asset preparation, remote deployment and GPU speech verification pending.

- RED before production: 8 failed, 1 passed in14.28s at helper and actual Director boundaries.
- Affected helper/Director set:169 passed; full suite:2129 passed in179.65s, one existing pynvml deprecation warning.
- Spec review:13 focused cases passed, both raw submitted-graph voice replays preserved authored dialogue and one policy; ten legacy cases matched base except intended default policy change. Quality review found no concrete P1/P2.
- No automatic rewrite of supplied Chinese dialogue or custom policy; no waveform, Audio slot, sampler or timing changes.
- The actual earlier failed prompt replays with zero Chinese control characters outside its authored dialogue. This does not prove English control prose can never be spoken; GPU validation remains pending.
- Other thread remains active; no remote updates, restarts or GPU submissions.
- Original untracked PyAV sidecar restored after the full suite.
