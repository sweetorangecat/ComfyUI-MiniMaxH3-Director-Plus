# Native target-audio implementation plan

> **For agentic workers:** Use `superpowers:subagent-driven-development` for
> implementation and independent specification/quality review. Complete the
> focused red/green checks before moving to the next dependent stage.

**Goal:** Connect explicit complete audio targets and original-reference Fish
utterances to U11 native AV continuation while retaining its history and
recovery guarantees. Use the accompanying approved design as the contract.

**Architecture:** A focused audio module validates recipes, sample timing and
immutable cache receipts. Ordinary Comfy preparation jobs produce missing
whole sentences. Native graph compilation connects validated target audio to
the existing context node; its AV helper installs the target, restores actual
history, and locks every audio tick. Existing joint-AV/default paths stay inert.

**Tech stack:** Python, existing Comfy queue and native audio VAE, PyTorch CPU
contract tests, standard WAV/file hashing, and the installed Fish bridge.

**Files:** Add focused `nodes/long_video_audio.py` and its tests; integrate only
the necessary seams in `long_video_av.py`, `long_video_runtime.py`,
`long_video_plan.py`, `long_video_graph.py`, `long_video_entry.py`,
`long_video_service.py`, `long_video_assets.py`, `long_video_store.py`, and
`__init__.py`. Corresponding tests stay beside the current long-video suites.
Avoid changing short Director/Fish policy or unrelated UI and media modules.

## 1. Pure audio contract and clocks

- [ ] Write focused failing recipe/clock/cache contract tests and record RED.
- Add `nodes/long_video_audio.py` and focused CPU tests.
- Validate full-window WAV/provenance, original Audio1/Subject/Sx binding,
  soundscape policy, complete utterance deadlines, finite values and geometry.
- Define stable sentence identities and immutable waveform/receipt checks.
- Test invalid recipes and overflow first; then implement the bounded helpers.
- [ ] Run focused audio tests and record GREEN before the native AV seam.

## 2. Target-conditioned AV preparation

- [ ] Write absent-target, protected-history, full-mask and repair regressions.
- Extend `long_video_av.py` and its context node with optional target inputs.
- Install the target before copying actual parent/forward audio, then zero the
  full audio mask. Keep video behavior and the absent-target path unchanged.
- Reject audio-open target repairs; test exact protected values and suffixes.
- [ ] Run audio/AV/runtime tests before graph integration.

## 3. Graph and original voice adapter

- [ ] Add failing graph/entry tests for targets and original Fish adaptation.
- Carry explicit target recipes through long contracts and window requests.
- Add queued target loading/audio VAE encoding and receipt nodes as needed.
- Preserve short Fish restrictions and long graph mux pruning. Retain original
  requested voice/provenance when adapting `fish_lock` inside native execution.
- Test default branches, explicit target graphs, repair, preview and finish.
- [ ] Verify short/default graphs remain unchanged and focused tests pass.

## 4. Preparation queue, cache and fingerprints

- [ ] Write recovery, stop, source-change and cache-reuse regressions first.
- Prepare/cache complete Fish utterances through ordinary queue jobs.
- Publish immutable sentence and full target receipts before creating native
  request hashes; reconcile stop/retry/recovery without duplicate jobs.
- Include source and actual target identity in generation scope; keep sentence
  reuse independent of project/export dimensions, codecs and video retries.
- Test changed sources, lost jobs, concurrent publication and export-only reuse.
- [ ] Verify exact final target identities precede native request publication.

## 5. Integration review and verification

- [ ] Complete independent specification review, then quality review.
- Run focused long-video/audio/Guide tests during implementation.
- Obtain independent specification review, then independent quality review.
- Run the full Python suite after fixes; back up/restore the existing untracked
  PyAV execution report. Run relevant unchanged JS compatibility checks only
  when touched interfaces justify them.
- Commit and push the feature branch after verification. Preserve source/local
  edits during scoped synchronization. Do not deploy, restart or enqueue GPU
  work until the separately running film is finished and the queue is idle.

## Status

- Design and implementation plan recorded; implementation pending.
- No GPU certification or complete 90-second validation claimed.
