# Director duration entry implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development for the independent prompt/skill task and review; root owns the entry and workflow integration.

**Goal:** Use the original Director's duration, prompt, images and voice controls for both short and long videos. A duration over 15 seconds selects the existing native AV continuation automatically.

**Architecture:** The U11 output inspects the actual prompt graph before evaluating lazy sampling inputs. Short videos delegate unchanged. Long videos compile the original globally timed H3 prompt into the internal contract, apply the supported long route to a detached graph, and run the existing preflight automatically. The Director hosts progress and existing real-media review; no source-media approval is fabricated and no new asset slots are added.

**User direction:** This design follows the user's explicit instructions: one original Director, automatic routing above 15 seconds, no required recommendation/file/preflight buttons, and H3 skill guidance for a single full-length timeline. The original reference/audio capacity stays unchanged. Storyboard panel count may grow with duration/complexity within legibility and the same upload budget.

**Tech Stack:** Python, ComfyUI lazy nodes, vanilla JavaScript extension, existing disk-backed AV runtime.

## Task 1: Compile the original prompt and teach the existing H3 skill

Files: new `nodes/long_video_prompt.py`, new `tests/test_long_video_prompt.py`; update the existing installed `h3-video-prompt-enhancer/SKILL.md` and its long-video reference in place, with backups outside the skill.

- [x] Define `contract_from_prompt(prompt, target_seconds, settings=None)` returning the existing schema-v1 contract, validated by `validate_contract`, with no external file requirement.
- [x] Tests first: 30/90/120-second single-language six-section prompt, global timestamp ranges (`[0-8s]`, `[00:08-00:16]`, `[16-30秒]`), dialogue verbatim and only once, original Sx/Subject/Audio bindings, no supplied voice, inherited image count, malformed/overlapping/missing ranges produce actionable prompt errors rather than JSON instructions.
- [x] Timed phase prose stays with its actual event. Never copy the entire detailed description into every window or put dialogue in global/static sections. Plain untimed continuous-scene prose may be represented as continuous action; complex unstated event times are not invented. Explicit dialogue lacking usable timing must get a concise prompt-format error rather than be repeated across windows.
- [x] No fabricated dimensions, uploads, asset approvals, observations or added image/audio slots. Use a reference-relative 1:1 scale constraint only as an instruction, not a measured fact.
- [x] Preserve the legacy JSON exporter as an internal/advanced compatibility tool; ordinary H3 output is one full-length prompt pasted into the original Director, not a second user workflow.
- [x] Skill examples teach global film time, phases/contacts/results, complete timed utterances, continued action rather than replay, the existing image/audio limits, and duration/complexity-driven storyboard panels with legibility checks. Read skill before changing it; back up touched files; verify a real example with the actual compiler.

## Task 2: Automatic backend routing in the original output

Files: `nodes/long_video_entry.py`, `nodes/director.py`, `tests/test_long_video_entry.py`.

- [x] Red tests: duration 15 delegates to short unchanged; 16/90 selects long without a config node; lazy status requests no images/guide/audio for long; internal windows do not recurse; original graph, voice, seeds, models, media, output options stay intact.
- [x] Extend the Director duration widget beyond 15 without allowing an oversized duration into a physical H3 sampler. The physical request normalizer remains capped. A direct invalid old output must explain the required original U11 output upgrade.
- [x] Resolve long inputs only from the actual connected output ancestry, fail on ambiguous directors without evaluating models; retain legacy linked short-duration behavior, while automatic long totals use the original editable duration. Retain old explicit config support.
- [x] Apply REF2VA/SelfLift/quality-two-stage/VOSR2 manual baseline to the detached generation graph. Preserve voice choice, media and actual selected model assets; unsupported Fish/reused-audio or unsupported hard endpoints report a clear supported-route limitation.
- [x] Build stable automatic project identity from original recipe/seed/duration; run ordinary preflight before asset/model work; launch and expose progress using the existing UI event.

## Task 3: Put progress in the Director and retire the extra default panel

Files: `js/minimax_h3_director_plus_v11.js`, `js/long_video.js`, `tools/build_u11_workflow.py`, original U11 example and API template, relevant existing Python/JS tests.

- [x] Duration input has no 15-second maximum and is never silently clamped by low-memory UI rendering. It states that >15 uses automatic continuation; short route hints remain accurate.
- [x] Ordinary Director UI has no recommendation button, file picker or manual preflight. Progress/review lives in the same Director, preserves video playback while polling, and survives saved canvas reload using the actual returned project identity.
- [x] Legacy explicit config nodes remain loadable for old graphs, with advanced controls collapsed. Default original U11/API template has only the original Director and upgraded output; no new standalone workflow or extra default long config node.
- [x] In-place saved-workflow migration preserves every unrelated node, existing media, widget choice, seed and link. Remove only the generated default-off config node; preserve active expert configs and their data.

## Task 4: Verify, review, commit, push and deploy

- [ ] Run targeted new/affected Python and JavaScript tests, then the full suites once on the frozen result.
- [ ] Independent spec review and quality review; fix material issues before delivery.
- [ ] Commit explicit changed paths on the existing `codex/u11-native-av-long-video` branch and push it; do not reset or publish unrelated source main changes.
- [ ] Back up and update source, actual local install and idle AutoDL over the latest authorized SSH connection. Preserve custom upscale edits, deleted examples and source research/media.
- [ ] Restart remote only with an empty queue. Verify loaded duration schema, original saved workflow, JS and endpoints over SSH/HTTP. Local isolated startup smoke must be stopped afterward. Record whether a real GPU long-video generation was performed; loading tests do not certify quality.


## Frozen verification and integration status

Independent spec and quality reviews approved after fixing runtime cache NaN serialization, nested Director identity, connected expert configuration preservation, optional endpoint packing, explicit storyboard roles/numbering, linked image crop hints, and integrated button event propagation. Final frozen regression: 2073 Python tests passed (one existing pynvml deprecation warning); 104 JavaScript tests passed. No GPU generation was submitted. The original H3 skill was updated in place and its 30-second, twelve-panel example passed the actual compiler and CPU preflight. Original skill backups and verification evidence are outside the project under the task temporary directories.

Source/installed-plugin integration proceeds with scoped backups and in-place original U11 migration. AutoDL SSH currently closes before its protocol banner and the instance is reported stopped; this revision must not be described as remotely deployed. Production GPU and final dynamic quality acceptance remain separate from these code/input checks.
