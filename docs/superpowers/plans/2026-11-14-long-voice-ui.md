# Original Director long-voice UI consistency

> Use subagent-driven-development: TDD implementation, spec review, then quality
> review. Backend baseline `d6a6023` is independently reviewed and installed
> locally; do not modify Python/runtime/media or the running remote experiment.

## Reproduction and scope

`allowedPerformancePresets` unconditionally removes quality two-stage for
`fish_lock`, `restoreSamplingOptions` rejects Fish SelfLift, and rendering calls
both without accounting for the >15s native adapter. Thus the original GUI can
drop a saved long-video quality/SelfLift selection or hide the author recipe
although the backend supports explicit long Fish target preparation. Short Fish
restrictions remain intentional. Current feedback also says automatic
continuation without explaining that each candidate waits for actual review.

## Required behavior

- Use the original duration widget to distinguish finite >15s native long
  requests from short/invalid values. Long Fish must retain/expose the existing
  quality two-stage and SelfLift choices; short Fish still excludes them.
- Reflect existing backend requirements: author recipe still requires REF2VA;
  original long audio_reuse remains unsupported. Do not silently change voice,
  mode, files, seed, duration or profile; do not expand actual backend support.
- Preserve an explicit incompatible author choice and its explanation rather
  than substitute a different recipe. Fix long-Fish compatibility/explanations
  without changing short-mode behavior or adding controls/options.
- Original progress/help plainly states: candidate window -> actual review /
  observed state / accept -> enhancement and next window. It is not unattended
  whole-film candidate generation. Mention the existing eight-static plus one
  runtime-parent allowance in long help; no extra upload control.
- Keep Fish full-soundscape requirements conditional/accurate; UI eligibility
  does not promise voice fidelity, missing backend availability or GPU quality.

## Implementation and checks

Files: `js/minimax_h3_director_plus_v11.js`, `js/long_video.js` and the existing
`tests/js/director_restore.test.mjs` / `tests/js/long_video.test.mjs` harnesses.

1. Write failing actual-render/restore regressions for REF2VA/Fish at24s keeping
   quality/SelfLift/combat eligibility;15s remains restricted. Include invalid
   duration and unchanged audio filenames/role bindings. Record RED.
2. Implement a small shared duration-aware eligibility seam used by preset
   filtering, restoration, method rendering and author compatibility. Keep
   existing control order and serialized slots. No Python or workflow JSON.
3. Verify long help/progress review-gating and quota wording through the real
   render harness, with no extra visible options. Run all three existing JS
   suites and syntax checks; record GREEN.
4. Independent spec then quality review; fix concrete findings and repeat
   relevant checks. Commit/push only scoped JS/tests and this plan. Preserve
   root sidecar and existing source/local changes. Local file sync and remote
   updates are root's responsibility, still gated on other GPU work ending.

Status: implementation and independent spec/quality reviews complete; no open
P1/P2 findings. Root independently verified all149 JS tests, both syntax checks
and `git diff --check`. Python2303 and H3 compiler33 remain the verified backend
baseline; this correction changes no Python or workflow JSON.

Quality review found that the actual four root/document input/change handlers
truncated24.5 to24 before strict eligibility ran. The duration-specific parser
now preserves the invalid numeric value, reports the existing integer-duration
error and serializes24.5 (or null for nonfinite drafts), so global Queue cannot
silently submit the previous24. The existing backend rejects these invalid
values. Seed and dimension parsing are unchanged. Actual editor regressions
recorded RED149:141 passed/8 failed, then GREEN149/149 with zero skips. Final
spec review approved all requirements; quality re-review closed the P2.

Local source installation and remote deployment are separate release actions.
The other film's last GPU enhancement is still running; no remote files or
processes were changed for this UI correction.
