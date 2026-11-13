# Native attention compatibility for SelfLift global streaming

Use subagent-driven-development: one TDD implementer, independent spec review,
then independent quality review. Root handles real-runtime probes, release and
GPU comparison. Do not modify the active remote sampler or restart its server.

## Observed failure and cause

The actual original-U11 baseline15s job
`fb917f7a-235c-4eae-80c7-d45441de3df6` uses functional release9c26513.
Its guide claims Sol-Attn/global_stream, but installed Avatar logs
`attention override detected (make_override.<locals>.override)` and
`native sparse blocks=0/50`, disabling head/GPU streaming and using full-sequence
compatibility attention. Observed VRAM31905MiB/32760MiB and low-stage42–53s/step
are measurements of this run, not universal performance estimates.

`performance._apply_sol_attention` loads the external
ComfyUI-SolAttn_triton.SolAttnPatch generic optimized_attention_override.
Installed selflift-Avatar/h3_global_tiling explicitly adapts ComfyUI native
BlockSparseAttention H3 block patches and its matching native override. It
rejects/forces full-sequence fallback for the generic external override.
`run_selflift` currently trusts guide sol_attn_applied instead of checking this
actual model compatibility, so author/long strict requirements also have a
false-positive gap.

Read-only source capture:
`C:/Users/sugar/AppData/Local/Temp/h3-native-attention-live-source-kssd8wid`.
The running server actually exposes `BlockSparseAttention`; its definition is
in CAPTURE.json. Comfy core source SHA f205faa57e3d348ead3f60e59caed24c31ebf2976d8d50e9eaaa2e55b36ef067;
Avatar detector SHA2470fdb5b35b317e100a30537afc1c941cbe50d5f1e4c0ffd0550c48bccc3790.
Official API source: https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_sparse_attention.py.

## Required outcome

- Prefer the real ComfyUI native BlockSparseAttention sol-attn provider for
  SelfLift and native long windows; do not add UI choices or alter the standard
  sampler's existing attention policy. Native availability must not depend on
  the external legacy Sol plugin being installed.
- Preserve current quality contract: tau1.3, sparse window.2–1, min_tokens12288,
  extra_tokens256 where actually supported, exact_kv_and_rows conditioning
  sinks, all existing learned lifters/LoRA, schedule, dimensions, seed and AV
  masks. Do not introduce SLA/VSA, Morton or Sage/FP8 substitutions.
- Verify the resulting actual model has compatible native patches covering
  every H3 block with correct block identities/indices and a matching native
  attention override. A guide boolean or a callable name alone is not proof.
  Preserve/reject unknown upstream attention modifications explicitly; do not
  erase unrelated semantic patches to force success.
- Author recipe and native long windows fail before low-stage sampling when
  actual native/global-stream compatibility is unavailable. Older short
  compatibility routes may remain usable with accurate provider/fallback
  evidence; do not claim native stream from a legacy generic override.
- Reports distinguish requested mode, verified provider/block coverage and
  actual compatible route. Clear stale evidence after fallback/restoration.
  Existing non-SelfLift standard/low-VRAM routes remain unchanged.

## Implementation and verification

1. Reproduce the legacy override false positive with meaningful RED tests;
   cover native provider presence without legacy plugin, all-block coverage,
   partial/stale/wrong-block closures and unknown overrides, strict author/long
   preflight and old short compatibility. Preserve existing standard behavior.
2. Implement a narrow lazy provider/compatibility helper and SelfLift-only
   routing seam. Candidate files: new `nodes/native_attention.py`,
   `nodes/performance.py`, `nodes/sampling_experiments.py`,
   `nodes/execution_report.py`, the small report whitelists in
   `nodes/two_stage.py` / `nodes/stream_output.py` and focused tests. No workflow JSON,
   global environment, AV timeline or UI changes.
3. Run focused checks. Root additionally validates against actual installed
   Comfy core and captured Avatar detector on CPU, then the full Python suite
   with the protected root sidecar backed up/restored. Run JS only if affected.
4. Independent spec then quality reviews; fix findings and repeat appropriate
   checks. Root commits/pushes scoped files and updates local installation.
5. Wait for the currently running baseline to finish before any remote source
   write/restart. Preserve its result as old-route evidence. Deploy the verified
   fix, then use identical story/assets/seed for a fair native-baseline versus
   author comparison, with and without H3 voice reference. Check actual native
   block/global-stream logs and material artifacts, not only flags.

Status: implementation and independent spec/quality reviews complete, no open
P1/P2. Spec review found deferred child-object replacements could invalidate
inspected blocks; the followup rejects block containers, indexed blocks,
block-forward subtrees and complete attention subtrees while preserving them.
Exact whole diffusion-model replacements and existing MLP chunk hooks remain
supported. Focused522 tests passed after18 RED failures reproduced that gap.

Root final combined suite passed2392 Python tests; protected PyAV sidecar SHA
was restored exactly. H3 compiler/export33 checks passed against this worktree;
JS sources are unchanged from the separately verified149-test frontend release.
The production helper passed actual Comfy native factory/actual Avatar CPU
detector checks for50/50 blocks, unknown/deferred hook rejection and extra256.
A separate tiny real Comfy Kitchen sol-attn GPU kernel smoke passed on SM8.9;
neither test loads H3 model weights or approves a generated film. Original
four-case GPU comparison and the new native entry remain rollout validations.
