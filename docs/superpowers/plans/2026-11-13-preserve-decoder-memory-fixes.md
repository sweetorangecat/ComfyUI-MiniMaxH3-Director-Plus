# Preserve existing decoder memory fixes before Wenwu rollout

## Problem and scope

A read-only comparison against the active AutoDL plugin found two useful
server patches missing from this branch. Overwriting the complete production
tree would restore large temporary allocations even though the author recipe
itself had passed its unit tests. Preserve the two existing implementations;
do not change frame values, sampling, reference bindings, or workflow topology.

1. `validate_frames_for_reconstruction` must bound finite masks and RGB FP32
   conversion to four frames while reducing the dynamic range over the complete
   movie. Alpha participates in finite checks, but not RGB dynamic range.
2. `MiniMaxH3SafeVAEDecode` must release completed model residency and empty the
   cache before VAE loading/output allocation. Existing output-device and dtype
   hooks must still be restored on errors. Unsupported VAEs fail before cleanup.

The production files match the independently inspected remote snapshots byte
for byte. No server process, active queue, or GPU was changed for this port.

## Verification

- Focused baseline: 24 passed.
- New regressions against the old implementation: 5 failed, 33 passed.
- After preserving the two fixes: 38 passed.
- Independent specification review: approved; 38 focused tests passed.
- Independent code quality review: approved; 39 focused tests passed,
  including the existing execution-report decode test.
- Final Python regression: 2143 passed in 209.14 seconds. The existing
  `pynvml` deprecation warning remains. The pre-existing untracked execution
  report was backed up and restored with its original SHA-256.

The tests exercise real CPU tensor operations over a 257-frame batch, late
nonfinite RGB/alpha values, global range across individually flat chunks,
cleanup order, exact returned values, and hook restoration on failures. They do
not certify GPU peak usage, latency, decoded seams, or author-recipe quality.

## Deployment boundary

Keep the remote versions of these two files untouched. Deploy only the other
15 reviewed production deltas, including the new `wenwu_recipe.py` and
`long_video_prompt.py`, after the user's other running long-video task finishes
and the ordinary Comfy queue is idle. Guard remote file identities before
writing, preserve backups, and verify the restarted process actually loads the
new author recipe before any controlled video comparison.
