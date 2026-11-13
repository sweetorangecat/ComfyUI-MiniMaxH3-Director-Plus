# Measured manual VOSR2 tile batching on roomy supported backends

Follow subagent-driven-development after the native attention fix's spec and
quality reviews. One scoped TDD implementation, then independent spec/quality
checks. Keep all active server jobs untouched and retain existing UI choices.

## Actual evidence

Original15s baseline `fb917f7a-235c-4eae-80c7-d45441de3df6` completed in2022s;
sampling594.84665s and VOSR2 reconstruction1260.62872s. VOSR2 used
manual_baseline/image_batch1,362frames,1920x1098, backend_scale1. The installed
Linux compatibility backend declares H3_TILE_BATCH_SUPPORTED=True.

Read-only source review confirmed that engine SHA
`5b29e33faecefcc8b050c98bede9e41252dcae96098c17c7652dd83251fca617`
generates full-frame noise before batching, then batches independent spatial
tiles with unchanged accumulation order. It processes source frames serially;
frame_batch2 is not a further multiplier of simultaneous DiT tile batches.
The claim applies to this actual implementation, not every VOSR backend.

Cached extraction `8b41d7b5-0e32-4fe3-9e02-c7200dc09bba` saved actual colored
native frames72–83; its execution_cached includes sampler21, so no additional
H3 sampling occurred. Saved8bit PNGs were prepared once as the same float32
1920x1098 tensor. They are not claimed to be bit-identical to the original
unquantized cached RGB tensor. Same seed+72,weights,precision,manual settings,
frame_batch2,dino_batch2,scale1 and temporal_cacheFalse; only image_batch varied.

GPU probe order1/4/8/8/4/1 after separate two-frame warmups measured12frames:

| Tile batch | Reconstruction seconds | Sampled device peak |
| --- | --- | --- |
| 1 | 42.5832 / 40.5829 | 8515158016bytes |
| 4 | 31.4995 / 31.4659 | 8515158016bytes |
| 8 | 32.7217 / 32.7165 | 8750039040 / 8548712448bytes |

Actual device reports RTX4080SUPER /32760MiB. CUDA memory sampling50ms includes
the idle Comfy context and is not an exact instantaneous peak guarantee.
Batch4 versus1:PSNR47.6638dB,MAE.00179912,max absolute.51171875;
temporal-difference MAE.00282032. Repeated batch1 was exactly equal; batch4
rounds repeat the same differences. There is no pixel-identity promise.
The worst pixel was on the reflective hair accessory at frame77,x703,y118.
Actual full-size face/hair and worst-pixel comparison images showed no added
stripe in those inspected regions. This is a limited12-frame comparison;
whole15s motion/skin and other input cases remain subject to actual review.

Evidence:
`E:/AI短视频/凡人杂七杂八/南宫婉_现代工作日Vlog/15_文武双全作者配方对照_20261113/04_测试记录/runs/wenwu15-20261114-r1/baseline_h3/cached-native12/vosr-batching-7ac2e50534/RESULT.json`
and its `visuals/` directory; remote raw float tensors remain in
`/root/autodl-tmp/h3-wenwu-vosr-micro-ff55b53e5ac5`.

## Candidate and checks

- Change only manual_baseline's spatial image_batch from1 to measured4 when
  supports_tile_batch is exactlyTrue and measured valid total>=30GiB/free>=24GiB.
  Keep1 otherwise, including low-VRAM/unknown/unsupported environments.
- Preserve existing legacy4/8 policy and experimental_32gb behavior. Do not
  apply8 to manual: it was slower than4 in this probe.
- Keep every other profile/loader/video parameter identical. Add no mode,
  controls, resolution/frame changes, precision changes or temporal caching.
- RED tests for the new guarded manual4 behavior; thresholds, invalid/bool
  memory values, unsupported capability and existing profiles remain covered.
- Run affected profile/video-finish/stream tests with sidecar preservation.
  Independent reviews and root verification precede commit/deployment.
- Then run the original15s story in the same four voice/recipe cases using one
  final functional release. Review actual whole output and measure sampling,
  VOSR2 and total separately; the small probe is not a full-film speed claim.

Status: measured manual4 candidate implemented; RED6 failures/81 passed became
87 profile tests passed. Independent spec and quality reviews approved the
guarded policy with no P1/P2; quality additionally checked5832 profile cases.
Root final combined suite passed2392 Python tests, restoring the protected
sidecar exactly. The production rollout and original full15s story comparison
remain separate from this completed12-frame probe; this plan does not mark
their visual quality or whole-job speed as passed.
