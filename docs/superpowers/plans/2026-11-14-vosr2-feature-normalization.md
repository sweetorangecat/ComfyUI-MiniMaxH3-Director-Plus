# VOSR2 Linux DINO feature normalization repair

The user rejected the two-person live-action comparison. Keep the original workflow and distinguish generation defects from finishing defects. Short portrait controls can preserve photographic detail, but the current finishing path smooths that detail further.

## Confirmed defect and scope

The inspected deployed engine SHA256 is `5b29e33faecefcc8b050c98bede9e41252dcae96098c17c7652dd83251fca617`. Its `_venc_features` multiplies normalized RGB by255 immediately before bicubic resize and clamp01. Upstream `cswry/VOSR/inference_vosr_onestep.py` passes raw255 through `preprocess_raw_image`, which first divides by255. The compatibility implementation omitted that division. Intermediate layer17 does not require the final DINO norm; leave that independent contract unchanged.

Repair only the inspected preprocessing block with the existing opt-in maintenance tool. Preserve tile batching, cleanup, weights, seeds, target size, frame count and lack of temporal feature reuse. Recognize both the original scalar implementation and the previously batched implementation; reject unknown preprocessing before writing either file. Preserve exact original backups and repeated-application behavior. No UI or workflow JSON changes.

## Verification

- First reproduced two incorrect RGB/ImageNet results against an independent reference and the missing unknown-layout refusal: three failures, ten existing passes.
- Corrected the minimal preprocessing block. Expanded the maintenance tests to cover a previously batched engine, read-only inspection, exact backup retention, repeated application and all-file refusal before writes.
- Confirm the complete captured real source patches and compiles, with tile batching retained and a second application unchanged.
- Verification completed:116 patch/profile/finish tests passed; full Python regression2397 passed with only the existing pynvml deprecation warning. The captured real engine patches and compiles, retaining tile batching and idempotence. Inspect the change and preserve unrelated files before committing and pushing.
- Deploy the verified patch while the queue is idle and restart the existing Comfy process without changing its launch options. Verify loaded code and an actual original-entry output before calling deployment complete.

## GPU evidence and acceptance boundary

An in-memory experiment processed the same three saved lossless RGB8 native portrait samples at1920×1098 using the same weights, settings and absolute-frame seeds. Only normalization changed. Total elapsed including cold loading was46.36s corrected versus45.57s original; this is not a speed benchmark. Viewed corrected frames0/48 retain more visible skin surface variation than the original frame0. Residual smoothing and etched hair remain. These are independent sampled frames, not full-film temporal acceptance, and do not resolve the original generation-stage full-body failure.

Continue the controlled full-body prefix-grid comparison and retest the original15-second action with and without reference voice before claiming overall quality recovery. Do not queue the24-second continuation while that gate fails.
