# H3 Latent Upscaler API Compatibility Design

## Objective

Repair the U11 trained two-stage sampling route so it can call installed versions of `MinimaxH3LatentUpscaler3D` without failing after the first sampling pass. The repair must support the current ComfyUI v3 node contract and retain compatibility with older upscaler releases.

This is a focused P0 repair. Broader 1080p, 2K, and 4K quality changes will be researched and designed separately after this fix is verified and pushed.

## Confirmed Failure

The installed upscaler exposes a ComfyUI v3 classmethod with this effective contract:

```text
execute(latent, model_name, mode, align,
        enable_temporal_chunking, force_unload,
        device, precision)
```

U11 currently selects the node's `EXECUTE_NORMALIZED` wrapper, inspects a different callable, and only recognizes the older `enable_chunking` or `keep_proportion` fields. On the current node version, the wrapper receives no usable normalized arguments and eventually reports that all required `execute()` arguments are missing.

Because the failure happens after the first H3 sampling pass, it wastes generation time before aborting.

## Considered Approaches

### Selected: signature-driven compatibility adapter

Resolve the real implementation callable, inspect its declared parameters, build only the fields supported by that version, and call the implementation directly when the registered class uses a ComfyUI v3 normalized wrapper.

This preserves compatibility without pinning dependencies and keeps upstream model loading, temporal chunking, and unloading behavior in the upstream node.

### Rejected: pin an older upscaler release

Pinning would avoid a code change locally but would fail again when another installation updates the custom node. It would also prevent use of the newer temporal chunking and forced unload controls.

### Rejected: embed the upscaler implementation in U11

Copying the neural upscaler would duplicate model architecture, weight loading, precision handling, and VRAM cleanup logic. That creates unnecessary maintenance and model-compatibility risk.

## Adapter Design

`run_trained_latent_upscaler()` remains the single U11 boundary for the external node.

The adapter will:

1. Resolve either registered node ID:
   - `MinimaxH3LatentUpscaler3D`
   - `MinimaxH3LatentUpscalerNode3D`
2. Prefer the class's real `execute()` implementation when `FUNCTION` points to `EXECUTE_NORMALIZED`.
3. Inspect the chosen callable and construct the common inputs:
   - `latent`: video-only 24-channel H3 latent
   - `model_name`: the exact configured H3 3D latent upscaler model
   - `mode`: multiplier mode with the planned scale
   - `align`: 32-pixel alignment when supported
   - `device`: `cuda`
   - `precision`: `bf16`
4. Add version-specific controls only when declared:
   - Current API: `enable_temporal_chunking=True`, `force_unload=True`
   - Intermediate API: `enable_chunking=True`
   - Older API: `keep_proportion=False`
5. Fall back to an instance method only for legacy node classes whose real implementation is not callable at class level.

The adapter will not pass unknown fields and will not silently guess positional argument order.

## Data Flow and VRAM Behavior

The existing two-stage data flow remains unchanged:

```text
first H3 pass
  -> separate AV latent
  -> unload first-stage H3 allocation
  -> upscale video latent only
  -> preserve audio latent unchanged
  -> concatenate AV latent
  -> second low-sigma H3 pass
```

For the current upscaler API, temporal chunking is enabled to reduce long-video peak allocation and end-frame instability. `force_unload=True` returns the upscaler to CPU and clears its CUDA allocation before the second H3 pass.

No new random noise, resolution policy, LoRA policy, or post-processing behavior is introduced by this repair.

## Validation and Error Handling

Before invoking the node, the adapter will verify that the resolved callable exposes the required semantic inputs for one supported contract. Unsupported contracts fail immediately with an error that names the missing fields and registered node ID.

After invocation, the adapter will retain the existing structural checks and strengthen dimension validation:

- result must contain a `samples` tensor;
- tensor rank must be 4D or 5D;
- channel count must be 24;
- output spatial dimensions must be larger than the input when scale is greater than 1;
- invalid results stop before AV concatenation and second sampling.

Errors must identify the upscaler contract problem rather than exposing an ambiguous normalized-wrapper traceback.

## Tests

Tests will be written before production changes and will cover:

1. Current API with `enable_temporal_chunking` and `force_unload`.
2. ComfyUI v3 `EXECUTE_NORMALIZED` registration while confirming that the real `execute()` receives all arguments.
3. Intermediate API with `enable_chunking`.
4. Older API with `keep_proportion`.
5. Unknown or incomplete contracts failing before model execution with a clear message.
6. Invalid channel counts and non-growing output dimensions being rejected.
7. Existing two-stage routing, workflow, API-template, and documentation tests remaining compatible.

After focused tests pass, the complete plugin test suite will run. Implementation is complete only when the full suite passes from a clean main-branch working tree.

## Delivery

The implementation will be committed directly to the main branch and pushed to its configured remote. The completion report will include the commit ID, focused-test result, full-suite result, and any remaining limitation that affects the later quality research.
