# Community Short Video Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add an opt-in SelfLift short-video experiment, explicit motion LoRA profiles, and faithful selected SR routing while retaining low-VRAM operation.

**Architecture:** A pure planning/adapter module owns experimental contracts. Existing director, scheduler and sampler dispatch to it only when selected. Append optional widgets and VAE links to preserve workflow compatibility; keep the stable path's numerical recipe.

**Tech Stack:** Python, ComfyUI node adapters, JavaScript extension, pytest and Node test runner.

---

## Task 1: Experimental adapter and integration

Files: create `nodes/sampling_experiments.py`, `tests/test_sampling_experiments.py`; modify `nodes/director.py`, `nodes/performance.py`, `nodes/two_stage.py`, official U11 JSON.

- [x] Write and run failing tests for `validate_sampling_options(request)`, `plan_selflift_dimensions(width,height,duration,total,free)`, `motion_lora_chain(profile)`, `selflift_dependency_report(registry,upscaler_name)` and `run_selflift(model,positive,vae,sampler,sigmas,latent,seed,guide,registry=None)`.

```python
def test_selflift_preserves_portrait_grid():
    p = plan_selflift_dimensions(1080, 1920, 5, 24, 22)
    assert p['second_stage_width'] < p['second_stage_height']
    assert p['second_stage_width'] * p['second_stage_height'] <= 1344 * 768
```

- [x] Implement strict validation (REF2VA/no voice/quality_two_stage/4–5s/up toFHD), 32px-aligned target and matching upstream0.65 low-grid rounding; reject <20GB total or <18GB free. Adapter requires modern sample method with unload/tiling arguments, resolves installed learned weights, calls `sample(...transition_step=4,lowres_scale=0.65,rho=0,highres_tiling=False,upscaler_unload=True)`, and validates returned full AV shapes.
- [x] Director builds target-resolution conditioning forSelfLift and performs dependency checks before execution. Scheduler returns simple8/split4. Model router preserves matchedv4, disablesSage only forSelfLift, and refuses fallback afterSelfLift selection. Sampler appends optionalVAE and dispatches to adapter before standard two-stage code. Save executed steps/route/timing in guide.
- [x] Add motion profiles to existing LoRA application with legacy fallback only for old guides; new explicit profileoff means no community chain. Connect both example sampler VAE inputs.
- [x] Run adapter+schema+director focused tests and compile changed Python modules. Fix failures before review.

## Task 2: Schema/UI and final SR selection

Files: modify `nodes/schema.py`, `js/minimax_h3_director_plus_v9.js`, `nodes/stream_output.py`; tests `tests/test_schema.py`, `tests/js/director_restore.test.mjs`, `tests/test_stream_output.py`.

- [x] Write failing tests for defaultstandard/off, invalid experimental mode combinations, preserving all low-VRAM mode choices, new widget restoration and three SR options for standard/two-stage presets.

```python
def test_two_stage_accepts_explicit_vosr2():
    r = normalize_request({'mode':'REF2VA','duration':5,
        'performance_preset':'质量优先二采样','postprocess_mode':'vosr2'})
    assert r['postprocess_mode'] == 'vosr2'
```

- [x] Append optional director widgets and API fields; UI puts sampling method/motion profile in advanced二采 controls, defaultstandard/off. Non二采 or nonREF/no-voice switches reset incompatible UI experiment options; API explicitly rejects invalid combinations. Do not remove low-VRAM menus.
- [x] Permit/show VOSR2,SeedVR2,AI in final SR selection. Prioritize explicitly selected SR over planned downscale and native bypass. Missing selected engine raises preflight error. Keep legacy native/Lanczos/RTX API modes.
- [x] Run `node --test tests/js/*.test.mjs` and focused pure/CPU tests. Review actual diff for widget append order and no accidental silent engine fallback.

## Task 3: Review and delivery

Files: create `docs/SelfLift短视频实验.md`; retain supplied-workflow research in `docs/research/`.

- [x] Document exact installation source, initial settings,20GB/18GB startup gate,4–5s/REF/no-voice limitation,8GB existing low-VRAM path, and absence of GPU quality qualification.
- [x] Independent spec review, then code quality review; fix findings and re-run only affected checks.
- [x] Verify JSON links, Python compilation, Node tests and focused tests; record existing-environment limitations separately.
- [x] Commit only task files, integrate without resetting user changes, and report settings and validation limits. Prior user main-push authorization applies; never force-push.

## Delivery notes

Completed sampling adapter, explicit motion profiles, faithful SR routing including smart/fallback/API paths, VAE wiring in canvas/API/builder, and runtime output metadata. Review found and fixed upstream negative-conditioning and inherited-attention override issues. Validation:853 related Python checks,43 Node checks, compilation and bidirectional graph link checks. CPU tests use real Torch/pinned Comfy AV helpers with external runtime boundaries substituted; GPU/AutoDL qualification remains outstanding and is explicitly documented.
