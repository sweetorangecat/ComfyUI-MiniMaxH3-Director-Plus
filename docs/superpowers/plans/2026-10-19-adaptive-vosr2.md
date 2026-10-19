# Adaptive VOSR2 Implementation Plan

**Goal:** Reconstruct only the dimensions needed by the requested minimum, retaining source detail and a simple UI.

**Architecture:** Exact rational sizing feeds both the output encoder and VOSR2 adapter. Fractional resizing is implemented through the existing integer backend's scale=1 restoration path. No additional sampler or external runtime is added.

**Tech Stack:** Python, PyTorch, existing TE-Speed-VOSR2, JavaScript, pytest and Node test runner.

- [x] Change expectations in `tests/test_video_finish.py`, `tests/test_vosr2_profiles.py`, `tests/test_single_pass_sr.py`, `tests/test_stream_output.py`; add fractional backend-input and nonfinite-output cases. Run affected tests and confirm old integer rounding fails.
- [x] Implement rational minimum sizing and bounded float preparation in `nodes/video_finish.py`; report `backend_scale`, `processing_input_size` and `sizing_strategy` in `nodes/execution_report.py`.
- [x] Remove forced input shrinking from `nodes/director.py` and `nodes/stream_output.py`; carry actual minimum dimensions into encoding and metadata.
- [x] Move VOSR2 profile controls into collapsed advanced details in `js/minimax_h3_director_plus_v10.js`, preserving saved values and open state. Verify with the existing DOM lifecycle tests.
- [x] Update the affected usage documentation. Run the Python suite excluding the pre-existing deleted example fixture, run JS tests, and perform actual local backend inference on a short sample.
- [x] Complete independent review, repair legacy-guide sizing and validate all eight compatibility combinations.

Delivery: commit only task files, push the current branch and fast-forward the clean local installed plugin. Report the resulting revision after these operations succeed.

User delegated implementation choices; execute inline in the existing workspace. Existing unrelated edits and deleted example remain untouched.
