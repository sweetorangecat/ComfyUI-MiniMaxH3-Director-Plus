# Native parent reference implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development for
> implementation, specification review, then independent quality review.

**Goal:** Bind real accepted native parent frames to continuation references in
the original Director, with immutable lineage and existing image limits.

**Architecture:** One focused policy/media module owns exact CPU extraction,
receipt verification and a private IMAGE node. Existing compilation and
coordination call the same overlay before binding/fingerprint checks. Original
duration automatically reserves one existing position; advanced catalogs can
replace a designated selected phase ID.

**Tech stack:** Python, existing ProjectStore, PyAV/Pillow CPU media, Comfy IMAGE
nodes, JSON Schema and existing Python/JS regressions. No GPU or new UI.

## Task1: Policy, physical source and immutable PNG

Files: create `nodes/long_video_parent.py`,
`tests/test_long_video_parent.py`.

- [ ] Write failing tests for `delivery_last` selecting source_endpoint-1 and
  `forward_last` requiring the exact reviewed forward range. A three-color CPU
  movie must yield the actual requested frame, not its last frame or a static
  reference. Reject rejected/nonadjacent parent, missing receipt/native file,
  wrong hash, out-of-range frame, source replacement during extraction and
  partial publication before a readable PNG exists.
- [ ] Implement policy validation and a resolver using the current accepted
  ProjectStore parent and native preview artifact. Decode only one bounded
  physical clip on CPU. Store exact frame PNG and immutable lineage; recheck
  coordination before publication and return the frozen descriptor.
- [ ] Implement verification and the private IMAGE loader using only bound
  project artifacts. Assert no resize and exact decoded pixel equality; reject
  arbitrary paths, stale current parent or PNG/source tampering.
- [ ] Run `python -m pytest tests/test_long_video_parent.py -q` with the existing
  test stubs. Record RED/GREEN and stop inference/downloads if encountered.

## Task2: Canonical contract and both graph handshakes

Files: `nodes/long_video_plan.py`, `nodes/long_video_graph.py`,
`nodes/long_video_service.py`, `nodes/long_video_store.py`, `__init__.py`,
`templates/u11_long_video_contract.schema.json`, corresponding tests.

- [ ] Write failing root/override policy schema-runtime agreement tests.
  Examples use `parent_reference={"mode":"replace","picture_id":2,
  "frame_policy":"delivery_last"}`. Unknown modes/fields, unavailable or
  unselected ID2, and unreviewed forward policies must fail.
- [ ] Carry policy to compiled requests; freeze the resolved source before
  static reference policy/create_take. The actual build_window_graph must
  consume the same frozen descriptor, never resolve the latest file anew.
- [ ] Overlay the private node link and preprocessing source at the designated
  original slot before compacting. Include actual lineage in binding trace,
  per-take generation fingerprint, persisted request/record and native code
  identity; do not modify the global catalog/asset snapshot.
- [ ] Verify eleven catalog IDs -> selected3–5 with Picture2 replacement.
  Original Picture/Subject/Sx/Audio and literal `<d>` contents stay unchanged.
  Add a soft-parent retention instruction in the existing six-field prompt;
  keep model controls outside spoken dialogue and preserve preprocessing.
- [ ] Add races/retry/repair/preview/finish regressions: existing frozen images
  are verified/reused; parent changes invalidate dependent work; missing or
  corrupt frozen images cannot be replaced silently. Run focused parent,
  graph/store/service/plan tests to GREEN.

## Task3: Ordinary duration and original quotas

Files: `nodes/long_video_entry.py`, `nodes/long_video_preflight.py` and their
tests; no additional visible selector or workflow JSON.

- [ ] Write failing original-duration append tests: four ordinary images
  remain original1–4, continuation adds a generated localPicture5. First
  windows and all <=15s paths keep their existing inputs unchanged.
- [ ] Add internal append policy only to ordinary long entry. Append after the
  current ordinary image sequence, preserving all original bindings. Reject
  nine static images and advanced explicit selection+append during pure
  preflight, before any model/hash/job/extraction work.
- [ ] Preflight uses a clearly synthetic unapproved descriptor to exercise
  counts and production binding logic; it never reads parent media or grants
  content approval. Assert spies for extraction/queue/model loading remain
  uncalled. Run entry/preflight/parent tests to GREEN.

## Task4: Review, verification and gated rollout

- [ ] Independent spec review, fix issues and re-review; only then independent
  quality review. Preserve root PyAV sidecar SHA
  `579731293876124363454d6e762bb4835458a2242bb4b8c0e49da8ee4e26a7a8`.
- [ ] Run full Python suite with backup/finally restoration and existing JS
  compatibility checks justified by changed entry/preflight behavior.
- [ ] Update existing long-video docs/H3 guidance with the one automatic entry,
  reserved existing picture position, designated advanced replacement, and
  actual implementation/GPU boundaries. Do not create another user workflow.
- [ ] Commit and push the existing feature branch, preserve local modifications
  and deleted workflows during targeted installation. Defer source-E driver
  changes and remote deployment/restart/GPU tests until the other film ends.

## Status

Design derived from the user's explicit parent endpoint requirement and the
existing reference audit. Implementation pending; no new GPU approval.
