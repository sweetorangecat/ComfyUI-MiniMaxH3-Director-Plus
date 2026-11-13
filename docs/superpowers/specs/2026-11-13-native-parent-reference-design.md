# Actual parent-frame references in native continuation

## Authorized scope

Finish the original U11 long-video entry by conditioning each continuation on
the actual accepted adjacent parent's native image, while retaining its true
clean AV history. The user explicitly requires the Office90 internal catalog's
Picture2 to be replaced before sampling; a static 11-ID selection audit is not
this implementation. No new production workflow or prominent UI controls.
Do not alter the running Office90 experiment or deploy/restart/queue GPU work
until that separate film and its queued work finish.

## Policy and existing quotas

Keep the original UI's nine image positions, three audio positions, the actual
Director's nine simultaneous-picture limit, and the existing internal catalog
of eleven possible original IDs. These are distinct limits. Neither this
feature nor a larger total duration expands them.

An optional internal `parent_reference` policy is carried by the canonical
contract and exact-window overrides:

```json
{"mode":"replace", "picture_id":2, "frame_policy":"delivery_last"}
```

`replace` requires that original ID to exist and be selected in the current
window. It replaces that designated phase reference, preserving all numbering,
Subject/Sx/Audio bindings and the number of active pictures. It does not choose
an identity/scale/prop reference to replace on the user's behalf. This supports
Office90's eleven original IDs -> three-to-five selected images, including2.

For the ordinary Director-duration entry, use an internal `mode: append`
policy: automatically add one generated parent reference after the existing
pictures, retaining all original numbering and files. This consumes one of the
existing nine conditioning positions, not a new upload control. Preflight must
reject more than eight ordinary static pictures before hashing/loading/queueing
and explain that one existing position is required for continuity. H3 skills
must plan static assets within that allowance. No automatic replacement of an
arbitrary ninth identity/scale/storyboard image. `append` with an explicit
advanced `reference_selection` is rejected; those callers designate an existing
slot with `replace`. First windows do not create or require a parent image.

Advanced contracts without this optional policy retain their old behavior.
The ordinary >15s entry installs append internally. Short paths remain inert.

## Frame, source and review

Default `delivery_last` means `child.window.source_endpoint - 1` in the accepted
parent's full native preview, matching its actual delivered end. Do not choose
the last physical safety-tail frame, enhanced video, old P01, latest folder,
expected state, or an unrelated accepted take. Verify actual adjacency,
accepted review, receipt/file hashes, source frame range and native geometry.

The explicit `forward_last` exception selects
`child.window.source_endpoint + forward_frames - 1`, only with a matching actual
forward-tail review. It preserves Office90's previously reviewed W02 frame327
exception and does not treat accepted delivery approval as tail approval.
Frame policy and scope are frozen; do not silently switch between them.

Decode the exact native preview frame on CPU, with no resize, retouching,
enhancement or video re-encoding. Publish an immutable PNG and receipt binding
project, child window, parent take, native source artifact/hash, review identity,
frame index/policy, dimensions and PNG SHA. Verify source bytes before and after
extraction and publish only complete files. Hashes/extraction grant no new
visual approval; approved clean AV remains the true temporal history.

## One binding handshake

Add a focused `nodes/long_video_parent.py` for policy, immutable extraction,
lineage verification and a private IMAGE loader node. Reuse ProjectStore and
ordinary CPU media APIs. Never upload through a browser or require users to
select per-window files. The private loader accepts only a bound project
artifact, verifies hashes/current adjacent accepted source, and cannot become
an arbitrary filesystem loader.

The Coordinator resolves/freezes this source before reference policy,
fingerprinting and create_take. The compiled graph applies the exact same
frozen overlay before its immutable reference-binding comparison. The
per-take request/binding trace records actual node link, source and local
Picture role. Global base_graph/catalog remain unchanged; no generated asset
is added to global asset hashes or causes previous windows to invalidate.
Source/PNG changes affect the dependent generation chain and caches.

Use the real reference-preprocessing check for the new binding. Keep all six
official fields and literal dialogue. Add an English, nonspoken retention
instruction identifying the actual local parent picture as soft observed
continuation state, not a hard first/last frame or a second command to replay
old actions. Preserve original reference mappings; no fictitious Video/Audio.

Static preflight may validate counts/selection with an explicitly unobserved
placeholder only. It must not decode, write, hash models, enqueue or grant
approval. Runtime missing/rejected/unreviewed/stale sources stop clearly;
there is no static-picture fallback. Reservation checks before publication and
take creation must reject concurrent parent replacement. The queued IMAGE
node rechecks frozen source/PNG ownership before use.

## Resume, repair and evidence

Reusing an existing take, previewing, exporting or repairing must verify and
reuse its frozen image/receipt. They do not choose a new parent or re-extract a
missing/corrupt frozen PNG. A new explicit base generation may resolve a new
current accepted source through the normal invalidation path. Old files are
retained. Repair requires the same current parent/history and reference
identity; a changed parent cannot be restored through an old repair request.

Tests must exercise actual CPU video -> PNG extraction and pixel equality,
frame/range/review selection, both binding handshakes, 11-ID replace2 mapping,
ordinary append and early quota rejection, literal dialogue/voice invariance,
source/PNG tampering, source-change races, atomic publication, recovery and
short/default compatibility. Add the module to native code identity. CPU
passing is not new GPU/motion/voice certification. Existing 766dd23 audio
conditioning and decoder-memory fixes must remain intact.
