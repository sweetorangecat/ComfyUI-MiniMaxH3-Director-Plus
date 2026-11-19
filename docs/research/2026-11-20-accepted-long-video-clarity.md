# Long-video clarity against the accepted short reference

## Evidence and scope

The user accepted the picture in `portrait15-clone-selflift-1088` and requested the same texture in longer output, followed by an actual distant-face test. Evidence is in `outputs/diagnostics/accepted_long_reference_20261120/`. The remote GPU identifies itself as RTX4080SUPER with32760MiB; these timings are not a5090 benchmark.

The accepted15s native source is1632×1088,24fps,360frames, SHA256 `6b64af11606c904d1bd51cce8e12cb568a81ba63867cc48b1a1b31696cd552f3`. It used the original picture and voice, an explicitly chosen original-picture frame0 guide, SelfLift6+4, lowres_scale0.7, Euler/beta10, CFG1, native Sol/global_stream/rho0 and the trained2D tier3 lifter. It is a portrait-dialogue reference, not evidence for every scene or action.

## A genuinely new second paragraph

`long30-shot1-selflift-1088` used fresh dialogue, conditioning and seed111331209 with the same original assets and geometry. Its15s generation took1589.379s; separate conditioning preparation took47.207s. Source video SHA256 `aeddad7dab9eabe9a855444301662ecf87ba585a70d49d693df108e7bd3a8ae8`.

Periodic raw pictures, full-frame views and rapid-motion neighbours were inspected. Hair and skin detail are comparable to the accepted source in these samples. Some polished facial rendering remains in both; this is not a claim to have restored photographic pores absent from the generated source. The sampled pictures did not show the former white-strand failure or accumulating scene darkening. CPU Whisper recovered the three authored sentences; CAM++ cosine to the voice reference was0.75853. Those are supporting checks, not listening, lip-sync or visual acceptance certificates.

`final30/accepted_reference30.mp4` combines the accepted paragraph and the new paragraph, with an authored cut at15s. It is30s,720frames,1632×1088. All720 decoded RGB frames and compressed video packets match the corresponding source clips. Clocks and audio timing passed. Assembly plus validation took8.379s, which excludes generation. Final SHA256 `f288dd012a88dc369ad255539d343a4dccc440631da125a6644f6a98f05e5504`. This validates preserving independent short segments through delivery, not an uncut30s generation or unlimited narrative continuity.

## Actual automatic internal-profile test

`long-quality-auto15` exercised the actual internal graph builder and candidate Director, sampler, decoder, color bypass and carried producer gate. The existing FHD3:2 preset resolves to1760×1184; the implementation preserves this selection rather than forcing the diagnostic1632×1088 canvas. The first stage is1216×832, with the actual6+4/0.7 route and tier3 weight hash recorded in the producer evidence.

The15s result contains360frames with exact24fps PTS and BT.709 limited tags. Video SHA256 `2e188b6a8a3a6f121d4ce9a05b549831a646c7ae62e0640ff7e4b4bdf2018f5c`; all16 downloaded evidence files matched remote hashes. The selected matching-time raw face comparisons and rapid-motion neighbours retained the accepted level of detail. Generation took1898.896s; preparation47.049s and cache probe5.145s were separate. This test deliberately saved/reloaded conditioning across a private-process restart to release the text encoder; it is not a full-process peak-memory benchmark.

A real Comfy cache probe with identical visible inputs produced1152×768 from the ordinary Director and1760×1184 from the distinct internal Director type. Mutable hidden prompt metadata alone was insufficient for cache isolation and is not used to activate this route.

## Distant-face tests

`long-quality-far10-full` completed the entire cold original conditioning/model/sampling/decode graph in one private process, without prepared-conditioning reload. Three original approved picture assets, including the active crop from the same4px/cm scale sheet, and one original voice were actually encoded; no hard frame0 was inserted. Output1920×1088,240frames,10s;1289.593s wall time. All33 downloaded files matched remote hashes. Video SHA256 `84c7fb49239636d7b39da546fe0abe935fd1790c4b6b4177ca45398a13e6a4de`.

**Framing failed:** the woman occupied most of the image height rather than the intended one third, and the opening room differed before the office established. The full-body face appears clear in the inspected source frames, but this cannot be counted as a successful far/small-face test.

The scene-first, match-sized-reference follow-up `long-quality-far6-match-full` completed in698.473s,144frames,1920×1088. All25 downloaded files matched their remote hashes. Video SHA256 `730440957f725ff7ed2b236385d9961ded26a59f5042833a8cf02a5d7c260921`. It actually enters a wide scene atframe11 (about0.458s); the woman is small and the face is approximately60px high in the inspected middle frames. The raw6s video was shown to the user. All frame clocks passed; periodic full pictures and unenhanced nearest-neighbour face crops were reviewed. Identity is still readable, but eyes/skin/hair are simplified at this scale. The unrequested opening shot change means it does not pass the intended continuous locked-shot composition. This is real distant-face evidence with limits, not a proof that an enlarged60px face can retain portrait-scale detail. Scene order, prompt and reference encoding size changed together; do not attribute the outcome to only one of them.

The cold10s run sampled a peak27,860MiB GPU and59.986GiB cgroup memory; the prepared15s automatic run sampled27,870MiB GPU and38.708GiB cgroup memory. Different process/preparation scopes make these unsuitable for a direct memory-saving claim. The15s prepared test is not promoted to a cold full-process15s memory guarantee.

## Implementation boundaries

The automatic profile is limited to independent REF2VA / quality-two-stage / SelfLift / continuity /1080p FHD requests on28GB+ hardware, with at least24GB free. Smaller hardware, public short requests and other recipes keep their former planning. The ordinary picture/audio limits and original keyframe roles are unchanged. No portrait is automatically promoted to a scene-opening keyframe.

Qualified segments keep the measured native FHD grid and0.7 first-stage ratio, bypass unnecessary SR/color processing, and carry actual completed sampler/lifter evidence into publication. Reused accepted segments and final assembly must have the same quality receipt, including when an earlier segment is redone while later segments are retained. Changed generation/shot/seed records that the coordinator will rerender are excluded from this reuse comparison.

The final related regression run passed640 tests. This is code-path verification, not visual approval of future generations.

## Verified installation and cleanup

Runtime source commit `2f1c2fbcb768518ba86c5a3de34bca6bf4d87d7a` was pushed on `codex/u11-native-av-long-video`. The local installed plugin was updated only after every affected source matched its old normalized baseline; unrelated Python contents were verified unchanged. A scoped installed commit `7d48735470624e2a6d66a3a4527267725ce6ed30` records the12 selected source/test paths;49 installed-profile tests passed. Existing differing Director/sampling test files and other dirty paths were preserved.

Remote11 runtime files were backed up, hash-checked and installed only with both queues idle. Production restarted asPID38921, with the original command/cwd/environment and the private tests'4GiB headroom / disabled async offload. All three required Director/internal-Director/output node types registered; the two Directors expose identical input schemas, all11 deployed file hashes match, the queue is idle and the original database quick-check returned`ok`. These global memory flags can affect offload performance; unchanged short sampler settings do not establish unchanged timing for every ordinary job. No new workflowJSON or user options were introduced.

The completed private6197 service alone was stopped. The AutoDL instance remains on. After media/evidence verification, eight exact owned prepared/latent cache files were removed,906365270bytes (about0.84GiB). Original inputs, models, videos, raw pictures, JSON evidence and deployment backups were retained. Receipts are under `deployment-2f1c2fbc/` in the diagnostic directory. Future boots must still be checked for actual runtime flags; this receipt establishes the current process, not an unverified startup-script migration.
