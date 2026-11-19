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

**Framing failed:** the woman occupied most of the image height rather than the intended one third, and the opening room differed before the office established. The full-body face appears clear in the inspected source frames, but this cannot be counted as a successful far/small-face test. A scene-first, match-sized-reference6s follow-up is being evaluated separately; do not attribute its result solely to one changed parameter or publish the failed composition as the requested far shot.

## Implementation boundaries

The automatic profile is limited to independent REF2VA / quality-two-stage / SelfLift / continuity /1080p FHD requests on28GB+ hardware, with at least24GB free. Smaller hardware, public short requests and other recipes keep their former planning. The ordinary picture/audio limits and original keyframe roles are unchanged. No portrait is automatically promoted to a scene-opening keyframe.

Qualified segments keep the measured native FHD grid and0.7 first-stage ratio, bypass unnecessary SR/color processing, and carry actual completed sampler/lifter evidence into publication. Reused accepted segments and final assembly must have the same quality receipt, including when an earlier segment is redone while later segments are retained. Changed generation/shot/seed records that the coordinator will rerender are excluded from this reuse comparison.

The final related regression run passed640 tests. This is code-path verification, not visual approval of future generations. Deployment and the final distant-face result must be recorded after they actually occur. Keep the AutoDL instance running; clean only completed owned diagnostic tensors after media/evidence are secured.
