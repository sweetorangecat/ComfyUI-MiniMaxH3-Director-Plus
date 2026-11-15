# Native repair masks and phase-reference review

The office90 regression retains an accepted 1569-frame prefix (65.375 seconds).
That prefix is not a complete 90-second delivery. A failed trial or an empty
queue does not complete the task and does not authorize shutdown before the
requested film has passed its actual review.

Two subsequent window6 trials failed despite exact protected tensors and an
unchanged complete audio stream:

| Take | Actual review finding | Disposition |
| --- | --- | --- |
| `218223d5151b48d98d8aee44a6523834` | Screen visibly closes, then reopens later before the late fixed quiet phase. | Rejected; no enhancement or continuation. |
| `3a3b57d40e7f4df0843778d5f9aa8245` | Early fixed reference creates whole-frame double exposure at native frames90–93; the screen later lowers while hands rest away from it. | Rejected; no enhancement or continuation. |

The reference's native provenance is real, but its camera/body state is not
interchangeable with the repair's current state. Exact latent preservation
cannot establish a clean decoded entrance, correct prop motion, or acceptable
video. These failures are already visible before VOSR2.

The next isolated trial removes the incompatible fixed future phase and
describes a connected low horizontal hand withdrawal onto wood. It retains
the original seed, weights, actual accepted parent, soft image references,
complete native audio, and correct video steps0–22. Compared with the earlier
no-reference baseline, it changes the actual motion instructions. Compared
with the immediate early-reference trial, it also removes that fixed phase.
It is not yet a verified solution. The original project config and prior
take requests remain immutable; revised execution identities are recorded
on new takes.

## Adapter defect reproduced on CPU

`normalize_repair` and `prepare_av` already support declared temporal
`video_weights`. The SelfLift adapter instead rejected every fractional
value before reaching the upstream global sampler. A regression through
real `prepare_av` followed by `run_selflift` failed with that exact guard.
It did not fail because of a missing GPU or a test import.

The adapter now accepts fractional video masks only when the entire actual
mask exactly matches a valid declared temporal repair recipe, including
any protected reference interval. Undeclared, spatially changed, or
mismatched weights remain errors. Audio still requires binary masks;
geometry, finite range and the original full-sequence Avatar/Sol-Attn route
remain checked. The ordinary short-video route does not call this validator.

CPU acceptance of a mask is an interface check. The upstream model's
decoded transition and the full film still require actual media review.
