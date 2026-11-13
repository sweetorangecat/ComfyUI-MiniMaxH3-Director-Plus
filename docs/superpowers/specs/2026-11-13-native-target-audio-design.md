# Native long-video target audio

## Purpose and evidence boundary

Keep the original U11 Director and its existing input limits. Add an explicit
way to condition video on a complete prepared audio timeline, rather than
relying on every native AV window to repeat precisely the requested words.
The existing `none` and `h3_reference` joint-AV routes remain unchanged unless
an explicit target-audio recipe is supplied. No separate production workflow
or additional prominent UI controls are introduced.

The related 90-second experiment is still incomplete. One accepted window
demonstrated full target-audio preservation after restoring actual parent
history; another preserved Fish audio but failed visual prop continuity.
Tensor preservation, ASR, audible quality, lip sync, and visual content are
separate acceptance conditions. This implementation is not GPU certification.

## Two producers, one explicit contract

1. An internal/API recipe supplies a full physical-window WAV and provenance.
   It is authoritative for the complete soundscape, not an additional numbered
   H3 audio reference. Ordinary callers need not use this interface.
2. The existing `fish_lock` choice can prepare complete, timed utterances from
   the original Audio1, cache them, and compose a target before video sampling.
   It must never silently assign Audio1 to an unbound ordinary speaker.

Each prepared target records physical window geometry, waveform SHA-256,
sample rate, channel and sample counts, original source identities, whole
dialogue identities/text/Subject/Sx bindings, actual durations, placement,
generation seed, and requested/resolved backend/model/code provenance.
Supplied manifests are checked against actual files and original bindings;
their existence does not grant audible or visual approval.

Fish's minimum automatic scope is explicit Audio1-bound cloned speech with
intentional silence in the gaps, or an explicitly supplied complete background
recipe. Fully locked audio prevents H3 from inventing ambience or effects.
Reject an unsupported/missing soundscape recipe clearly; do not silently remove
requested room tone, loop unreviewed parent material, or claim Foley that was
never supplied. Other speakers require explicit waveform/voice provenance.
Conflicting nonempty `target_dialogue` fails; an empty value may derive complete
utterances from the existing timed `<d>` lines. Preserve the original reference
transcript and the existing Fish dependency requirements.

## Timing and immutable preparation

Use the existing 24-fps global clock and native audio tick geometry. At 32 kHz,
one 40-Hz audio tick is 800 samples. Map frame positions using rational sample
arithmetic. A whole utterance must fit its explicit deadline and delivered
interval. No word truncation, rate changes, automatic speech compression, or
post-render muting/deletion is allowed. Padding applies only to declared
background/silence and known encoder suffix geometry, never to make an
overlong utterance appear valid.

Run Fish in ordinary Comfy preparation jobs, not in the coordinator's service
thread. Reserve preparation separately from native takes, since the final
target hash does not exist yet. Reconcile interrupted/missing jobs and publish
immutable sentence/target receipts before native request fingerprints.
Sentence caches are shared by content identity across projects. Their keys
include text, language, original reference bytes/transcript, backend/model/code
identities, explicit generation parameters and a stable recorded sentence seed;
they exclude output dimensions/codec, project name, window index and video
retry. Cached speech survives export changes and video retries. Actual source
or generated-target changes invalidate the appropriate generation chain.

## Native AV seam

Load the validated target and use the official audio VAE encoding contract.
Install its finite, correctly shaped audio latent before the existing AV
history/forward copy. Restore the actual parent clean audio values exactly,
including reviewed forward material, and set every audio noise-mask element to
zero. Retain the existing video masks and open only their existing new-video
region. Do not rebuild history from a planned WAV or a decoded approximation.

Target-conditioned repair inherits the failed take's complete audio and
rejects open-audio ranges or `lock_audio=False`. Changing target audio requires
a new base generation, with existing descendant invalidation. Preview and
finish reuse stored clean AV; they do not re-run Fish, composition or encoding.

Translate original long `fish_lock` requests inside this native adapter while
retaining requested voice mode and provenance. Do not globally relax the short
Fish/two-stage policy. Continue pruning the old Fish bridge mux override from
native long graphs. Label the new method
`target_audio_conditioned_native_video`; do not call it untouched joint AV.

## Verification and rollout

CPU regressions cover whole-waveform overflow, source/hash replacement,
speaker bindings, exact clock/geometry, immutable cache/recovery, export-only
reuse, exact parent/forward audio restoration, all-zero audio mask, repair
restrictions, and unchanged default/short routes. Stub Fish and VAE seams;
never present those tests as voice fidelity or GPU quality evidence.

Work is offline until the user's other running long-video task ends and the
remote queue is idle. Existing user edits, media, diagnostics and workflow
deletions must remain intact. GPU content failures remain failures even when
all technical audio checks pass.
