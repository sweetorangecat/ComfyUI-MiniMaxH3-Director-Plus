# Preserve SelfLift texture through VOSR2

The existing LMS path produces fine diagonal native texture which VOSR2
amplifies into visible scratches. A pinned, architecture-matched 2D tier1
lifter substantially reduces this in the reviewed short paired export.
The full345-frame native probe retains the same broad motion, but it does
not establish recovery of reference skin detail or full90-second acceptance.

Implement an optional internal lifter adapter on the original SelfLift route.
Use the exact reviewed checkpoint and pinned MIT-licensed inference source.
Invoke the installed Avatar progressive sampler through its explicit
`latent_lifter` interface; do not replace functions or monkey-patch globals.
Retain actual AV conditioning, original masks, Euler schedule, steps, shifts,
LoRAs and global-stream context. Keep the author/combat recipe on exact LMS.
Legacy or unsupported installations retain the existing sampler behavior.
The UI and saved workflow do not gain any controls or nodes.

1. Add failing tests for producer compatibility, AV/mask identity, fallback,
   strict source/checkpoint identity, tensor geometry and cleanup on failure.
2. Add the pinned source/license and a lazy adapter; select the reviewed
   model only on compatible, verified native global-stream execution.
3. Record the actual lifter and checkpoint identity in execution evidence.
4. Run relevant unit tests and ordinary/author regression checks.
5. Back up remote plugin files, deploy by hashes, restart only with an idle
   queue, and validate the original production route with actual GPU clips.
6. Review enhanced motion/texture, a continuation and the complete long film
   before claiming long-film acceptance. Keep the instance running.

Direct1080p sampling and lower sparse-attention thresholds remain unexecuted
private hypotheses; do not mix them into the measured lifter change.
