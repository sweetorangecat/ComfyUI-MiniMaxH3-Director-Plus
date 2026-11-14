# Long-video dialogue rendering regression

The long-video prompt compiler emitted all action phases before emitting any
dialogue. A window could therefore describe looking up, speaking, closing the
lips and looking down, then return to the earlier speaking interval to supply
the actual words. The parser also left `says naturally in Mandarin:` in the
action lane while attaching the words to a separate `says exactly` instruction.

The compiler now merges action phases and complete utterances in chronological
local time, including individual `timed_actions`. Author-supplied speech delivery
stays next to its utterance. Frame ranges remain in the structured audit; the
model receives natural timestamp ranges. Speaker IDs, audio bindings, complete
spoken text, reference counts and continuation masks retain their meanings.
The ordinary 4–15 second route does not invoke this long-video compilation path.

## Evidence as of November 15, 2026

The diagnostic device reported **RTX 4080, 32 GB**. These are not RTX 5090
measurements. All three classic-route probes used seed 111330200, the same three
original pictures, the same original voice reference, model and sampling recipe.
Only the director prompt and output prefixes differed between probes. The
physical clips contained 328 frames at 1344×768 and 24 fps; the inspected delivery
interval was frames 0–293, or 12.25 seconds.

| Probe | Prompt ID | Result |
|---|---|---|
| Replay of the old working opening | `b73c82b2-db0b-433a-b41b-b909bf20216a` | Full opening in unprompted ASR; early card handling prevents claiming exact narrative compliance. |
| Current compiled prose on the same classic graph | `c448e166-c8c5-46da-ab73-f3828d24e685` | Two unprompted ASR models place the opening at about 7.48 s; the whole line is absent at the delivery endpoint. No long-video mask or context nodes were present. |
| Repaired compiled prose on the same classic graph | `27cd26b8-f700-4806-8ac0-983782f0d2f0` | Full opening in unprompted small-model ASR at about 0.56–5.56 s. This is earlier than the scheduled 1.5–7.8 s, so exact timing is not established. |

Every twelfth native frame through frame 293 and native-resolution frame 48 were
inspected. Texture remains close to the old opening: hair strands, skin shading
and fabric detail survive. This is not proof of exact reference-image recovery.
ASR output is machine transcription evidence, not a claim of listening or a
voice-identity assessment.

The prompt-only reproduction implicates prompt rendering in the delayed speech;
it does not exclude other continuation defects. The earlier poor 45-second
trial also had a separate input regression: its only reference was a generated
late-window parent frame rather than the original face, scene and scale sheet.

Three new regressions failed before the fix and passed after it. The related
entry, prompt, plan, parent and runtime suite passed **308 tests**, with the
existing PyTorch `pynvml` deprecation warning. GPU time for the repaired classic
probe was about 637 seconds, comparable to the unrepaired probe; no speedup is
claimed. A complete repaired long-film run remains a separate acceptance gate.
