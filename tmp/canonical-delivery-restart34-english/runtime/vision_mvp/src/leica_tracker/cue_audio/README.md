# Fixed English direction audio

The library covers measured directions, arrival, an opening greeting, six
framing-wait variations and three countdown numbers. Healthy waits use a
1.5-second initial delay and at least eight seconds between phrase starts,
with a three-phrase limit per continuous wait. Countdown audio follows the
photo flow's actual stage callbacks.

These short WAV clips are generated once from the teacher-provided
`en_US-ljspeech-high` Piper model. The app serves the checked clips to the iPad;
it does not run speech synthesis, a microphone, Ollama, or another robot-control
program during an encounter.

All clips are mono PCM16 at 22,050 Hz. `manifest.json` records the allowlisted
cue names, exact phrases, durations and file hashes. `unlock.wav` is 80 ms of
silence for browser audio activation during an explicit user gesture.

The source model and configuration are not included. Their SHA-256 values and
the isolated build dependencies are recorded in the manifest. Regeneration uses
`vision_mvp/scripts/render-direction-audio.py --execute` with the checked model
and workspace-local build dependencies. The default invocation is read-only.

The original voice model card and source record are retained alongside the
clips. The card describes the LJSpeech dataset as public domain. The Piper
runtime used for the one-time build is GPL-3.0; its source is
<https://github.com/OHF-Voice/piper1-gpl>. No Piper runtime or model is bundled
with these playback assets.
