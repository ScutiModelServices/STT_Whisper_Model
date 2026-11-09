# Purpose

**Kind:** audio → text

**Group:** Speech-to-Text

**Tags:** `cpu`, `streaming`, `word-timestamps`

| | |
|---|---|
| Input | Raw 16 kHz mono signed 16-bit PCM (no other formats; caller decodes). Options: language, beam size, word timestamps. |
| Output | JSON: text, language, segments, word timestamps, duration. |
| Runs on | CPU (faster-whisper, `base.en` default). |

**Use for:** transcription with word timestamps. Model half of `ToolServices/Scribe`.
**Not for:** audio decoding, speaker labels.
