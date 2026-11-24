# Purpose

**Kind:** audio → text

**Group:** Speech-to-Text

**Tags:** `cpu`, `streaming`, `word-timestamps`, `variants`

| | |
|---|---|
| Input | Raw 16 kHz mono signed 16-bit PCM (no other formats; caller decodes). Options: language, beam size, word timestamps. |
| Output | JSON: text, language, segments, word timestamps, duration. |
| Runs on | CPU by default (faster-whisper, `base.en`, int8); an NVIDIA GPU when the `device` setting is `cuda` and the GPU libraries are installed (`uv sync --extra gpu`). The Model variant card lists only the model size, number format and device this machine can run, with an estimate of the memory each takes. |

**Use for:** transcription with word timestamps. Model half of `ToolServices/Scribe`.
**Not for:** audio decoding, speaker labels.
