# STT_Whisper_Model

The one place Whisper speech-to-text is served: `faster-whisper` (CTranslate2) with `base.en` by default, on the CPU.
Projects send 16 kHz mono PCM over HTTP and get text, segments and word timestamps back. It has the same admin page,
tokens, rate limits, metrics, settings and update checks as the other model services.

It is the model half of `AI/ToolServices/Scribe`: Scribe keeps ffmpeg decoding, per-user quota, the upload store and live
dictation, and calls this service for transcription through a connection saved on Scribe's admin page.

## API

| Call | Purpose |
|---|---|
| `POST /v1/transcribe` | Body: raw PCM. Query: `language`, `beam_size`, `word_timestamps`, `stream`. → `{"text", "language", "segments", "words", "duration_s", "model", "duration_ms"}`. |
| `GET /health` | `{"status": "ok", "model", "ready", "num_thread"}`; `ready` is true once the model is loaded. No token needed. |

**Input.** The body is 16 kHz, mono, signed 16-bit little-endian PCM (`application/octet-stream`), what
`ffmpeg -f s16le -ar 16000 -ac 1` makes. The service does not decode other formats. An empty body or an odd byte count
is 422; more than *Longest audio per call* (`SERVICE_MAX_SECONDS`, default 150, at most 900) is 413.

**Query options.** `language` (default `en`; `auto` detects; a two or three letter code), `beam_size` (1 to 10,
default 1), `word_timestamps` (default `true`), `stream` (default `false`). Any other option is refused with 422. An
English-only model (a name ending in `.en`) refuses any language other than `en` with 422.

**Result.** `segments` are `{"start_s", "end_s", "text"}`; `words` are the same shape per word (empty when
`word_timestamps` is false); `text` joins the non-empty segments with a space. An empty `text` is valid (silence).

**Streaming, per segment.** With `stream=true` the reply is `application/x-ndjson`: one
`{"type": "segment", "start_s", "end_s", "text"}` line per segment, then `{"type": "done", ...}` holding the same body a
call without `stream` returns. A refusal before the first segment is a normal HTTP error; a failure after it is sent as
`{"type": "error", "status", "message"}` and ends the stream. A kept answer is streamed back segment by segment.

**Cache.** *Answers kept per replica* (`SERVICE_CACHE_SIZE`, default 256, 0 = off) keeps the last answers for the same
audio and options and sends them again without transcribing (`X-Cache: hit`). Decoding is deterministic, so every request
is kept. New weights empty the cache. The cache is per replica, so with N replicas the hit rate is about 1/N.

**Tokens.** A token's "allowed projects" list has no effect here: there is one model and no project models.

## Admin page

`/admin/` (password `SERVICE_ADMIN_PASSWORD`): tokens with rate limit and expiry, usage numbers, CPU threads, the
Settings card (update check hours, in-flight cap, answers kept, longest audio, silence filter, and the restart options
model, number format, load at start), and the Updates card (**Check now** and **Download new weights** for the Hugging
Face repository). New weights are downloaded once into the shared models volume; the other replicas notice within a few
seconds and load them on their next request. `/metrics` (admin password) is Prometheus text per token.
`whisper_request_seconds` is a histogram of how long each `/v1` call took on the replica that answers.

Changing the CPU threads reloads the model on that replica (CTranslate2 fixes its threads when the model is built); the
next call loads it again. Set the value to the physical core count.

## Run

On first run, open `/admin/` → Updates → Download. Until weights are downloaded, `/health` reports `ready: false` and inference returns 503. Loading never downloads. See [download steps and curl alternative](docs/guide-web-ui.md#downloaded-weights).

```bash
docker compose --env-file .env.compose up -d --build        # http://<host>:29300
```

Without Docker: `run.bat` (uv, listens on `127.0.0.1:29300`). Open `/admin/` → Updates → Download once to install the model (about 150 MB
for `base.en`); transcription never downloads it. See `deployment.md`.

Example (needs `ffmpeg`):

```bash
ffmpeg -v error -i speech.wav -f s16le -ar 16000 -ac 1 - | curl -X POST "http://127.0.0.1:29300/v1/transcribe" \
  -H "X-Internal-Token: whisper_..." --data-binary @-
```

## Tests

```bash
uv sync
uv run pytest
```

The tests use a fake model; none loads real weights.

## Limits worth knowing

- One transcription at a time per replica; extra calls wait, and more than the in-flight cap get 429.
- The rate limit is per replica.
- Speed on a 6-core CPU, measured in Scribe: `base.en` about 1.6x real time, `small.en` about 0.5x. Raise the model and
  *Longest audio per call* together, and keep the audio under the caller's timeout.
- Model errors are never sent to the caller (they can hold paths): the reply is "Transcription failed. Try again."

## Try it from the admin page

Open `/admin/` and use **Try it** to call the public API with a service token held only in memory. Runs use normal token scopes, limits and caches; the panel shows HTTP status, latency, responses, cancellation and a curl equivalent with a token placeholder. See [the web UI guide](docs/guide-web-ui.md#try-it-send-a-real-request).

Transcribe accepts an audio file and sends raw 16 kHz mono signed 16-bit little-endian PCM to `POST /v1/transcribe`. Browser-decodable encoded audio is converted in the browser; raw `.pcm` input must already have that format. Language, beam size, word timestamps and streaming are query options. Decoded audio is capped at 150 seconds in the browser; the server can impose a lower limit. The curl body for converted audio is `@input.pcm`, not the original encoded file.

## Documentation

- [Purpose: input, output and what it is for](docs/purpose.md)
- [Deployment](deployment.md)
- [Changelog](CHANGELOG.md)
- [Docker on another machine](docs/guide-docker-deployment-device.md)
- [Docker on this PC](docs/guide-docker-same-device.md)
- [Run without Docker](docs/guide-no-docker-same-device.md)
- [Test it from the command line](docs/guide-cli.md)

## Admin pages and navigation

Open `/admin/` and sign in. Overview opens first; each sidebar destination has its own URL fragment and supports browser Back/Forward and direct links:

- `#overview`: health, model readiness, replicas, active tokens and request counts.
- `#playground`: service-specific request forms and streamed or media responses.
- `#tokens`: create, edit, expire and revoke scoped access tokens.
- `#runtime`: CPU threads where supported, replica information and live/restart settings.
- `#updates`: check versions and explicitly download weights.
- `#connections`: caller addresses and public endpoints.
- `#models` and `#voices`: project models and cloned voices, shown only when supported.

Collapse/Expand switches between labeled sidebar and icon rail on desktop and mobile. The non-secret sidebar preference is saved per service in this browser. Navigation preserves form drafts and active Playground requests; sign-out still clears credentials and cancels work. Page names appear once in the main header. No preview data or cross-service selector ships in the real admin UI. Existing authentication, quotas, settings precedence and explicit-download policy remain in effect.
