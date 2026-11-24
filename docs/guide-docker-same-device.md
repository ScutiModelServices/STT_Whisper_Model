# STT_Whisper_Model: Docker Desktop on this PC

Run Whisper on the CPU in Docker Desktop on the machine you work on, so other programs on it can call it.

## Requirements

- Docker Desktop with Compose v2 (Windows: WSL2 backend).
- About 0.5 GB of RAM **per replica** for `base.en` (each replica holds the model); `small.en` needs about 1 GB. Keep `SERVICE_REPLICAS=1` on a PC. Docker Desktop on Windows caps its engine at about half of your RAM unless you raise it in `.wslconfig`.
- Network access on the first start to download the weights (about 150 MB for `base.en`) from Hugging Face.
- CPU only (CTranslate2, int8).

## Steps

1. Configure. In `STT_Whisper_Model/`:
   ```bash
   cp .env.example .env
   ```
   Set `SERVICE_ADMIN_PASSWORD` (empty = no auth and no admin page, fine for a local test). Leave `SERVICE_PORT=29300`. Leave `SERVICE_REPLICAS=1`.
2. Start:
   ```bash
   docker compose --env-file .env.compose up -d --build
   ```
   The admin Download action downloads about 150 MB of weights into the `stt_whisper_model_models` volume. `stt_whisper_model_lb` starts only after the service is healthy
   (up to 300 s grace). Watch with `docker compose ps` and `docker compose logs -f stt_whisper_model`.
3. Check:
   ```bash
   curl http://127.0.0.1:29300/health
   ```
   `ready` is `false` until the model has loaded, then `true`.
4. Open the admin page: `http://127.0.0.1:29300/admin/` (see [guide-web-ui.md](guide-web-ui.md)) and make a token for each caller.
5. Try a call:
   ```bash
   ffmpeg -v error -i speech.wav -f s16le -ar 16000 -ac 1 - | curl -X POST http://127.0.0.1:29300/v1/transcribe -H "X-Internal-Token: whisper_..." --data-binary @-
   ```
   Expect a JSON answer.
6. Callers connect to the published port. The caller holds the URL and token:
   - Caller on the host: `http://127.0.0.1:29300`.
   - Caller in another Docker stack on the same host: `http://host.docker.internal:29300`.

## Settings

In `.env.compose` next to `docker-compose.yml` (start with `docker compose --env-file .env.compose up -d --build`):

| Variable | Default | Effect |
|---|---|---|
| `SERVICE_REPLICAS` | `1` | Number of `stt_whisper_model` containers. Read by Compose. |
| `SERVICE_MEM_LIMIT` | `0` (no cap) | RAM cap per container, for example `1g` (`base.en` int8 uses about 0.3 GiB). An out-of-memory kill then stays in the container instead of starving the host. Read by Compose. |
| `SERVICE_PORT` | `29300` | Host port `stt_whisper_model_lb` publishes. |
| `SERVICE_ADMIN_PASSWORD` | empty | Password for `/admin/`. When set, every call except `/health` must send a token as `X-Internal-Token`. Empty = no auth. |
| `SERVICE_NUM_THREAD` | CTranslate2 decides | Starting value for the CPU threads card; a value saved on `/admin/` wins. Use the number of physical cores (`lscpu`: cores x sockets), not logical CPUs; more threads than cores slows every call. Changing it reloads the model on that replica; the next call loads it again. |

On the **Settings** card of `/admin/` (saved in `settings.json` beside the tokens, shared by all replicas; a saved
value beats the environment variable named in brackets): update check hours (`SERVICE_UPDATE_CHECK_HOURS`, 6),
in-flight cap (`SERVICE_MAX_INFLIGHT`, 0 = 4), answers kept per replica (`SERVICE_CACHE_SIZE`, 256, 0 = off), longest audio per call (`SERVICE_MAX_SECONDS`, 150, at most 900), silence filter (`SERVICE_VAD_FILTER`, off), base model
(`SERVICE_MODEL`, restart; `base.en`, `small.en` or a Hugging Face repository of CTranslate2 files), number format (`SERVICE_COMPUTE_TYPE`, restart: `int8`, `int8_float32` or `float32`) and load at start (`SERVICE_WARM`, restart). Compose does not pass these; add the variable
under `environment:` of `stt_whisper_model` to use one as a starting value.

## Daily use

```bash
docker compose ps          # state
docker compose logs -f stt_whisper_model
docker compose down        # stop (the weights volume is kept)
docker compose --env-file .env.compose up -d --build && docker compose restart stt_whisper_model_lb   # after a code change
```

## Problems

| Symptom | Fix |
|---|---|
| `address already in use` on 29300 | Another copy (`run.bat`, or the same stack on another compose project) is running. Stop it or change `SERVICE_PORT`. |
| `/health` shows `ready: false` for minutes | First start: the weights are downloading or loading. Check `docker compose logs -f stt_whisper_model`. |
| 401 Invalid token | The caller's token is missing or revoked; make one on `/admin/`. |
| Container restarts in a loop | Out of RAM. Raise the Docker Desktop memory limit or lower the replica count. |
| `stt_whisper_model_lb` never starts | No replica is healthy yet (first download, or out of memory). `docker compose logs stt_whisper_model`. |
| Replicas restart in a loop | Out of RAM. Lower `SERVICE_REPLICAS`. |
| 422 on `/v1/transcribe` | An empty body, an odd byte count (the body must be 16 kHz mono s16le PCM), an unknown query option, or a language other than `en` on a `.en` model. |
| 413 | The audio is longer than *Longest audio per call*. |
| 429 | Over the token's rate limit, or too many calls waiting for the model. |
| Slow answers | `base.en` runs at about 1.6x real time and `small.en` at about 0.5x on 6 cores. Use a smaller model, or set CPU threads on `/admin/` to the physical core count. |
| nginx 502 after scaling | `stt_whisper_model_lb` not restarted after the replica count changed. |

The load balancer also joins external `citadel_core` as `stt_whisper_model_lb`. Start the Citadel stack first so that network exists; tool services on it use `http://stt_whisper_model_lb:29300`. Docker project and API image are `stt_whisper_model` and `stt_whisper_model:latest`; persistent volumes have explicit `stt_whisper_model_*` names. Volume migration and rollback instructions are in [CHANGELOG.md](../CHANGELOG.md).
