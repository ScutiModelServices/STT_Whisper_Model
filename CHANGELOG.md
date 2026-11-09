# Changelog

## Unreleased

- Model variants and GPU: new `device` setting (`cpu` or `cuda`) and the GPU number formats `float16` and `int8_float16`. A new **Model variant** card on the admin page lists model size x number format x device (`tiny.en` to `large-v3`, with `large-v3-turbo`), each with an estimate of the memory it takes (arithmetic, not measured), its download size and whether it is already downloaded, and offers only what this machine can run: free memory and the NVIDIA GPU are read when the page opens (`src/hardware.py`). `GET /admin/variants` lists them and `PUT /admin/variant` saves model, device and format together (409 when the variant does not fit, nothing saved); a restart applies it. The GPU libraries (cuBLAS 12, cuDNN 9, about 1.1 GB) are the optional `gpu` extra: `uv sync --extra gpu`. Without them a GPU variant shows why it cannot be chosen, and `device=cuda` fails loading with a 503 hint. The shared `web/index.html` gained the card (tag `variants`).

- Env files: `run.bat` (`--dotenv`) now loads `.env.local`, then `.env`; the first file to set a name wins and a real environment variable still beats both, so a host run can keep `SERVICE_OLLAMA_URL=http://127.0.0.1:11434` in `.env.local` while Docker uses `http://host.docker.internal:11434`. `.env.compose` is for Docker Desktop on this device (`docker compose --env-file .env.compose up -d --build`). `.env` stays the file Deploy option 2 pushes to the deployment device, where a plain `docker compose up -d --build` reads it. `.env.local` and `.env.compose` are git-ignored and Docker-ignored (`.env.*`). New shared `settings.load_env_files`.
- Admin page and shared styles: polish layer in `web/index.html` (focus ring, hover and pressed states, layered shadow on the sign-in card and dialogs, tabular numbers, one entrance motion per page, themed scrollbars, `prefers-reduced-motion`). No behaviour change.

- Admin page: a closeable warning on the Overview appears while no weights are downloaded, with a button to the Updates page. While **Download new weights** runs, the Updates card shows a progress bar with megabytes downloaded (`progress` in `GET /admin/updates`; new shared `src/progress.py`). Replicas share the numbers through a small file next to the weights marker; where that folder is not shared (Ollama services) the bar shows on the replica doing the download.

- Fix: a caller that hangs up while the first piece of a stream is still being made now stops the work for nobody. The request timing moved from `@app.middleware("http")` to a plain ASGI middleware (`Measure` in `src/disconnect.py`), because the decorator form hides a hang-up from `request.is_disconnected()`; metrics are unchanged.

- Admin UI: separate hash-routed pages in a collapsible sidebar; health-first Overview, one main-header page title, capability-specific navigation, draft-preserving navigation and responsive desktop/mobile layout. Existing admin and public API actions are preserved.

- Admin page: the CPU threads card and tile are hidden when `CONFIG.tags` has `host-threads`, and the Project models card is shown only with `project-models`.
- Admin page: a note under the CPU threads card says a change reloads the model (`CONFIG.threadNote`).

- Admin page: `CONFIG.tags` (same tags as `docs/purpose.md`) switch tag-driven notes: `gpu` shows a CPU-threads note, `license-restricted` shows a licence note on the Updates card.

- Breaking: startup, warm-up and requests load downloaded weights only. First run requires `/admin/` → Updates → Download; missing weights keep health/admin available and inference returns 503. Checks never download; explicit Download replaces weights and tells every replica to reload.
- Fix: after Download, a CPU threads change or a sync from another replica drops the loaded model, the service now reloads it in the background (when "Load the model at start" is on) instead of showing "Starting" on `/admin/` until the first request.
- Updates shows downloaded and online versions, newer-version and not-downloaded states; Download remains available when online comparison fails.

- `SERVICE_MODELS_DIR` sets where a local (no Docker) run keeps downloaded model weights: `settings.apply_models_dir()` maps it to `HF_HOME` and `OLLAMA_MODELS` at start.
- Docs: `docs/guide-cli.md`, a PowerShell-only way to test the service.

### Breaking: common service environment variables

Service-owned environment variables now use `SERVICE_*`. Old names are no longer read.
Rename old names once in your real `.env`, Compose `environment:` block, and CI/secret settings.

| Old name | New name |
|---|---|
| `WHISPER_ADMIN_PASSWORD` | `SERVICE_ADMIN_PASSWORD` |
| `WHISPER_CACHE_SIZE` | `SERVICE_CACHE_SIZE` |
| `WHISPER_COMPUTE_TYPE` | `SERVICE_COMPUTE_TYPE` |
| `WHISPER_HOST` | `SERVICE_HOST` |
| `WHISPER_MAX_INFLIGHT` | `SERVICE_MAX_INFLIGHT` |
| `WHISPER_MAX_SECONDS` | `SERVICE_MAX_SECONDS` |
| `WHISPER_MEM_LIMIT` | `SERVICE_MEM_LIMIT` |
| `WHISPER_MODEL` | `SERVICE_MODEL` |
| `WHISPER_NUM_THREAD` | `SERVICE_NUM_THREAD` |
| `WHISPER_PORT` | `SERVICE_PORT` |
| `WHISPER_REPLICAS` | `SERVICE_REPLICAS` |
| `WHISPER_TOKENS_FILE` | `SERVICE_TOKENS_FILE` |
| `WHISPER_UPDATE_CHECK_HOURS` | `SERVICE_UPDATE_CHECK_HOURS` |
| `WHISPER_VAD_FILTER` | `SERVICE_VAD_FILTER` |
| `WHISPER_WARM` | `SERVICE_WARM` |
| `WHISPER_WEB_DIR` | `SERVICE_WEB_DIR` |
| `WHISPER_WEIGHTS_MARKER` | `SERVICE_WEIGHTS_MARKER` |

- Move the default service port from 9230 to 29300, including API replicas and the nginx load balancer. Update any explicit old port in your real `.env`, saved tool-service `backends.json` connections, and browser or MCP client configuration once. With Compose defaults, recreate the stack to use the new host and container ports; a custom host-port override stays as configured, while container ports still change to 29300.

- Renamed `STTWhisperModel` to `STT_Whisper_Model`; Docker project, API image, services, load balancer and volumes now use `stt_whisper_model` names. Callers on `citadel_core` connect to `stt_whisper_model_lb` at the existing port.
- Docker volume migration: existing volumes with old names are orphaned; renaming does not move data. Stop the old stack before copying; never use `down -v`. The commands below assume the previous default Compose project name; substitute actual volume names if `-p` or `COMPOSE_PROJECT_NAME` overrode it.
  - `whisper_models` → `stt_whisper_model_models`: `docker volume inspect whisper_models && docker volume create stt_whisper_model_models && docker run --rm -v whisper_models:/from:ro -v stt_whisper_model_models:/to alpine:3.21 sh -c "cd /from && cp -a . /to/"`
  - `whisper_tokens` → `stt_whisper_model_data`: `docker volume inspect whisper_tokens && docker volume create stt_whisper_model_data && docker run --rm -v whisper_tokens:/from:ro -v stt_whisper_model_data:/to alpine:3.21 sh -c "cd /from && cp -a . /to/"`
- To retain an old volume instead, set its top-level volume definition to `external: true` and `name: <old-volume-name>` in a local Compose override. Rollback: stop the renamed stack, restore the previous checkout and folder name, and restart the old stack with its original volumes; copy back newer data first if the renamed stack accepted writes.

- `run.bat` loads `.env` from the service folder for a local run: it passes `--dotenv`, the service prints which variable names it loaded (never the values), and a real environment variable still wins. A plain `python src/app.py` and Docker never read `.env`.
- Untracked folders now start with `.` or `_`: `zip_versions/` is `_zip_versions/`. A local `voices`, `tem_code` or `zip_versions` folder from an older checkout needs renaming by hand.
- `run.bat` is now a menu (run the service, deploy, build the release ZIP) and `run.bat run|deploy|zip` skips it. `tools/deploy.bat`, `tools/make_zip.bat` and `tools/deploy.local.example.bat` are gone; the local deploy target is `tools/deploy.local.ini` (copy `tools/deploy.local.example.ini`).
- Admin page: **Connect to this service** card with the addresses other services can use (this page's address, same PC, Docker network, `host.docker.internal`) and every endpoint, each with a Copy button.
- Admin **Try it** panel: descriptor-driven public API requests, live text streams, JSON/media output, status and latency, cancellation and redacted curl. Auth choice (b): pasted public token stays in memory; normal scope, limits and caches apply. Shared `web/playground.js` is copied across all model and tool services.
- Corrected the external `faster_whisper.STT_Whisper_Model` loader name so real transcription works after the service rename.

- First version: `faster-whisper` (`base.en`, int8) served on the CPU, with the admin page, tokens (rate limit,
  expiry), metrics, Settings card, update checks, weight downloads and answer cache shared with the other model
  services. Built from STC_S1Mini_Model.
- `POST /v1/transcribe` takes 16 kHz mono s16le PCM as the body and `language`, `beam_size`, `word_timestamps` and
  `stream` as query options. `stream=true` sends one JSON line per segment, then `done`.
- Settings: `max_seconds`, `vad_filter`, `compute_type` (and the shared ones).
