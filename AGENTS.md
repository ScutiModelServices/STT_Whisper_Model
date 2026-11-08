# STT_Whisper_Model

faster-whisper (`base.en`, int8, CPU) served over HTTP, behind the shared model-service admin page, tokens and settings. Port 29300, env prefix `SERVICE_`, token prefix `whisper_`. The model half of `AI/ToolServices/Scribe` (which keeps ffmpeg, quota, uploads and live dictation). Requirements: `AI/ModelServices/_docs/WHISPER-SERVICE-REQUIREMENTS.md`.

## Agent instructions
This file is the one instruction file for every coding agent; `CLAUDE.md` only imports it (`@AGENTS.md`). Edit this file, never `CLAUDE.md`.

## Commands (from this folder, venv `.venv`, managed by uv)
- Install: `uv sync` (`winget install astral-sh.uv` if uv is missing)
- Test: `uv run pytest -q` (fake model, no weights). `tests/test_admin_ui.py` needs Playwright and its browser and is skipped without it. On Windows, if uv fails with "cannot move the file to a different disk drive", set `UV_CACHE_DIR` to a folder on the same drive and `UV_LINK_MODE=copy`.
- Run: `run.bat run`, or `uv run python src/app.py` (add `--dotenv` to load `.env`), or `docker compose up -d --build`

## Rules
- Load downloaded weights only. No startup, warm-up or request downloads; missing weights return 503 with an admin Updates Download hint. Only the Updates download callback may fetch weights.
- Follow the `model-service-parity` skill and `AI/ModelServices/_docs/SHARED-FEATURES.md` (what must match in every model service) and `AI/ModelServices/_docs/SERVICE-SPECIFIC.md` (what differs on purpose). A change to a shared feature goes to every service; a new difference goes into `SERVICE-SPECIFIC.md` in the same change.
- `web/index.html`, `src/tokens.py`, `src/settings.py`, `src/updates.py` and `.dockerignore` are copies shared with the other model services. Change one, copy it to the others, run `python tools/check_sync.py` in each repo.
- Built from STC_S1Mini_Model (same app shape and shared files). Differences: `POST /v1/transcribe` takes raw 16 kHz mono s16le PCM as the body (no ffmpeg here; the caller decodes) and `language`, `beam_size`, `word_timestamps`, `stream` as query options (others 422); streaming is one JSON line per segment; no `trust_remote_code`, no project models, no `temperature`/`max_tokens`; `.en` models refuse other languages (422).
- Decoding is deterministic, so every request is cached (key = audio + options + engine generation). Model errors are logged and never sent (they can hold paths).
- Changing `num_thread` drops the loaded model: CTranslate2 fixes its threads when the model is built, so the next call reloads it. This also bumps `generation`, which empties the cache.
- Weights are the Hub repo `Systran/faster-whisper-<name>` (a name with a slash is used as given). `engine.load` loads downloaded files only; use Updates → Download to install them.
- Behaviour changes update `README.md`, `deployment.md`, `docs/guide-web-ui.md`, `.env.example` and `CHANGELOG.md` in the same commit.
- Never read or print `.env`, `tokens.json` or `settings.json`.
- Commit with the noreply author email `214214011+scutiii2@users.noreply.github.com` (another address is rejected on push). Push `main`, then run `python tools/publish_live.py`. Build Docker only when the Dockerfile or the dependencies changed.
