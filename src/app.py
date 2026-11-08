"""HTTP face of Whisper (faster-whisper), a speech-to-text model run in this process on the CPU.

POST /v1/transcribe?language=&beam_size=&word_timestamps=&stream= with 16 kHz mono s16le PCM as the body returns
{"text", "language", "segments", "words", "duration_s", ...}. With stream=true the answer arrives as JSON lines, one
per segment. GET /health reports whether the model is loaded."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
from pathlib import Path
import time
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field

from cache import AnswerCache
from engine import DEFAULT_MODEL, PCM_BYTES_PER_SECOND, EngineError, WhisperEngine, default_marker
from options import SETTINGS
import settings
from tokens import LatencyHistogram, TokenStore
from updates import Updates

_log = logging.getLogger(__name__)

PROJECT_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,47}$"


class TranscribeQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")  # an unknown option is refused, not silently ignored

    language: str = Field("en", pattern=r"^([a-z]{2,3}|auto)$")  # "auto" = detect; needs a multilingual model
    beam_size: int = Field(1, ge=1, le=10)
    word_timestamps: bool = True
    stream: bool = False  # true: the answer arrives as JSON lines, one per segment


class SettingsUpdate(BaseModel):
    values: dict[str, Any]  # None resets an option to its environment variable or default


class NewToken(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    projects: list[Annotated[str, Field(pattern=PROJECT_PATTERN)]] | None = Field(None, max_length=32)
    rate: int | None = Field(None, ge=1, le=100_000)  # calls per minute
    expires_days: int | None = Field(None, ge=1, le=3650)


class TokenScope(BaseModel):
    projects: list[Annotated[str, Field(pattern=PROJECT_PATTERN)]] | None = Field(None, max_length=32)
    rate: int | None = Field(None, ge=1, le=100_000)


class Threads(BaseModel):
    num_thread: int = Field(ge=1, le=256)


def create_app(client: WhisperEngine | None = None, admin_password: str | None = None,
               tokens: TokenStore | None = None, warm: bool = False,
               updates: Updates | None = None, check_hours: float = 0, max_inflight: int | None = None,
               cache_size: int | None = None, config: settings.Config | None = None) -> FastAPI:
    """`admin_password` (SERVICE_ADMIN_PASSWORD) unlocks /admin/, where each project gets a token of its own; when it
    is set, every call except /health must send one of those tokens in the X-Internal-Token header. Unset = no
    auth at all (local use only). `tokens` holds the project tokens (SERVICE_TOKENS_FILE). `cache_size` > 0 keeps that
    many answers per replica (SERVICE_CACHE_SIZE). `config` (the admin page's settings) replaces `warm`, `check_hours`,
    `max_inflight` and `cache_size` and keeps them live; without it they stay fixed."""
    tokens = tokens or TokenStore(os.environ.get("SERVICE_TOKENS_FILE", "tokens.json"))
    client = client or WhisperEngine(config.get("model") if config else os.environ.get("SERVICE_MODEL", DEFAULT_MODEL),
                                     compute_type=config.get("compute_type") if config else "int8", marker=default_marker())
    inflight = 0
    cache = AnswerCache()

    def cache_limit() -> int:
        """Answers this replica keeps; 0 = off."""
        return config.get("cache_size") if config else (
            cache_size if cache_size is not None else int(os.environ.get("SERVICE_CACHE_SIZE") or 0))

    def audio_limit() -> int:
        """Longest clip in bytes of PCM."""
        seconds = config.get("max_seconds") if config else int(os.environ.get("SERVICE_MAX_SECONDS") or 150)
        return seconds * PCM_BYTES_PER_SECOND

    def inflight_limit() -> int:
        """Calls this replica lets wait for the model at once; more get 429. 0 = 4."""
        value = config.get("max_inflight") if config else (
            max_inflight if max_inflight is not None else int(os.environ.get("SERVICE_MAX_INFLIGHT") or 0))
        return value or 4

    updates = updates or Updates(client.model, cached_sha=lambda: client.cached_sha(), remote_sha=lambda: client.remote_sha(),
                                 download=lambda: client.pull(), unload=lambda: client.unload(),
                                 description="Looks for newer model weights on the Hugging Face Hub without downloading.",
                                 reload_note="Weights downloaded. Every replica drops the old copy within a few seconds and loads the new weights on its next request.")

    async def check_forever() -> None:
        """Checks every `update_check_hours` (0 = off). The value is read again every second, so the admin page can change it."""
        last = None
        while True:
            wait = (config.get("update_check_hours") if config else check_hours) * 3600
            now = time.monotonic()
            if wait > 0 and (last is None or now - last >= wait):
                last = now
                try:
                    status = await updates.check()
                    if status["error"]:
                        _log.warning("Update check failed: %s", status["error"])
                except Exception:
                    _log.exception("Update check failed")
            await asyncio.sleep(min(wait, 1.0) if wait > 0 else 1.0)

    async def sync_forever() -> None:
        """Applies what another replica saved (CPU threads, silence filter) and drops the model it replaced with new weights."""
        while True:
            saved = settings.load(tokens.settings_path)
            if saved.get("num_thread") and saved["num_thread"] != client.num_thread:
                client.num_thread = int(saved["num_thread"])
            client.vad_filter = config.get("vad_filter")
            client.sync()
            # A dropped model (download, threads change, sync) leaves /health "Starting" until a request loads it: reload once per drop.
            if config.get("warm") and not client.ready and client.generation != warmed:
                await warm_up()
            await asyncio.sleep(1)

    warmed = 0  # generation of the last warm-up attempt

    async def warm_up() -> None:
        nonlocal warmed
        warmed = client.generation
        try:
            await client.warm()
        except Exception:
            _log.exception("Model warm-up failed; the first request will try again")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        background: list[asyncio.Task] = []
        try:
            if config.get("warm") if config else warm:
                # In the background: warm-up loads downloaded weights only, and /health must answer meanwhile.
                background.append(asyncio.create_task(warm_up()))
            if config or check_hours > 0:
                background.append(asyncio.create_task(check_forever()))
            if config:
                background.append(asyncio.create_task(sync_forever()))
            yield
        finally:
            for task in background:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            await client.close()

    def authorize(request: Request, x_internal_token: str = Header("")) -> None:
        """Checks the token. A token's projects list has no meaning here (one model, no project models). Unset admin password = no auth."""
        if not admin_password:
            return
        row = tokens.lookup(x_internal_token)
        if row is None:
            raise HTTPException(401, "Invalid token")
        wait = tokens.throttle(row)
        if wait:
            raise HTTPException(429, "Too many requests for this token", headers={"Retry-After": str(wait)})
        request.state.token_id = row["id"]

    def admin(x_internal_token: str = Header("")) -> None:
        if not admin_password or not hmac.compare_digest(x_internal_token, admin_password):
            raise HTTPException(401, "Admin password required (set SERVICE_ADMIN_PASSWORD)")

    app = FastAPI(title="Whisper service", lifespan=lifespan)

    async def slot():
        nonlocal inflight
        limit = inflight_limit()
        if limit and inflight >= limit:
            raise HTTPException(429, "Whisper is busy: too many requests waiting", headers={"Retry-After": "5"})
        inflight += 1
        try:
            yield
        finally:
            inflight -= 1

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "model": client.model, "ready": client.ready,
                "num_thread": client.num_thread}

    def build(reply: dict[str, Any], started: float) -> dict[str, Any]:
        return {**reply, "model": client.model, "duration_ms": round((time.perf_counter() - started) * 1000)}

    def cache_key(pcm: bytes, query: TranscribeQuery) -> str:
        """Everything that shapes the answer (decoding is deterministic)."""
        head = json.dumps([client.generation, client.vad_filter, query.language, query.beam_size, query.word_timestamps])
        return hashlib.sha256(head.encode() + pcm).hexdigest()

    def line(event: dict[str, Any]) -> str:
        return json.dumps(event, ensure_ascii=False) + "\n"

    def ndjson(lines: Any, cached: str) -> StreamingResponse:
        return StreamingResponse(lines, media_type="application/x-ndjson", headers={"X-Cache": cached})

    async def replay(result: dict[str, Any]):
        """A kept answer in the shape of a stream: one line per segment, then `done`."""
        for segment in result["segments"]:
            yield line({"type": "segment", **segment})
        yield line({"type": "done", **result})

    async def live_lines(events: Any, first: tuple[str, Any] | None, started: float, key: str | None, size: int, pcm: bytes):
        """One JSON line per segment, then `done` with the same body a call without `stream` returns. A failure
        after the first piece can only be sent as an `error` line."""
        segments: list[dict[str, Any]] = []
        language = "en"
        try:
            async def all_events():
                if first is not None:
                    yield first
                async for event in events:
                    yield event

            async for kind, value in all_events():
                if kind == "segment":
                    segments.append(value)
                    yield line({"type": "segment", **{k: value[k] for k in ("start_s", "end_s", "text")}})
                else:
                    language = value["language"]
        except EngineError as error:
            yield line({"type": "error", "status": 422 if error.status == 400 else (error.status or 500), "message": str(error)})
            return
        except Exception:
            _log.exception("Whisper transcribe failed")
            yield line({"type": "error", "status": 500, "message": "Whisper could not answer"})
            return
        finally:
            await events.aclose()
        result = build(client.body(segments, language, pcm), started)
        if key:
            cache.put(key, result, size)
        yield line({"type": "done", **result})

    @app.post("/v1/transcribe")
    async def transcribe(request: Request, query: Annotated[TranscribeQuery, Query()], response: Response,
                         _auth: None = Depends(authorize), _slot: None = Depends(slot)) -> Any:
        """The body is raw 16 kHz mono signed 16-bit little-endian PCM; the caller decodes (ffmpeg) before it calls."""
        started = time.perf_counter()
        limit = audio_limit()
        if int(request.headers.get("content-length") or 0) > limit:
            raise HTTPException(413, f"Audio is longer than {limit // PCM_BYTES_PER_SECOND} seconds")
        pcm = await request.body()
        if len(pcm) > limit:
            raise HTTPException(413, f"Audio is longer than {limit // PCM_BYTES_PER_SECOND} seconds")
        if not pcm or len(pcm) % 2:
            raise HTTPException(422, "The body must be 16 kHz mono signed 16-bit little-endian PCM")
        size = cache_limit()
        key = cache_key(pcm, query) if size else None
        kept = cache.get(key, size) if key else None
        if kept is not None:
            kept = {**kept, "duration_ms": round((time.perf_counter() - started) * 1000)}
            if query.stream:
                return ndjson(replay(kept), "hit")
            response.headers["X-Cache"] = "hit"
            return kept
        args = (pcm, query.language, query.beam_size, query.word_timestamps)
        try:
            if query.stream:
                events = client.transcribe_stream(*args)
                try:
                    first = await anext(events, None)  # the first segment is waited for here, so a refusal is still an HTTP error
                except BaseException:
                    await events.aclose()
                    raise
                return ndjson(live_lines(events, first, started, key, size, pcm), "miss")
            reply = await client.transcribe(*args)
        except EngineError as error:
            raise HTTPException(422 if error.status == 400 else (error.status or 500), str(error)) from None
        except Exception:
            _log.exception("Whisper transcribe failed")
            raise HTTPException(500, "Whisper could not answer") from None
        result = build(reply, started)
        if key:
            cache.put(key, result, size)
        response.headers["X-Cache"] = "miss"
        return result

    @app.put("/admin/threads", dependencies=[Depends(admin)])
    def set_threads(body: Threads) -> dict[str, Any]:
        """Applies to this replica now and is saved; the other replicas apply it within a few seconds. It beats SERVICE_NUM_THREAD."""
        client.num_thread = body.num_thread
        settings.save(tokens.settings_path, num_thread=body.num_thread)
        return {"num_thread": client.num_thread}

    @app.get("/admin/workers", dependencies=[Depends(admin)])
    def workers() -> dict[str, Any]:
        """Read-only: the replica count is a Docker Compose setting the service cannot change."""
        def number(name: str) -> int | None:
            value = os.environ.get(name, "").strip()
            return int(value) if value.isdigit() else None

        return {"replicas": number("SERVICE_REPLICAS"), "parallel": 1}

    @app.get("/admin/updates", dependencies=[Depends(admin)])
    async def update_status() -> dict[str, Any]:
        """Return the last check without I/O; zero check hours disables only the timer."""
        return {"enabled": True, **updates.status()}

    @app.post("/admin/updates/check", dependencies=[Depends(admin)])
    async def check_updates() -> dict[str, Any]:
        return {"enabled": True, **await updates.check()}

    @app.post("/admin/updates/weights", dependencies=[Depends(admin)])
    async def update_weights() -> dict[str, Any]:
        try:
            return {"enabled": True, **await updates.update_weights()}
        except Exception as error:
            _log.exception("Weight update failed")
            raise HTTPException(502, f"Weight update failed: {error}") from None

    @app.get("/admin/tokens", dependencies=[Depends(admin)])
    def list_tokens() -> dict[str, Any]:
        return {"tokens": tokens.list()}

    @app.post("/admin/tokens", dependencies=[Depends(admin)])
    def create_token(body: NewToken) -> dict[str, Any]:
        return tokens.create(body.name.strip(), body.projects, body.rate, body.expires_days)

    @app.get("/admin/settings", dependencies=[Depends(admin)])
    def get_settings() -> dict[str, Any]:
        return {"settings": config.describe() if config else []}

    @app.put("/admin/settings", dependencies=[Depends(admin)])
    def put_settings(body: SettingsUpdate) -> dict[str, Any]:
        """Saves options; null resets one. Every replica reads the shared file, so a change applies everywhere within
        a second, except options marked `restart`."""
        if config is None:
            raise HTTPException(409, "Settings are fixed in this setup")
        try:
            config.set(body.values)
        except ValueError as error:
            raise HTTPException(422, str(error)) from None
        return {"settings": config.describe()}

    @app.put("/admin/tokens/{token_id}", dependencies=[Depends(admin)])
    def update_token(token_id: str, body: TokenScope) -> dict[str, bool]:
        if not tokens.update(token_id, body.projects, body.rate):
            raise HTTPException(404, "No such token")
        return {"ok": True}

    latency = LatencyHistogram()

    @app.middleware("http")
    async def measure(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        if request.url.path.startswith("/v1/"):
            latency.observe(time.perf_counter() - started)
        token_id = getattr(request.state, "token_id", None)
        if token_id:
            await run_in_threadpool(tokens.record, token_id, (time.perf_counter() - started) * 1000, response.status_code >= 500)
        return response

    @app.get("/metrics", dependencies=[Depends(admin)])
    def metrics() -> PlainTextResponse:
        """Prometheus text per token, from the shared token file (up to a minute behind). Send the admin password."""
        return PlainTextResponse(tokens.metrics("whisper") + latency.render("whisper"))

    @app.delete("/admin/tokens/{token_id}", dependencies=[Depends(admin)])
    def revoke_token(token_id: str) -> dict[str, bool]:
        if not tokens.revoke(token_id):
            raise HTTPException(404, "No such token")
        return {"ok": True}

    web = Path(os.environ.get("SERVICE_WEB_DIR", Path(__file__).resolve().parents[1] / "web"))
    if web.is_dir():
        app.mount("/admin", StaticFiles(directory=web, html=True))

    return app


def main() -> None:
    import sys

    import uvicorn

    if "--dotenv" in sys.argv[1:]:  # run.bat asks for it; a plain `python src/app.py` and Docker never read .env
        env_file = Path(__file__).resolve().parents[1] / ".env"
        if env_file.is_file():
            names = settings.load_env_file(env_file)
            print(f".env: loaded {', '.join(names) or 'no variables'} (names only; a real environment variable wins)", flush=True)
        else:
            print(".env: no file in the service folder", flush=True)
    settings.apply_models_dir()  # SERVICE_MODELS_DIR -> HF_HOME / OLLAMA_MODELS

    store = TokenStore(os.environ.get("SERVICE_TOKENS_FILE", "tokens.json"))
    config = settings.Config(store.settings_path, SETTINGS)
    client = WhisperEngine(config.get("model"), compute_type=config.get("compute_type"), vad_filter=config.get("vad_filter"),
                           marker=default_marker())
    saved = settings.load(store.settings_path)
    threads = saved.get("num_thread") or os.environ.get("SERVICE_NUM_THREAD", "").strip()
    if threads:
        client.num_thread = int(threads)
    app = create_app(client=client, admin_password=os.environ.get("SERVICE_ADMIN_PASSWORD") or None,
                     tokens=store, config=config)
    host, port = os.environ.get("SERVICE_HOST", "127.0.0.1"), int(os.environ.get("SERVICE_PORT", "29300"))

    class Server(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets)  # after uvicorn's own "running on" line
            shown = "127.0.0.1" if host in ("0.0.0.0", "::") else host
            print(f"INFO:     Admin page: http://{shown}:{port}/admin/", flush=True)

    Server(uvicorn.Config(app, host=host, port=port)).run()


if __name__ == "__main__":
    main()
