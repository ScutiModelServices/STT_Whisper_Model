"""faster-whisper (CTranslate2, int8) run in this process on the CPU: 16 kHz mono s16le PCM in, text and timestamps out.

Decoding is deterministic (no sampling, `beam_size` default 1), so every request can be kept in the answer cache.
One transcription runs at a time per replica (CTranslate2 already uses every core); scale with replicas.
Changing the CPU threads reloads the model (CTranslate2 fixes them when the model is built); the next call loads it.

shortcut: PCM only, ffmpeg stays with the caller; an English-only model (`*.en`) refuses any other language."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Iterator

DEFAULT_MODEL = "base.en"
COMPUTE_TYPES = ("int8", "int8_float32", "float32")
PCM_BYTES_PER_SECOND = 16000 * 2  # 16 kHz, mono, 16-bit

_log = logging.getLogger(__name__)


class EngineError(RuntimeError):
    """The model could not answer. `status` is the HTTP status the service maps it to."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def repo_of(model: str) -> str:
    """`base.en` is Systran's `faster-whisper-base.en` on the Hub; a name with a slash is a repository already."""
    return model if "/" in model else f"Systran/faster-whisper-{model}"


class Loaded:
    """A loaded faster-whisper model that transcribes one clip."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def stream(self, pcm: bytes, language: str | None, beam_size: int, words: bool, vad: bool,
               stop: threading.Event) -> Iterator[tuple[str, Any]]:
        """("segment", dict) as the model makes them, then ("done", {"language"}). Setting `stop` ends it after the
        segment being made."""
        import numpy as np

        audio = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
        segments, info = self.model.transcribe(audio, language=language, beam_size=beam_size,
                                               word_timestamps=words, vad_filter=vad)
        for segment in segments:
            if stop.is_set():
                return
            yield "segment", {
                "start_s": round(segment.start, 2), "end_s": round(segment.end, 2), "text": segment.text.strip(),
                "words": [{"start_s": round(w.start, 2), "end_s": round(w.end, 2), "text": w.word.strip()}
                          for w in (segment.words or [])],
            }
        yield "done", {"language": info.language}


def load(model: str, compute_type: str, threads: int) -> Loaded:
    """Load downloaded files only; missing weights leave the service available for admin downloads."""
    from faster_whisper import WhisperModel
    from huggingface_hub import snapshot_download

    repo = repo_of(model)
    path = snapshot_download(repo_id=repo, local_files_only=True, token=False)
    return Loaded(WhisperModel(path, device="cpu", compute_type=compute_type, cpu_threads=threads, local_files_only=True))


def _remote_sha(repo: str) -> str:
    from huggingface_hub import HfApi

    return HfApi().model_info(repo, token=False).sha


def _cached_sha(repo: str) -> str | None:
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(repo_id=repo, local_files_only=True, token=False)).name  # the folder is named by commit
    except OSError:
        return None


def _download(repo: str) -> None:
    from huggingface_hub import snapshot_download
    from progress import hub_blobs, hub_total, tracking

    with tracking(repo, hub_total(repo, None), hub_blobs(repo)):
        snapshot_download(repo_id=repo, token=False)


def default_marker() -> Path:
    """File in the shared models volume whose content changes when one replica downloads new weights."""
    configured = os.environ.get("SERVICE_WEIGHTS_MARKER")
    return Path(configured) if configured else Path(os.environ.get("HF_HOME") or "~/.cache/huggingface").expanduser() / ".whisper-weights-version"


class WhisperEngine:
    """The `loader`, `remote_sha`, `cached_sha` and `download` callables are injectable for tests. `loader(model)`
    returns an object with `stream(pcm, language, beam_size, words, vad, stop)`."""

    def __init__(self, model: str = DEFAULT_MODEL, *, compute_type: str = "int8", vad_filter: bool = False,
                 marker: Path | None = None, loader: Callable[[str], Loaded] | None = None,
                 remote_sha: Callable[[str], str] = _remote_sha, cached_sha: Callable[[str], str | None] = _cached_sha,
                 download: Callable[[str], None] = _download) -> None:
        if compute_type not in COMPUTE_TYPES:
            raise ValueError(f"compute_type must be one of {', '.join(COMPUTE_TYPES)}, not {compute_type!r}")
        self.model = model
        self.compute_type = compute_type
        self.vad_filter = vad_filter
        self._threads: int | None = None
        self._marker = marker
        self._seen = self._read_marker()
        self._loader = loader or (lambda name: load(name, self.compute_type, self._threads or 0))
        self._remote_sha_of, self._cached_sha_of, self._download_of = remote_sha, cached_sha, download
        self._base: Loaded | None = None
        self._load_lock = threading.Lock()
        self._gate = asyncio.Semaphore(1)
        self.generation = 0  # part of the answer cache key; changes whenever the loaded model is dropped

    @property
    def num_thread(self) -> int | None:
        return self._threads

    @num_thread.setter
    def num_thread(self, count: int) -> None:
        if count != self._threads:
            self._threads = count
            self._forget()  # CTranslate2 fixes its threads at build time: the next call loads the model again

    @property
    def ready(self) -> bool:
        return self._base is not None

    async def close(self) -> None:
        self._base = None

    async def warm(self) -> None:
        """Load the model now so the first request does not pay for it."""
        await asyncio.to_thread(self._base_model)

    def _base_model(self) -> Loaded:
        with self._load_lock:
            if self._base is None:
                try:
                    self._base = self._loader(self.model)
                except OSError as error:
                    raise EngineError("Model weights unavailable. Ask an admin to press Download on the Updates card.", 503) from error
            return self._base

    def _language(self, language: str) -> str | None:
        if language == "auto":
            language = ""
        if self.model.endswith(".en") and language not in ("", "en"):
            raise EngineError(f"{self.model} only knows English; use a multilingual model for {language!r}", 400)
        return language or None

    def _events(self, pcm: bytes, language: str, beam_size: int, words: bool, stop: threading.Event) -> Iterator[tuple[str, Any]]:
        code = self._language(language)
        try:
            yield from self._base_model().stream(pcm, code, beam_size, words, self.vad_filter, stop)
        except EngineError:
            raise
        except Exception as error:  # model errors vary and may hold paths: log them, never send them
            _log.exception("Transcription failed")
            raise EngineError("Transcription failed. Try again.", 500) from error

    @staticmethod
    def body(segments: list[dict[str, Any]], language: str, pcm: bytes) -> dict[str, Any]:
        return {"text": " ".join(s["text"] for s in segments if s["text"]), "language": language,
                "segments": [{k: s[k] for k in ("start_s", "end_s", "text")} for s in segments],
                "words": [w for s in segments for w in s["words"]],
                "duration_s": round(len(pcm) / PCM_BYTES_PER_SECOND, 2)}

    async def transcribe(self, pcm: bytes, language: str = "en", beam_size: int = 1, words: bool = True) -> dict[str, Any]:
        stop = threading.Event()
        async with self._gate:
            def run() -> dict[str, Any]:
                events = list(self._events(pcm, language, beam_size, words, stop))
                return self.body([v for k, v in events if k == "segment"], events[-1][1]["language"], pcm)

            return await asyncio.to_thread(run)

    async def transcribe_stream(self, pcm: bytes, language: str = "en", beam_size: int = 1,
                                words: bool = True) -> AsyncIterator[tuple[str, Any]]:
        """("segment", dict) pieces, then ("done", {"language"}). Holds the slot until the last piece; when the caller
        closes the iterator the transcription is stopped."""
        stop = threading.Event()
        async with self._gate:
            events = self._events(pcm, language, beam_size, words, stop)
            try:
                while (event := await asyncio.to_thread(next, events, None)) is not None:
                    yield event
            finally:
                stop.set()  # a running `next` then ends by itself; closing the generator here could race with it

    # Updates: the Hub revision of the weights, shared by all replicas through one models volume.

    async def cached_sha(self) -> str | None:
        return await asyncio.to_thread(self._cached_sha_of, repo_of(self.model))

    async def remote_sha(self) -> str:
        return await asyncio.to_thread(self._remote_sha_of, repo_of(self.model))

    async def pull(self) -> None:
        await asyncio.to_thread(self._download_of, repo_of(self.model))

    async def unload(self) -> None:
        """Forget the loaded model here and tell the other replicas to do the same; the next call loads the new weights."""
        if self._marker is not None:
            self._marker.parent.mkdir(parents=True, exist_ok=True)
            self._seen = uuid.uuid4().hex
            self._marker.write_text(self._seen, "utf-8")
        self._forget()

    def sync(self) -> None:
        """Called every second: drop the loaded model when another replica downloaded new weights."""
        current = self._read_marker()
        if current != self._seen:
            self._seen = current
            self._forget()

    def _forget(self) -> None:
        with self._load_lock:
            self._base = None
            self.generation += 1  # kept answers are from the old weights

    def _read_marker(self) -> str | None:
        try:
            return self._marker.read_text("utf-8") if self._marker else None
        except OSError:
            return None
