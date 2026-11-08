"""Pin the external faster-whisper loader contract independently of inference fakes."""

import sys
from types import ModuleType

from engine import Loaded, load


def test_load_uses_external_whisper_model_class_and_cached_weights(monkeypatch):
    calls = []
    snapshots = []
    faster_whisper = ModuleType("faster_whisper")
    hub = ModuleType("huggingface_hub")

    class WhisperModel:
        def __init__(self, path, **options):
            calls.append((path, options))

    def snapshot_download(**options):
        snapshots.append(options)
        return "cached-model"

    faster_whisper.WhisperModel = WhisperModel
    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "faster_whisper", faster_whisper)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    loaded = load("base.en", "int8", 4)

    assert isinstance(loaded, Loaded)
    assert isinstance(loaded.model, WhisperModel)
    assert calls == [("cached-model", {"device": "cpu", "compute_type": "int8", "cpu_threads": 4, "local_files_only": True})]
    assert snapshots == [{"repo_id": "Systran/faster-whisper-base.en", "local_files_only": True, "token": False}]
