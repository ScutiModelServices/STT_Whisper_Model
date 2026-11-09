"""The Whisper variants an admin can choose: model size x number format x device, each with an estimate of what it takes,
and whether this machine can run it.

The estimates are arithmetic, not measurements. They were compared with real use on the RTX 4050 Laptop (2026-11-09,
10 s of generated noise as audio; GPU memory from `nvidia-smi`, process memory from the working set), after the first
overheads proved 2 to 2.5 times too high. Measured / estimated, in MB:
`small.en` GPU `float16` 660 / 836, `int8_float16` 447 / 543, CPU `int8` 317 / 443;
`medium.en` GPU `float16` 1979 / 2096, `int8_float16` 1109 / 1173, CPU `int8` 820 / 1073; `base.en` GPU `float16` 251 / 428.
Every estimate is at or above the measured use. `large` models and long audio were not measured.

The estimates are arithmetic: the weights at the format's bytes per parameter, times 1.2 for
activations, plus a fixed overhead (the runtime on CPU, the CUDA context on GPU). They say "about this much" and are
shown as estimates on the admin page. The disk size is the real size of the Hub files (CTranslate2 float16).

shortcut: free memory is read when the page is opened; two services starting at once can both see the same free memory."""

from __future__ import annotations

from typing import Any

# (model name or Hub repository, parameters in millions, disk MB of the downloaded files)
MODELS = (
    ("tiny.en", 39, 74),
    ("base.en", 74, 145),
    ("small.en", 244, 463),
    ("medium.en", 769, 1459),
    ("mobiuslabsgmbh/faster-whisper-large-v3-turbo", 809, 1546),
    ("large-v3", 1550, 2947),
)
FORMATS = {"cpu": ("int8", "float32"), "cuda": ("int8_float16", "float16")}
BYTES_PER_PARAMETER = {"int8": 1, "int8_float16": 1, "float16": 2, "float32": 4}
OVERHEAD_MB = {"cpu": 150, "cuda": 250}
FREE_SHARE = {"cpu": 0.85, "cuda": 0.9}  # keep some memory for the rest of the machine


def label(model: str) -> str:
    return model.rsplit("/", 1)[-1].replace("faster-whisper-", "")


def estimate_mb(parameters_m: int, compute_type: str, device: str) -> int:
    return round(parameters_m * BYTES_PER_PARAMETER[compute_type] * 1.2 + OVERHEAD_MB[device])


def fit(device: str, need_mb: int, hardware: dict[str, Any], cuda_error: str | None) -> tuple[bool | None, str]:
    """(fits, reason). `None` = the free memory could not be read, so nothing is claimed."""
    if device == "cuda":
        gpu = hardware.get("gpu")
        if gpu is None:
            return False, "No NVIDIA GPU is visible to this service."
        if cuda_error:
            return False, cuda_error
        limit = gpu["vram_free_mb"] * FREE_SHARE["cuda"]
        return (need_mb <= limit, "" if need_mb <= limit else f"Needs about {need_mb} MB of GPU memory; {gpu['vram_free_mb']} MB is free.")
    free = hardware.get("ram_available_mb")
    if free is None:
        return None, "Free memory could not be read."
    limit = free * FREE_SHARE["cpu"]
    return (need_mb <= limit, "" if need_mb <= limit else f"Needs about {need_mb} MB of memory; {free} MB is free.")


def listing(hardware: dict[str, Any], current: tuple[str, str, str], cuda_error: str | None,
            downloaded: set[str]) -> list[dict[str, Any]]:
    """Every variant, smallest first. `current` is (model, device, compute_type) in use; it always fits, because the
    memory it holds already counts as used. `downloaded` holds the models whose weights are on disk."""
    rows = []
    for model, parameters, disk_mb in MODELS:
        for device, formats in FORMATS.items():
            for compute_type in formats:
                need = estimate_mb(parameters, compute_type, device)
                in_use = (model, device, compute_type) == current
                fits, reason = (True, "") if in_use else fit(device, need, hardware, cuda_error)
                where = "GPU" if device == "cuda" else "CPU"
                downloaded_text = " · downloaded" if model in downloaded else ""
                english = " · English only" if model.endswith(".en") else ""
                rows.append({
                    "title": f"{label(model)} · {compute_type} · {where}",
                    "detail": f"about {need} MB {'GPU memory' if device == 'cuda' else 'memory'} · {disk_mb} MB download{downloaded_text}{english}",
                    "id": f"{model}|{device}|{compute_type}", "model": model, "name": label(model), "device": device,
                    "compute_type": compute_type, "memory_kind": "GPU memory" if device == "cuda" else "memory",
                    "estimated_mb": need, "disk_mb": disk_mb, "fits": fits, "reason": reason,
                    "downloaded": model in downloaded, "current": in_use,
                    "english_only": model.endswith(".en"),
                })
    return sorted(rows, key=lambda row: (row["estimated_mb"], row["device"]))


def find(rows: list[dict[str, Any]], variant_id: str) -> dict[str, Any] | None:
    return next((row for row in rows if row["id"] == variant_id), None)
