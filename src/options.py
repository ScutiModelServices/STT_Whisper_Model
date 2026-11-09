"""The options the admin page can change. Each can still start from its environment variable; a saved value wins."""

from engine import COMPUTE_TYPES, DEFAULT_MODEL, DEVICES
from settings import Setting

SETTINGS = [
    Setting("update_check_hours", "Check for updates every (hours)", "float", 6.0, "SERVICE_UPDATE_CHECK_HOURS", 0, 720,
            help="0 = only when you press Check now"),
    Setting("max_inflight", "Calls allowed to wait for the model", "int", 0, "SERVICE_MAX_INFLIGHT", 0, 10000,
            help="Per replica; more get 429. 0 = 4"),
    Setting("cache_size", "Answers kept per replica", "int", 256, "SERVICE_CACHE_SIZE", 0, 100000,
            help="The last answers for the same audio and options, sent again without transcribing. Decoding is deterministic, so every request is kept. 0 = off"),
    Setting("max_seconds", "Longest audio per call (seconds)", "int", 150, "SERVICE_MAX_SECONDS", 1, 900,
            help="Longer audio gets 413. Keep it under what the caller's own timeout allows at the model's speed"),
    Setting("vad_filter", "Skip silence before decoding", "bool", False, "SERVICE_VAD_FILTER",
            help="Voice activity filter. Off matches the old in-process Scribe"),
    Setting("model", "Base model", "text", DEFAULT_MODEL, "SERVICE_MODEL", restart=True, blank=False,
            help="A faster-whisper name such as base.en or small.en, or a Hugging Face repository of CTranslate2 files. After a restart, download it from the Updates card"),
    Setting("compute_type", "Number format", "text", "int8", "SERVICE_COMPUTE_TYPE", restart=True, blank=False,
            help=f"One of {', '.join(COMPUTE_TYPES)}. int8 is the fastest on a CPU; float32 uses about four times the memory. float16 and int8_float16 are for a GPU"),
    Setting("device", "Device", "text", "cpu", "SERVICE_DEVICE", restart=True, blank=False,
            help=f"One of {', '.join(DEVICES)}. cuda needs an NVIDIA GPU and the GPU libraries (uv sync --extra gpu). Use the Model variant card, which only offers what this machine can run"),
    Setting("warm", "Load the model at start", "bool", True, "SERVICE_WARM", restart=True),
]
