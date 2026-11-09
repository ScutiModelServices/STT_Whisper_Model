import pytest
from fastapi.testclient import TestClient

import app as app_module
import settings
import variants
from app import create_app
from conftest import make_engine
from options import SETTINGS
from tokens import TokenStore

ADMIN = {"X-Internal-Token": "admin"}
GPU = {"name": "Test GPU", "vram_total_mb": 6000, "vram_free_mb": 5000, "compute_capability": "8.9"}


def hw(ram=8000, gpu=GPU):
    return {"cpu_cores": 8, "ram_total_mb": 16000, "ram_available_mb": ram, "gpu": gpu}


def rows(hardware=None, current=("base.en", "cpu", "int8"), cuda_error=None, downloaded=frozenset()):
    return variants.listing(hardware or hw(), current, cuda_error, set(downloaded))


def by_id(rows_, variant_id):
    return variants.find(rows_, variant_id)


def test_estimate_scales_with_the_number_format():
    assert variants.estimate_mb(244, "int8", "cpu") < variants.estimate_mb(244, "float32", "cpu")
    assert variants.estimate_mb(244, "float16", "cuda") == round(244 * 2 * 1.2 + 450)


def test_a_cpu_variant_that_needs_more_than_free_memory_does_not_fit():
    result = rows(hw(ram=1000))
    assert by_id(result, "tiny.en|cpu|int8")["fits"] is True
    large = by_id(result, "large-v3|cpu|float32")
    assert large["fits"] is False and "1000 MB is free" in large["reason"]


def test_gpu_variants_need_a_gpu_and_working_libraries():
    assert by_id(rows(hw(gpu=None)), "small.en|cuda|float16")["fits"] is False
    broken = by_id(rows(cuda_error="The CUDA library cublas64_12.dll is missing."), "small.en|cuda|float16")
    assert broken["fits"] is False and "cublas64_12.dll" in broken["reason"]
    assert by_id(rows(), "small.en|cuda|float16")["fits"] is True


def test_gpu_memory_limit_uses_free_vram():
    small_gpu = dict(GPU, vram_free_mb=1000)
    result = rows(hw(gpu=small_gpu))
    assert by_id(result, "base.en|cuda|float16")["fits"] is True
    assert by_id(result, "large-v3|cuda|float16")["fits"] is False


def test_unknown_free_memory_claims_nothing():
    assert by_id(rows(hw(ram=None)), "tiny.en|cpu|int8")["fits"] is None


def test_the_variant_in_use_always_fits_and_is_marked():
    result = rows(hw(ram=10), current=("large-v3", "cpu", "float32"))
    current = by_id(result, "large-v3|cpu|float32")
    assert current["current"] is True and current["fits"] is True


def test_downloaded_flag_follows_the_model():
    result = rows(downloaded={"small.en"})
    assert by_id(result, "small.en|cpu|int8")["downloaded"] is True
    assert by_id(result, "medium.en|cpu|int8")["downloaded"] is False


def test_turbo_is_a_full_repository_name_and_multilingual():
    row = [r for r in rows() if r["name"] == "large-v3-turbo"][0]
    assert row["model"].startswith("mobiuslabsgmbh/") and row["english_only"] is False


@pytest.fixture
def served(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module.hardware, "probe", lambda: hw(ram=3000, gpu=None))
    store = TokenStore(tmp_path / "t.json")
    config = settings.Config(store.settings_path, SETTINGS)
    config.set({"update_check_hours": 0, "warm": False})
    engine = make_engine(model="base.en")
    return TestClient(create_app(engine, "admin", tokens=store, config=config)), config


def test_variants_endpoint_is_admin_only_and_reports_the_machine(served):
    client, _ = served
    assert client.get("/admin/variants").status_code == 401
    body = client.get("/admin/variants", headers=ADMIN).json()
    assert body["hardware"]["ram_available_mb"] == 3000 and "not measured" in body["estimate"]
    assert any(r["current"] and r["id"] == "base.en|cpu|int8" for r in body["variants"])


def test_choosing_a_variant_saves_model_device_and_format(served):
    client, config = served
    reply = client.put("/admin/variant", json={"id": "small.en|cpu|float32"}, headers=ADMIN)
    assert reply.status_code == 200 and reply.json()["restart_required"] is True
    assert (config.get("model"), config.get("device"), config.get("compute_type")) == ("small.en", "cpu", "float32")


def test_choosing_a_variant_that_does_not_fit_is_refused_and_saves_nothing(served):
    client, config = served
    reply = client.put("/admin/variant", json={"id": "small.en|cuda|float16"}, headers=ADMIN)
    assert reply.status_code == 409 and "GPU" in reply.json()["detail"]
    assert config.get("model") == "base.en" and config.get("device") == "cpu"
    assert client.put("/admin/variant", json={"id": "nope"}, headers=ADMIN).status_code == 404


def test_missing_gpu_libraries_give_the_install_hint_not_a_crash(monkeypatch):
    import sys
    import types
    import engine

    monkeypatch.setattr(engine, "_cuda_checked", [])
    fake = types.SimpleNamespace(get_cuda_device_count=lambda: 1)
    monkeypatch.setitem(sys.modules, "ctranslate2", fake)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name)))
    monkeypatch.setattr("ctypes.CDLL", lambda name, *a, **k: (_ for _ in ()).throw(OSError(name)))
    assert "uv sync --extra gpu" in engine.cuda_error()
