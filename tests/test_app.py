import asyncio
import json
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient

import settings
from app import create_app
from conftest import PCM, make_engine
from engine import EngineError
from options import SETTINGS
from tokens import TokenStore

URL = "/v1/transcribe"


def client(engine=None, admin_password=None, **options):
    return TestClient(create_app(engine or make_engine(), admin_password, **options))


def post(c, query="", pcm=PCM, **kwargs):
    return c.post(URL + query, content=pcm, **kwargs)


def events(reply):
    return [json.loads(row) for row in reply.text.splitlines()]


def test_each_project_gets_its_own_revocable_token(tmp_path):
    c = client(admin_password="admin", tokens=TokenStore(tmp_path / "t.json"))
    admin = {"X-Internal-Token": "admin"}
    assert post(c).status_code == 401
    made = c.post("/admin/tokens", json={"name": "scribe", "rate": 1}, headers=admin).json()
    use = {"X-Internal-Token": made["token"]}
    assert made["token"].startswith("whisper_") and post(c, headers=use).status_code == 200
    assert post(c, headers=use).status_code == 429  # one call a minute
    assert c.delete(f"/admin/tokens/{made['id']}", headers=admin).status_code == 200
    assert post(c, headers=use).status_code == 401
    assert c.get("/health").status_code == 200  # no token needed


def test_threads_apply_now_and_are_saved(tmp_path):
    engine = make_engine()
    c = client(engine, "admin", tokens=TokenStore(tmp_path / "t.json"))
    admin = {"X-Internal-Token": "admin"}
    assert c.put("/admin/threads", json={"num_thread": 0}, headers=admin).status_code == 422
    assert c.put("/admin/threads", json={"num_thread": 3}, headers=admin).json() == {"num_thread": 3}
    assert engine.num_thread == 3 and c.get("/health").json()["num_thread"] == 3
    assert settings.load(tmp_path / "settings.json")["num_thread"] == 3


def test_changing_threads_drops_the_loaded_model_so_the_next_call_reloads_it():
    engine = make_engine()
    c = client(engine)
    post(c)
    engine.num_thread = 2
    assert c.get("/health").json()["ready"] is False
    post(c)
    assert engine.loads == ["m", "m"]


def test_settings_are_admin_only_checked_and_apply_live(tmp_path):
    store = TokenStore(tmp_path / "t.json")
    config = settings.Config(store.settings_path, SETTINGS)
    config.set({"update_check_hours": 0, "warm": False})
    c = client(admin_password="admin", tokens=store, config=config)
    admin = {"X-Internal-Token": "admin"}
    assert c.get("/admin/settings").status_code == 401
    assert c.put("/admin/settings", json={"values": {"max_inflight": -1}}, headers=admin).status_code == 422
    assert c.put("/admin/settings", json={"values": {"max_inflight": 7}}, headers=admin).status_code == 200
    saved = {item["key"]: item for item in c.get("/admin/settings", headers=admin).json()["settings"]}
    assert saved["max_inflight"]["value"] == 7
    assert client(admin_password="admin").put("/admin/settings", json={"values": {}}, headers=admin).status_code == 409


def test_what_another_replica_saved_reaches_this_one(tmp_path):
    store = TokenStore(tmp_path / "t.json")
    config = settings.Config(store.settings_path, SETTINGS)
    config.set({"update_check_hours": 0, "warm": False})
    engine = make_engine()
    with TestClient(create_app(engine, "admin", tokens=store, config=config)):
        settings.save(store.settings_path, num_thread=5, vad_filter=True)
        for _ in range(300):  # the loop looks every second
            if engine.vad_filter:
                break
            time.sleep(0.02)
    assert engine.num_thread == 5 and engine.vad_filter is True


def test_model_dropped_by_a_saved_setting_loads_again_without_a_request(tmp_path):
    store = TokenStore(tmp_path / "t.json")
    config = settings.Config(store.settings_path, SETTINGS)
    config.set({"update_check_hours": 0, "warm": True})
    engine = make_engine()
    with TestClient(create_app(engine, "admin", tokens=store, config=config)) as c:
        settings.save(store.settings_path, num_thread=5)
        for _ in range(300):
            if engine.num_thread == 5 and len(engine.loads) == 2:
                break
            time.sleep(0.02)
        assert len(engine.loads) == 2
        assert c.get("/health").json()["ready"] is True


def test_startup_warms_the_model():
    engine = make_engine()
    with TestClient(create_app(engine, warm=True)) as c:
        for _ in range(100):
            if c.get("/health").json()["ready"]:
                break
            time.sleep(0.02)
        assert engine.loads == ["m"] and c.get("/health").json()["ready"] is True


def test_health_answers_while_the_model_is_still_loading():
    release = threading.Event()

    def slow(source):
        release.wait(5)
        return make_engine()._loader(source)

    with TestClient(create_app(make_engine(loader=slow), warm=True)) as c:
        assert c.get("/health").json()["ready"] is False
        release.set()


def test_warmup_failure_is_logged_without_stopping_service(caplog):
    def failing(source):
        raise RuntimeError("no weights")

    with TestClient(create_app(make_engine(loader=failing), warm=True)) as c:
        assert c.get("/health").status_code == 200
        for _ in range(100):
            if "warm-up failed" in caplog.text:
                break
            time.sleep(0.02)
    assert "warm-up failed" in caplog.text


def test_warmup_is_optional():
    engine = make_engine()
    with TestClient(create_app(engine)):
        assert engine.loads == []


def test_main_builds_the_engine_from_settings(monkeypatch, tmp_path):
    import app as module

    monkeypatch.setenv("SERVICE_TOKENS_FILE", str(tmp_path / "tokens.json"))
    monkeypatch.setenv("SERVICE_NUM_THREAD", "2")
    monkeypatch.setenv("SERVICE_PORT", "9999")
    seen, started = [], {}
    monkeypatch.setattr(module.WhisperEngine, "num_thread", property(lambda self: None, lambda self, count: seen.append(count)))
    monkeypatch.setattr("uvicorn.Server.run", lambda self: started.update(app=self.config.app, host=self.config.host, port=self.config.port))
    module.main()
    assert started["port"] == 9999 and started["host"] == "127.0.0.1" and seen == [2]


def test_transcribe_returns_text_segments_words_and_duration():
    engine = make_engine()
    body = post(client(engine)).json()
    assert body["text"] == "Hello world. Bye now." and body["language"] == "en" and body["model"] == "m"
    assert body["segments"] == [{"start_s": 0.0, "end_s": 1.0, "text": "Hello world."},
                                {"start_s": 1.0, "end_s": 2.0, "text": "Bye now."}]
    assert [w["text"] for w in body["words"]] == ["Hello", "world.", "Bye", "now."]
    assert body["duration_s"] == 1.0 and "duration_ms" in body
    assert engine._base.calls[0] == {"bytes": len(PCM), "language": "en", "beam_size": 1, "words": True, "vad": False}


def test_options_reach_the_model():
    engine = make_engine()
    c = client(engine)
    body = post(c, "?language=auto&beam_size=5&word_timestamps=false").json()
    assert engine._base.calls[0]["language"] is None and engine._base.calls[0]["beam_size"] == 5
    assert body["words"] == []


@pytest.mark.parametrize("query", ["?language=English", "?beam_size=0", "?beam_size=11", "?temperature=0.5", "?stream=maybe"])
def test_options_are_validated(query):
    assert post(client(), query).status_code == 422


def test_the_body_must_be_whole_samples_and_not_empty():
    c = client()
    assert post(c, pcm=b"").status_code == 422
    assert post(c, pcm=b"\x01\x00\x02").status_code == 422


def test_audio_over_the_limit_is_refused_before_it_is_transcribed():
    engine = make_engine()
    c = client(engine)
    assert post(c, pcm=b"\x00\x00" * 16000 * 151).status_code == 413  # default max_seconds is 150
    assert engine.loads == []


def test_an_english_only_model_refuses_other_languages():
    c = client(make_engine(model="base.en"))
    assert post(c, "?language=de").status_code == 422
    assert post(c, "?language=en").status_code == 200


def test_an_empty_transcript_is_valid():
    body = post(client(make_engine(segments=()))).json()
    assert body["text"] == "" and body["segments"] == [] and body["words"] == []


@pytest.mark.parametrize("failure,status", [(EngineError("bad", 400), 422), (EngineError("broken", 500), 500),
                                            (RuntimeError("boom"), 500)])
def test_engine_failures_map_to_http_errors(failure, status):
    reply = post(client(make_engine(fail=failure)))
    assert reply.status_code == status
    assert "boom" not in reply.text


def test_model_errors_never_leak_their_text():
    reply = post(client(make_engine(fail=RuntimeError("cannot open /secret/path"))))
    assert reply.status_code == 500 and "secret" not in reply.text


def test_stream_sends_one_line_per_segment_then_the_same_body_as_a_call_without_stream():
    reply = post(client(), "?stream=true")
    sent = events(reply)
    assert reply.headers["content-type"].startswith("application/x-ndjson") and reply.headers["X-Cache"] == "miss"
    assert [e["type"] for e in sent] == ["segment", "segment", "done"]
    assert sent[0] == {"type": "segment", "start_s": 0.0, "end_s": 1.0, "text": "Hello world."}
    plain = post(client()).json()
    assert {k: v for k, v in sent[-1].items() if k not in ("type", "duration_ms")} == {k: v for k, v in plain.items() if k != "duration_ms"}


def test_stream_errors_before_the_first_segment_are_http_errors_and_later_ones_are_error_lines():
    assert post(client(make_engine(fail=EngineError("bad", 400))), "?stream=true").status_code == 422
    assert post(client(make_engine(fail=EngineError("broken", 500))), "?stream=true").status_code == 500
    cut = post(client(make_engine(fail_after=1)), "?stream=true")
    sent = events(cut)
    assert cut.status_code == 200 and sent[0]["type"] == "segment" and sent[-1]["type"] == "error"
    assert "secret" not in cut.text


def test_the_in_flight_cap_answers_429_with_retry_after():
    async def run():
        engine = make_engine()
        real = engine.transcribe

        async def slow(*args, **kwargs):
            await asyncio.sleep(0.2)
            return await real(*args, **kwargs)

        engine.transcribe = slow
        app = create_app(engine, max_inflight=1)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as http:
            first = asyncio.create_task(http.post(URL, content=PCM))
            await asyncio.sleep(0.05)
            second = await http.post(URL, content=PCM)
            assert second.status_code == 429 and second.headers["Retry-After"] == "5"
            assert (await first).status_code == 200

    asyncio.run(run())


def test_the_same_audio_and_options_are_answered_from_the_cache():
    engine = make_engine()
    c = client(engine, cache_size=4)
    first, second = post(c), post(c)
    assert (first.headers["X-Cache"], second.headers["X-Cache"]) == ("miss", "hit")
    assert first.json()["text"] == second.json()["text"] and len(engine._base.calls) == 1
    post(c, "?beam_size=2")  # any change in the options is another answer
    post(c, pcm=b"\x02\x00" * 16000)
    assert len(engine._base.calls) == 3
    off = client(make_engine(), cache_size=0)
    assert [post(off).headers["X-Cache"] for _ in range(2)] == ["miss", "miss"]


def test_new_weights_empty_the_cache():
    engine = make_engine()
    c = client(engine, cache_size=4)
    post(c)
    assert post(c).headers["X-Cache"] == "hit"
    asyncio.run(engine.unload())
    assert post(c).headers["X-Cache"] == "miss"


def test_stream_fills_the_cache_and_a_kept_answer_streams_back_segment_by_segment():
    c = client(cache_size=4)
    streamed = post(c, "?stream=true")
    kept = post(c, "?stream=true")
    assert kept.headers["X-Cache"] == "hit"
    sent = events(kept)
    assert [e["type"] for e in sent] == ["segment", "segment", "done"]
    assert sent[-1]["text"] == events(streamed)[-1]["text"]
    assert post(c).headers["X-Cache"] == "hit"  # the same cache serves a call without stream


def test_a_kept_answer_is_not_served_without_a_token(tmp_path):
    c = client(admin_password="admin", tokens=TokenStore(tmp_path / "t.json"), cache_size=4)
    made = c.post("/admin/tokens", json={"name": "p"}, headers={"X-Internal-Token": "admin"}).json()
    use = {"X-Internal-Token": made["token"]}
    assert [post(c, headers=use).headers["X-Cache"] for _ in range(2)] == ["miss", "hit"]
    assert post(c).status_code == 401
