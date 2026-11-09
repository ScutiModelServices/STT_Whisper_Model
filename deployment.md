# Deploying STT_Whisper_Model

STT_Whisper_Model is one HTTP service (port 29300). It runs as `stt_whisper_model` replicas behind an nginx load balancer,
`stt_whisper_model_lb`. Callers use `http://<host>:29300`; they never reach a replica directly.

First run requires [Updates → Download](docs/guide-web-ui.md#downloaded-weights) after opening `/admin/`; health and admin stay available before weights are downloaded.

## Ways to run it

Each path has its own guide. Pick one.

| Path | Guide |
|---|---|
| Docker Desktop on this PC | [guide-docker-same-device.md](docs/guide-docker-same-device.md) |
| Docker on another always-on machine, published with `run.bat` (Deploy) | [guide-docker-deployment-device.md](docs/guide-docker-deployment-device.md) |
| Directly on Windows, no Docker | [guide-no-docker-same-device.md](docs/guide-no-docker-same-device.md) |

Once it runs, the admin page is covered by [guide-web-ui.md](docs/guide-web-ui.md).
