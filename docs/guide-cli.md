# Test it from the command line

Calls Whisper with PowerShell only, no admin page. Port 29300. Replace `<...>` values. The commands were written from the code and not yet run against a live server.

## Get a token

Every call except `/health` needs a token while `SERVICE_ADMIN_PASSWORD` is set. Admin calls take the admin password itself; a token works only on `/v1/*`. Create one token and reuse it. Each `POST /admin/tokens` call makes a new one, so do not put it in a loop. Revoke extras on the admin page.

```powershell
$base = "http://127.0.0.1:29300"
$pw   = @{ "X-Internal-Token" = "<SERVICE_ADMIN_PASSWORD>" }
$tok  = (Invoke-RestMethod -Method Post -Uri "$base/admin/tokens" -Headers $pw -ContentType "application/json" -Body '{"name":"cli-test"}').token
$tok
$h    = @{ "X-Internal-Token" = $tok }
Invoke-RestMethod "$base/health"
```

`$pw` (the admin password) is for `/admin/*` only. `$h` (the token) is for `/v1/*` only; using the password there gives `Invalid token`. `$tok` prints once: save it to skip this step next time, then set `$h` from it. With no admin password set there is no auth: skip `-Headers $h`.

## Call it

Convert the audio to raw 16 kHz mono 16-bit little-endian PCM, then send the file as the body. Convert to a file: PowerShell pipes would damage the bytes.

```powershell
ffmpeg -y -loglevel error -i "<speech.wav>" -f s16le -ar 16000 -ac 1 "$env:TEMP\speech.pcm"
Invoke-RestMethod -Method Post -Uri "$base/v1/transcribe?language=en" -Headers $h -ContentType "application/octet-stream" -InFile "$env:TEMP\speech.pcm"
```

Query options: `language`, `beam_size` (1 to 10), `word_timestamps`, `stream`. `language=auto` needs a multilingual model.

## Problems

| Answer | Cause |
|---|---|
| `401` | Wrong value: admin routes need the admin password, `/v1/*` needs a token. |
| `403` | The token's *projects* list does not include what you asked for. |
| `429` | Token rate used up or too many calls waiting; wait `Retry-After` seconds. |
| `422` | A field is unknown or out of range; the answer says which. |
| connection refused | Not running, or another port; see the run-path guides. |
