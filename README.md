# Rust ORT Silero VAD Service

Low-resource WebSocket VAD service for VPS deployment.

## Runtime

- Rust
- Official ONNX Runtime through the `ort` crate
- Silero VAD ONNX model
- WebSocket endpoint for long realtime sessions

## Endpoints

```text
GET /health
GET /vad
```

`/vad` expects binary WebSocket frames:

```text
format: pcm_s16le_mono
sample_rate: 16000
chunk_samples: 512
chunk_bytes: 1024
chunk_duration: 32 ms
```

The server sends JSON events:

```json
{"type":"ready","sample_rate":16000,"chunk_samples":512,"format":"pcm_s16le_mono"}
{"type":"speech_start","audio_time_ms":44870,"probability":0.62}
{"type":"speech_end","audio_time_ms":45050,"probability":0.019}
```

## Run

```cmd
cargo run --release -- --ort-lib "C:\path\to\onnxruntime.dll" --model models\silero_vad.onnx
```

Model:

```text
models/silero_vad.onnx  original ONNX model, 2.33 MB
```

Linux VPS example:

```bash
./rust-ort-vad \
  --host 0.0.0.0 \
  --port 8080 \
  --ort-lib /opt/onnxruntime/lib/libonnxruntime.so.1.26.0 \
  --model models/silero_vad.onnx \
  --threads 1 \
  --max-connections 8
```

## Notes

Keep one WebSocket connection open per realtime audio session. Send 512-sample PCM chunks every 32 ms. Do not send WAV, MP3, base64, or JSON audio payloads.
