import argparse
import asyncio
import json
import queue
import statistics
import sys
import time
from dataclasses import dataclass

import sounddevice as sd
import websockets


DEFAULT_URL = "wss://silero-vad-rust-ort.onrender.com/vad"
SAMPLE_RATE = 16_000
CHUNK_SAMPLES = 512
CHUNK_BYTES = CHUNK_SAMPLES * 2


@dataclass
class SentChunk:
    index: int
    sent_at: float
    audio_end_ms: int


def list_devices() -> None:
    print(sd.query_devices())


def pcm_callback(audio_queue: queue.Queue, status):
    def callback(indata, frames, _time_info, callback_status):
        if callback_status:
            print(f"[audio] {callback_status}", file=sys.stderr)
        if status["dropped"]:
            return
        try:
            audio_queue.put_nowait(bytes(indata))
        except queue.Full:
            status["dropped"] += 1

    return callback


def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[min(index, len(ordered) - 1)]


def fmt_ms(value):
    return f"{value:7.2f} ms"


async def receiver(ws, pending, latencies, counters, started):
    async for message in ws:
        now = time.perf_counter()
        event = json.loads(message)
        event_type = event.get("type", "unknown")

        if event_type == "ready":
            print(
                f"[{now - started:8.2f}s] READY "
                f"sample_rate={event.get('sample_rate')} "
                f"chunk_samples={event.get('chunk_samples')} "
                f"format={event.get('format')}"
            )
            continue

        audio_time_ms = event.get("audio_time_ms")
        probability = event.get("probability")
        latency_ms = None

        if audio_time_ms is not None:
            matched_key = None
            matched_chunk = None
            for key, chunk in pending.items():
                if chunk.audio_end_ms >= audio_time_ms:
                    matched_key = key
                    matched_chunk = chunk
                    break
            if matched_key is not None:
                latency_ms = (now - matched_chunk.sent_at) * 1000.0
                latencies.append(latency_ms)
                stale = [key for key in pending if key <= matched_key]
                for key in stale:
                    pending.pop(key, None)

        counters[event_type] = counters.get(event_type, 0) + 1

        parts = [f"[{now - started:8.2f}s]", event_type.upper()]
        if audio_time_ms is not None:
            parts.append(f"audio={audio_time_ms}ms")
        if probability is not None:
            parts.append(f"prob={probability:.3f}")
        if latency_ms is not None:
            parts.append(f"e2e={fmt_ms(latency_ms)}")
        if event_type == "error":
            parts.append(f"message={event.get('message')}")
        print(" ".join(parts))


async def reporter(latencies, counters, sent_counter, dropped_status, started):
    last_sent = 0
    last_time = started
    while True:
        await asyncio.sleep(1.0)
        now = time.perf_counter()
        sent = sent_counter["value"]
        current_rate = (sent - last_sent) / max(now - last_time, 1e-9)
        total_rate = sent / max(now - started, 1e-9)
        last_sent = sent
        last_time = now

        if latencies:
            avg = statistics.fmean(latencies)
            p95 = percentile(latencies, 0.95)
            worst = max(latencies)
            latency_text = (
                f"lat avg={fmt_ms(avg)} p95={fmt_ms(p95)} max={fmt_ms(worst)}"
            )
        else:
            latency_text = "lat avg=waiting p95=waiting max=waiting"

        print(
            f"[{now - started:8.2f}s] SPEED "
            f"sent={sent} current={current_rate:5.1f} chunks/s "
            f"avg={total_rate:5.1f} chunks/s "
            f"{latency_text} "
            f"dropped_audio={dropped_status['dropped']} "
            f"events={counters}"
        )


async def run(args):
    audio_queue = queue.Queue(maxsize=args.queue_size)
    dropped_status = {"dropped": 0}
    pending = {}
    latencies = []
    counters = {}
    sent_counter = {"value": 0}
    started = time.perf_counter()

    stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_SAMPLES,
        channels=1,
        dtype="int16",
        device=args.device,
        callback=pcm_callback(audio_queue, dropped_status),
    )

    print(f"Connecting to {args.url}")
    print("Press Ctrl+C to stop.")
    print(
        f"audio: sample_rate={SAMPLE_RATE} chunk_samples={CHUNK_SAMPLES} "
        f"chunk_bytes={CHUNK_BYTES} chunk_ms=32"
    )

    async with websockets.connect(
        args.url,
        ping_interval=20,
        ping_timeout=20,
        max_size=2**20,
    ) as ws:
        recv_task = asyncio.create_task(receiver(ws, pending, latencies, counters, started))
        report_task = asyncio.create_task(
            reporter(latencies, counters, sent_counter, dropped_status, started)
        )

        with stream:
            while args.seconds <= 0 or time.perf_counter() - started < args.seconds:
                try:
                    chunk = await asyncio.to_thread(audio_queue.get, True, 1.0)
                except queue.Empty:
                    continue

                if len(chunk) != CHUNK_BYTES:
                    print(f"[warn] got {len(chunk)} bytes, expected {CHUNK_BYTES}")
                    continue

                await ws.send(chunk)
                sent_counter["value"] += 1
                audio_end_ms = sent_counter["value"] * 32
                pending[sent_counter["value"]] = SentChunk(
                    index=sent_counter["value"],
                    sent_at=time.perf_counter(),
                    audio_end_ms=audio_end_ms,
                )

        report_task.cancel()
        await ws.close()
        await recv_task


def parse_args():
    parser = argparse.ArgumentParser(
        description="Realtime microphone client for Render Silero VAD WebSocket"
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--device", default=None, help="Input device index or name")
    parser.add_argument("--seconds", type=float, default=0, help="Stop after N seconds")
    parser.add_argument("--queue-size", type=int, default=64)
    parser.add_argument("--list-devices", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list_devices:
        list_devices()
        return
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
