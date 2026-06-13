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


def blank_status():
    return {
        "ready": False,
        "state": "connecting",
        "last_event": "-",
        "last_audio_ms": None,
        "last_probability": None,
        "last_latency_ms": None,
        "last_message": "",
    }


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


def write_status_line(text):
    width = 150
    sys.stdout.write("\r" + text[:width].ljust(width))
    sys.stdout.flush()


async def receiver(ws, pending, latencies, counters, status, started, verbose):
    async for message in ws:
        now = time.perf_counter()
        event = json.loads(message)
        event_type = event.get("type", "unknown")

        if event_type == "ready":
            status["ready"] = True
            status["state"] = "ready"
            status["last_event"] = "ready"
            status["last_message"] = (
                f"sample_rate={event.get('sample_rate')} "
                f"chunk_samples={event.get('chunk_samples')} "
                f"format={event.get('format')}"
            )
            if verbose:
                print(
                    f"\n[{now - started:8.2f}s] READY "
                    f"{status['last_message']}"
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
        status["last_event"] = event_type
        status["last_audio_ms"] = audio_time_ms
        status["last_probability"] = probability
        status["last_latency_ms"] = latency_ms
        status["last_message"] = event.get("message") or ""

        if event_type == "silence":
            status["state"] = "silence"
        elif event_type == "speech_start":
            status["state"] = "speech"
        elif event_type == "speech_end":
            status["state"] = "silence"
        elif event_type == "error":
            status["state"] = "error"

        parts = [f"[{now - started:8.2f}s]", event_type.upper()]
        if audio_time_ms is not None:
            parts.append(f"audio={audio_time_ms}ms")
        if probability is not None:
            parts.append(f"prob={probability:.3f}")
        if latency_ms is not None:
            parts.append(f"e2e={fmt_ms(latency_ms)}")
        if event_type == "error":
            parts.append(f"message={event.get('message')}")
        if verbose or event_type in {"speech_start", "speech_end", "error"}:
            print("\n" + " ".join(parts))


async def reporter(latencies, counters, sent_counter, dropped_status, status, started, interval):
    last_sent = 0
    last_time = started
    while True:
        await asyncio.sleep(interval)
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
            avg = p95 = worst = None

        last_latency = status["last_latency_ms"]
        last_latency_text = fmt_ms(last_latency) if last_latency is not None else "waiting"
        avg_text = fmt_ms(avg) if avg is not None else "waiting"
        p95_text = fmt_ms(p95) if p95 is not None else "waiting"
        max_text = fmt_ms(worst) if worst is not None else "waiting"
        probability = status["last_probability"]
        probability_text = f"{probability:.3f}" if probability is not None else "-"
        audio_text = (
            f"{status['last_audio_ms']}ms"
            if status["last_audio_ms"] is not None
            else "-"
        )

        line = (
            f"[{now - started:8.2f}s] SPEED "
            f"state={status['state']:<8} event={status['last_event']:<12} "
            f"audio={audio_text:<8} prob={probability_text:<5} "
            f"last={last_latency_text} avg={avg_text} p95={p95_text} max={max_text} "
            f"send={current_rate:5.1f}/s total={total_rate:5.1f}/s "
            f"dropped_audio={dropped_status['dropped']} "
            f"events={counters}"
        )
        write_status_line(line)


async def run(args):
    audio_queue = queue.Queue(maxsize=args.queue_size)
    dropped_status = {"dropped": 0}
    pending = {}
    latencies = []
    counters = {}
    status = blank_status()
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
        recv_task = asyncio.create_task(
            receiver(ws, pending, latencies, counters, status, started, args.verbose)
        )
        report_task = asyncio.create_task(
            reporter(
                latencies,
                counters,
                sent_counter,
                dropped_status,
                status,
                started,
                args.refresh,
            )
        )

        with stream:
            while args.seconds <= 0 or time.perf_counter() - started < args.seconds:
                try:
                    chunk = await asyncio.to_thread(audio_queue.get, True, 1.0)
                except queue.Empty:
                    continue

                if len(chunk) != CHUNK_BYTES:
                    print(f"\n[warn] got {len(chunk)} bytes, expected {CHUNK_BYTES}")
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
        print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Realtime microphone client for Render Silero VAD WebSocket"
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--device", default=None, help="Input device index or name")
    parser.add_argument("--seconds", type=float, default=0, help="Stop after N seconds")
    parser.add_argument("--queue-size", type=int, default=64)
    parser.add_argument("--refresh", type=float, default=0.25, help="Status refresh seconds")
    parser.add_argument("--verbose", action="store_true", help="Print every event")
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
