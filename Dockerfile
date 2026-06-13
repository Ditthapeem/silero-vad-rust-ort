FROM rust:1-bookworm AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release

FROM debian:bookworm-slim AS runtime

ARG ORT_VERSION=1.22.1
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl unzip \
    && curl -fsSL -o /tmp/onnxruntime.tgz \
        "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" \
    && mkdir -p /opt/onnxruntime \
    && tar -xzf /tmp/onnxruntime.tgz -C /opt/onnxruntime --strip-components=1 \
    && rm -rf /var/lib/apt/lists/* /tmp/onnxruntime.tgz

COPY --from=builder /app/target/release/rust-ort-vad /usr/local/bin/rust-ort-vad
COPY models ./models

ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib
ENV RUST_LOG=rust_ort_vad=info

CMD rust-ort-vad \
    --host 0.0.0.0 \
    --port "${PORT:-8080}" \
    --ort-lib /opt/onnxruntime/lib/libonnxruntime.so.1.22.1 \
    --model models/silero_vad.onnx \
    --sample-rate 16000 \
    --threads 1 \
    --max-connections 3 \
    --threshold 0.45 \
    --min-silence-ms 50 \
    --speech-pad-ms 10
