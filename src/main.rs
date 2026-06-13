use anyhow::{Context, Result, bail};
use axum::{
    Json, Router,
    extract::{
        State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    http::StatusCode,
    response::IntoResponse,
    routing::get,
};
use futures_util::{SinkExt, StreamExt};
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use serde::Serialize;
use std::{
    env,
    net::SocketAddr,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::{net::TcpListener, sync::Mutex, time::timeout};
use tracing::{error, info, warn};

const PCM_I16_BYTES_PER_SAMPLE: usize = 2;

#[derive(Clone)]
struct Config {
    host: String,
    port: u16,
    model: String,
    ort_lib: String,
    sample_rate: i64,
    threshold: f32,
    min_silence_ms: i64,
    speech_pad_ms: i64,
    threads: usize,
    max_connections: usize,
    idle_timeout_secs: u64,
}

#[derive(Clone)]
struct AppState {
    config: Arc<Config>,
    session: Arc<Mutex<Session>>,
    active_connections: Arc<AtomicUsize>,
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    runtime: &'static str,
    active_connections: usize,
    max_connections: usize,
    sample_rate: i64,
    chunk_samples: usize,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum VadEvent {
    Ready {
        sample_rate: i64,
        chunk_samples: usize,
        format: &'static str,
    },
    SpeechStart {
        audio_time_ms: i64,
        probability: f32,
    },
    SpeechEnd {
        audio_time_ms: i64,
        probability: f32,
    },
    Silence {
        audio_time_ms: i64,
        probability: f32,
    },
    Error {
        message: String,
    },
}

struct VadStream {
    sample_rate: i64,
    block_size: usize,
    context_size: usize,
    threshold: f32,
    min_silence_samples: i64,
    speech_pad_samples: i64,
    current_sample: i64,
    temp_end: i64,
    triggered: bool,
    input: Array2<f32>,
    state: ArrayD<f32>,
    sr: Array1<i64>,
}

fn ort_err<E: std::fmt::Display>(error: E) -> anyhow::Error {
    anyhow::anyhow!(error.to_string())
}

fn parse_args() -> Result<Config> {
    let mut config = Config {
        host: "0.0.0.0".to_string(),
        port: 8080,
        model: "models/silero_vad.onnx".to_string(),
        ort_lib: "/opt/onnxruntime/lib/libonnxruntime.so.1.26.0".to_string(),
        sample_rate: 16000,
        threshold: 0.5,
        min_silence_ms: 100,
        speech_pad_ms: 30,
        threads: 1,
        max_connections: 8,
        idle_timeout_secs: 60,
    };

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--host" => config.host = args.next().context("--host requires a value")?,
            "--port" => config.port = args.next().context("--port requires a value")?.parse()?,
            "--model" => config.model = args.next().context("--model requires a value")?,
            "--ort-lib" => config.ort_lib = args.next().context("--ort-lib requires a value")?,
            "--sample-rate" => {
                config.sample_rate = args
                    .next()
                    .context("--sample-rate requires a value")?
                    .parse()?
            }
            "--threshold" => {
                config.threshold = args
                    .next()
                    .context("--threshold requires a value")?
                    .parse()?
            }
            "--min-silence-ms" => {
                config.min_silence_ms = args
                    .next()
                    .context("--min-silence-ms requires a value")?
                    .parse()?
            }
            "--speech-pad-ms" => {
                config.speech_pad_ms = args
                    .next()
                    .context("--speech-pad-ms requires a value")?
                    .parse()?
            }
            "--threads" => {
                config.threads = args.next().context("--threads requires a value")?.parse()?
            }
            "--max-connections" => {
                config.max_connections = args
                    .next()
                    .context("--max-connections requires a value")?
                    .parse()?
            }
            "--idle-timeout-secs" => {
                config.idle_timeout_secs = args
                    .next()
                    .context("--idle-timeout-secs requires a value")?
                    .parse()?
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => bail!("unknown argument: {arg}"),
        }
    }

    if config.sample_rate != 8000 && config.sample_rate != 16000 {
        bail!("--sample-rate must be 8000 or 16000");
    }
    if !(0.0..=1.0).contains(&config.threshold) {
        bail!("--threshold must be between 0 and 1");
    }
    if config.threads == 0 {
        bail!("--threads must be at least 1");
    }
    if config.max_connections == 0 {
        bail!("--max-connections must be at least 1");
    }

    Ok(config)
}

fn print_help() {
    println!(
        "Rust ORT Silero VAD WebSocket service\n\
\n\
Usage:\n\
  rust-ort-vad [options]\n\
\n\
Options:\n\
  --host <ip>                 Bind host, default 0.0.0.0\n\
  --port <port>               Bind port, default 8080\n\
  --model <path>              Path to silero_vad.onnx\n\
  --ort-lib <path>            Path to libonnxruntime.so / onnxruntime.dll\n\
  --sample-rate <8000|16000>  Input sample rate, default 16000\n\
  --threshold <float>         Speech threshold, default 0.5\n\
  --min-silence-ms <ms>       End speech after silence, default 100\n\
  --speech-pad-ms <ms>        Speech padding, default 30\n\
  --threads <n>               ONNX Runtime threads, default 1\n\
  --max-connections <n>       WebSocket connection cap, default 8\n\
  --idle-timeout-secs <n>     Close idle sockets, default 60\n\
\n\
WebSocket:\n\
  GET /vad\n\
  Send binary pcm_s16le mono frames of exactly chunk_samples samples.\n\
  At 16 kHz chunk_samples=512. At 8 kHz chunk_samples=256."
    );
}

fn build_session(config: &Config, model: &str) -> Result<Session> {
    Session::builder()
        .map_err(ort_err)?
        .with_optimization_level(GraphOptimizationLevel::Level2)
        .map_err(ort_err)?
        .with_intra_threads(config.threads)
        .map_err(ort_err)?
        .with_inter_threads(config.threads)
        .map_err(ort_err)?
        .commit_from_file(model)
        .map_err(ort_err)
}

fn block_size(sample_rate: i64) -> usize {
    if sample_rate == 16000 { 512 } else { 256 }
}

fn context_size(sample_rate: i64) -> usize {
    if sample_rate == 16000 { 64 } else { 32 }
}

impl VadStream {
    fn new(config: &Config) -> Self {
        let block_size = block_size(config.sample_rate);
        let context_size = context_size(config.sample_rate);

        Self {
            sample_rate: config.sample_rate,
            block_size,
            context_size,
            threshold: config.threshold,
            min_silence_samples: config.sample_rate * config.min_silence_ms / 1000,
            speech_pad_samples: config.sample_rate * config.speech_pad_ms / 1000,
            current_sample: 0,
            temp_end: 0,
            triggered: false,
            input: Array2::<f32>::zeros((1, context_size + block_size)),
            state: ArrayD::<f32>::zeros(IxDyn(&[2, 1, 128])),
            sr: Array1::<i64>::from_vec(vec![config.sample_rate]),
        }
    }

    fn handle_pcm_chunk(
        &mut self,
        session: &mut Session,
        payload: &[u8],
    ) -> Result<Option<VadEvent>> {
        let expected_bytes = self.block_size * PCM_I16_BYTES_PER_SAMPLE;
        if payload.len() != expected_bytes {
            bail!(
                "invalid chunk size: got {} bytes, expected {} bytes",
                payload.len(),
                expected_bytes
            );
        }

        let mut samples = Vec::with_capacity(self.context_size + self.block_size);
        samples.extend_from_slice(
            &self
                .input
                .as_slice()
                .context("input buffer is not contiguous")?
                [self.block_size..self.block_size + self.context_size],
        );
        for sample in payload.chunks_exact(2) {
            let value = i16::from_le_bytes([sample[0], sample[1]]);
            samples.push(value as f32 / i16::MAX as f32);
        }

        let input = Array2::from_shape_vec((1, self.context_size + self.block_size), samples)?;
        let input_tensor = TensorRef::from_array_view(input.view()).map_err(ort_err)?;
        let state_tensor = TensorRef::from_array_view(self.state.view()).map_err(ort_err)?;
        let sr_tensor = TensorRef::from_array_view(self.sr.view()).map_err(ort_err)?;

        let outputs = session
            .run(ort::inputs![
                "input" => input_tensor,
                "state" => state_tensor,
                "sr" => sr_tensor,
            ])
            .map_err(ort_err)?;

        let probability = {
            let output = outputs["output"]
                .try_extract_array::<f32>()
                .map_err(ort_err)?;
            output.iter().next().copied().unwrap_or(0.0)
        };
        let state_output = outputs["stateN"]
            .try_extract_array::<f32>()
            .map_err(ort_err)?;
        self.state = ArrayD::from_shape_vec(
            IxDyn(&[2, 1, 128]),
            state_output
                .as_slice()
                .context("state output buffer is not contiguous")?
                .to_vec(),
        )?;
        self.input = input;

        self.current_sample += self.block_size as i64;
        Ok(self.vad_event(probability).or_else(|| {
            if self.triggered {
                None
            } else {
                Some(VadEvent::Silence {
                    audio_time_ms: self.current_sample * 1000 / self.sample_rate,
                    probability,
                })
            }
        }))
    }

    fn vad_event(&mut self, probability: f32) -> Option<VadEvent> {
        if probability >= self.threshold && self.temp_end != 0 {
            self.temp_end = 0;
        }

        if probability >= self.threshold && !self.triggered {
            self.triggered = true;
            let start_sample =
                (self.current_sample - self.speech_pad_samples - self.block_size as i64).max(0);
            return Some(VadEvent::SpeechStart {
                audio_time_ms: start_sample * 1000 / self.sample_rate,
                probability,
            });
        }

        if probability < self.threshold - 0.15 && self.triggered {
            if self.temp_end == 0 {
                self.temp_end = self.current_sample;
            }
            if self.current_sample - self.temp_end >= self.min_silence_samples {
                let end_sample = self.temp_end + self.speech_pad_samples - self.block_size as i64;
                self.temp_end = 0;
                self.triggered = false;
                return Some(VadEvent::SpeechEnd {
                    audio_time_ms: end_sample * 1000 / self.sample_rate,
                    probability,
                });
            }
        }

        None
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            env::var("RUST_LOG")
                .unwrap_or_else(|_| "rust_ort_vad=info,tower_http=warn".to_string()),
        )
        .init();

    let config = Arc::new(parse_args()?);
    ort::init_from(&config.ort_lib).map_err(ort_err)?.commit();

    let session = build_session(&config, &config.model)?;

    let state = AppState {
        config: Arc::clone(&config),
        session: Arc::new(Mutex::new(session)),
        active_connections: Arc::new(AtomicUsize::new(0)),
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/vad", get(vad_socket))
        .with_state(state);

    let address: SocketAddr = format!("{}:{}", config.host, config.port).parse()?;
    let listener = TcpListener::bind(address).await?;
    info!(
        "VAD service listening on {} with max_connections={} sample_rate={} chunk_samples={}",
        address,
        config.max_connections,
        config.sample_rate,
        block_size(config.sample_rate)
    );

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        runtime: "rust+official-onnxruntime",
        active_connections: state.active_connections.load(Ordering::Relaxed),
        max_connections: state.config.max_connections,
        sample_rate: state.config.sample_rate,
        chunk_samples: block_size(state.config.sample_rate),
    })
}

async fn vad_socket(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    let active = state.active_connections.load(Ordering::Relaxed);
    if active >= state.config.max_connections {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "too many active VAD sessions",
        )
            .into_response();
    }

    ws.on_upgrade(move |socket| handle_socket(socket, state))
        .into_response()
}

async fn handle_socket(socket: WebSocket, state: AppState) {
    state.active_connections.fetch_add(1, Ordering::Relaxed);
    let started = Instant::now();
    let result = run_socket(socket, state.clone()).await;
    state.active_connections.fetch_sub(1, Ordering::Relaxed);

    if let Err(error) = result {
        warn!("VAD socket closed with error: {error:#}");
    }
    info!("VAD socket closed after {:?}", started.elapsed());
}

async fn run_socket(socket: WebSocket, state: AppState) -> Result<()> {
    let (mut sender, mut receiver) = socket.split();
    let mut vad = VadStream::new(&state.config);
    let idle_timeout = Duration::from_secs(state.config.idle_timeout_secs);

    send_event(
        &mut sender,
        &VadEvent::Ready {
            sample_rate: state.config.sample_rate,
            chunk_samples: vad.block_size,
            format: "pcm_s16le_mono",
        },
    )
    .await?;

    loop {
        let message = match timeout(idle_timeout, receiver.next()).await {
            Ok(Some(Ok(message))) => message,
            Ok(Some(Err(error))) => bail!("websocket receive failed: {error}"),
            Ok(None) => break,
            Err(_) => bail!(
                "idle timeout after {} seconds",
                state.config.idle_timeout_secs
            ),
        };

        match message {
            Message::Binary(payload) => {
                let event = {
                    let mut session = state.session.lock().await;
                    vad.handle_pcm_chunk(&mut session, &payload)?
                };
                if let Some(event) = event {
                    send_event(&mut sender, &event).await?;
                }
            }
            Message::Ping(payload) => sender.send(Message::Pong(payload)).await?,
            Message::Pong(_) => {}
            Message::Close(_) => break,
            Message::Text(_) => {
                send_event(
                    &mut sender,
                    &VadEvent::Error {
                        message: "send binary pcm_s16le audio chunks, not text".to_string(),
                    },
                )
                .await?;
            }
        }
    }

    Ok(())
}

async fn send_event(
    sender: &mut futures_util::stream::SplitSink<WebSocket, Message>,
    event: &VadEvent,
) -> Result<()> {
    let payload = serde_json::to_string(event)?;
    sender.send(Message::Text(payload.into())).await?;
    Ok(())
}

async fn shutdown_signal() {
    if let Err(error) = tokio::signal::ctrl_c().await {
        error!("failed to listen for shutdown signal: {error}");
    }
    info!("shutdown signal received");
}
