use anyhow::{Context, Result, bail};
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use serde::Serialize;
use std::{env, time::Instant};

const PCM_I16_BYTES_PER_SAMPLE: usize = 2;

#[derive(Clone, Copy)]
enum Mode {
    Allocating,
    ReusedBuffers,
}

struct Config {
    model: String,
    ort_lib: String,
    sample_rate: i64,
    chunks: usize,
    threads: usize,
    mode: Mode,
    opt_level: GraphOptimizationLevel,
}

struct VadState {
    sample_rate: i64,
    block_size: usize,
    context_size: usize,
    input: Array2<f32>,
    state: ArrayD<f32>,
    sr: Array1<i64>,
}

#[derive(Serialize)]
struct BenchResult {
    mode: &'static str,
    model: String,
    sample_rate: i64,
    chunks: usize,
    audio_seconds: f64,
    elapsed_ms: f64,
    avg_chunk_ms: f64,
    p95_chunk_ms: f64,
    max_chunk_ms: f64,
    realtime_factor: f64,
    estimated_0_1_cpu_chunk_ms: f64,
}

fn ort_err<E: std::fmt::Display>(error: E) -> anyhow::Error {
    anyhow::anyhow!(error.to_string())
}

fn main() -> Result<()> {
    let config = parse_args()?;
    ort::init_from(&config.ort_lib).map_err(ort_err)?.commit();

    let result = run_benchmark(&config)?;
    println!("{}", serde_json::to_string_pretty(&result)?);

    // Avoid a Windows-only ONNX Runtime DLL cleanup crash seen with some local DLL builds.
    std::process::exit(0);
}

fn parse_args() -> Result<Config> {
    let mut config = Config {
        model: "models/silero_vad.onnx".to_string(),
        ort_lib: "/opt/onnxruntime/lib/libonnxruntime.so.1.26.0".to_string(),
        sample_rate: 16000,
        chunks: 20_000,
        threads: 1,
        mode: Mode::Allocating,
        opt_level: GraphOptimizationLevel::Level2,
    };

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => config.model = args.next().context("--model requires a value")?,
            "--ort-lib" => config.ort_lib = args.next().context("--ort-lib requires a value")?,
            "--sample-rate" => {
                config.sample_rate = args
                    .next()
                    .context("--sample-rate requires a value")?
                    .parse()?
            }
            "--chunks" => {
                config.chunks = args.next().context("--chunks requires a value")?.parse()?
            }
            "--threads" => {
                config.threads = args.next().context("--threads requires a value")?.parse()?
            }
            "--opt-level" => {
                let value = args.next().context("--opt-level requires a value")?;
                config.opt_level = match value.as_str() {
                    "disable" | "0" => GraphOptimizationLevel::Disable,
                    "level1" | "1" => GraphOptimizationLevel::Level1,
                    "level2" | "2" => GraphOptimizationLevel::Level2,
                    _ => bail!("--opt-level must be disable, level1, or level2"),
                };
            }
            "--mode" => {
                let mode = args.next().context("--mode requires a value")?;
                config.mode = match mode.as_str() {
                    "allocating" => Mode::Allocating,
                    "reused" | "reused_buffers" => Mode::ReusedBuffers,
                    _ => bail!("--mode must be allocating or reused"),
                };
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
    if config.chunks == 0 {
        bail!("--chunks must be at least 1");
    }
    if config.threads == 0 {
        bail!("--threads must be at least 1");
    }

    Ok(config)
}

fn print_help() {
    println!(
        "Silero VAD chunk benchmark\n\
\n\
Usage:\n\
  chunk_bench --ort-lib <path> [options]\n\
\n\
Options:\n\
  --model <path>              ONNX model, default models/silero_vad.onnx\n\
  --ort-lib <path>            Path to libonnxruntime.so / onnxruntime.dll\n\
  --sample-rate <8000|16000>  Input sample rate, default 16000\n\
  --chunks <n>                Number of chunks, default 20000\n\
  --threads <n>               ONNX Runtime threads, default 1\n\
  --opt-level <disable|level1|level2>  ORT graph optimization, default level2\n\
  --mode <allocating|reused>  Processing path, default allocating"
    );
}

fn run_benchmark(config: &Config) -> Result<BenchResult> {
    let mut session = Session::builder()
        .map_err(ort_err)?
        .with_optimization_level(config.opt_level)
        .map_err(ort_err)?
        .with_intra_threads(config.threads)
        .map_err(ort_err)?
        .with_inter_threads(config.threads)
        .map_err(ort_err)?
        .commit_from_file(&config.model)
        .map_err(ort_err)?;

    let chunk = benchmark_chunk(block_size(config.sample_rate));
    let mut vad = VadState::new(config.sample_rate);
    let mut chunk_times = Vec::with_capacity(config.chunks);
    let started = Instant::now();

    for _ in 0..config.chunks {
        let chunk_started = Instant::now();
        match config.mode {
            Mode::Allocating => vad.process_allocating(&mut session, &chunk)?,
            Mode::ReusedBuffers => vad.process_reused_buffers(&mut session, &chunk)?,
        }
        chunk_times.push(chunk_started.elapsed().as_secs_f64() * 1000.0);
    }

    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    chunk_times.sort_by(|left, right| left.total_cmp(right));

    let audio_seconds =
        config.chunks as f64 * block_size(config.sample_rate) as f64 / config.sample_rate as f64;
    let avg_chunk_ms = chunk_times.iter().sum::<f64>() / chunk_times.len() as f64;

    Ok(BenchResult {
        mode: match config.mode {
            Mode::Allocating => "allocating",
            Mode::ReusedBuffers => "reused_buffers",
        },
        model: config.model.clone(),
        sample_rate: config.sample_rate,
        chunks: config.chunks,
        audio_seconds,
        elapsed_ms,
        avg_chunk_ms,
        p95_chunk_ms: percentile(&chunk_times, 0.95),
        max_chunk_ms: chunk_times.last().copied().unwrap_or(0.0),
        realtime_factor: audio_seconds / (elapsed_ms / 1000.0),
        estimated_0_1_cpu_chunk_ms: avg_chunk_ms * 10.0,
    })
}

fn benchmark_chunk(block_size: usize) -> Vec<u8> {
    let mut chunk = Vec::with_capacity(block_size * PCM_I16_BYTES_PER_SAMPLE);
    let mut state = 0x1234_5678_u32;

    for index in 0..block_size {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let noise = ((state >> 16) as i32 - 32768) as f32 / 32768.0;
        let wave = ((index as f32) * 0.061_359_23).sin();
        let sample = ((wave * 0.18 + noise * 0.02) * i16::MAX as f32) as i16;
        chunk.extend_from_slice(&sample.to_le_bytes());
    }

    chunk
}

fn percentile(values: &[f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let index = ((values.len() - 1) as f64 * percentile).round() as usize;
    values[index.min(values.len() - 1)]
}

fn block_size(sample_rate: i64) -> usize {
    if sample_rate == 16000 { 512 } else { 256 }
}

fn context_size(sample_rate: i64) -> usize {
    if sample_rate == 16000 { 64 } else { 32 }
}

impl VadState {
    fn new(sample_rate: i64) -> Self {
        let block_size = block_size(sample_rate);
        let context_size = context_size(sample_rate);

        Self {
            sample_rate,
            block_size,
            context_size,
            input: Array2::<f32>::zeros((1, context_size + block_size)),
            state: ArrayD::<f32>::zeros(IxDyn(&[2, 1, 128])),
            sr: Array1::<i64>::from_vec(vec![sample_rate]),
        }
    }

    fn process_allocating(&mut self, session: &mut Session, payload: &[u8]) -> Result<()> {
        self.validate(payload)?;

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

        Ok(())
    }

    fn process_reused_buffers(&mut self, session: &mut Session, payload: &[u8]) -> Result<()> {
        self.validate(payload)?;

        let input_buffer = self
            .input
            .as_slice_mut()
            .context("input buffer is not contiguous")?;
        input_buffer.copy_within(self.block_size..self.block_size + self.context_size, 0);
        for (index, sample) in payload.chunks_exact(2).enumerate() {
            let value = i16::from_le_bytes([sample[0], sample[1]]);
            input_buffer[self.context_size + index] = value as f32 / i16::MAX as f32;
        }

        let input_tensor = TensorRef::from_array_view(self.input.view()).map_err(ort_err)?;
        let state_tensor = TensorRef::from_array_view(self.state.view()).map_err(ort_err)?;
        let sr_tensor = TensorRef::from_array_view(self.sr.view()).map_err(ort_err)?;

        let outputs = session
            .run(ort::inputs![
                "input" => input_tensor,
                "state" => state_tensor,
                "sr" => sr_tensor,
            ])
            .map_err(ort_err)?;

        let state_output = outputs["stateN"]
            .try_extract_array::<f32>()
            .map_err(ort_err)?;
        let state_buffer = self
            .state
            .as_slice_mut()
            .context("state buffer is not contiguous")?;
        state_buffer.copy_from_slice(
            state_output
                .as_slice()
                .context("state output buffer is not contiguous")?,
        );

        Ok(())
    }

    fn validate(&self, payload: &[u8]) -> Result<()> {
        let expected_bytes = self.block_size * PCM_I16_BYTES_PER_SAMPLE;
        if payload.len() != expected_bytes {
            bail!(
                "invalid chunk size for {} Hz: got {} bytes, expected {} bytes",
                self.sample_rate,
                payload.len(),
                expected_bytes
            );
        }

        Ok(())
    }
}
