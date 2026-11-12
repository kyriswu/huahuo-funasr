# FunASR ECI GPU Implementation Plan

**Goal:** Build and validate a self-contained GPU image providing offline ASR, diarization, and enrolled-speaker recognition.

**Architecture:** A single FastAPI process uses one serialized FunASR pipeline. Complete, versioned model snapshots are baked into the image; voiceprints persist separately under `/data`.

**Tech Stack:** Python 3.11, PyTorch CUDA 12.6, FunASR, FastAPI, FFmpeg, Docker, Alibaba ECI.

**Spec:** `docs/plans/2026-11-12-funasr-eci-design.md`

## Global Constraints

- Linux amd64; target NVIDIA T4; default GPU execution must fail if unavailable.
- No runtime downloads, dependency installation, or remote model code.
- Model snapshots and dependencies are versioned; voiceprints carry a model fingerprint.
- Milliseconds for all reported audio timestamps.
- One active inference per instance; 429 on busy; single writer per persisted voiceprint directory.

## Review Focus

- Malformed, empty, very long, or silent audio must have bounded resource use and clear responses.
- Unknown speakers must not silently receive the nearest known identity.
- A corrupted or incompatible voiceprint must fail visibly, without losing stored data.
- Multiple calls must not retain a previous request's speaker count or options.
- A GPU deployment without working CUDA must fail before reporting readiness.

## Task 1: Reproducible model-inclusive image

- [x] Add a pinned dependency specification and immutable upstream source reference.
- [x] Implement `scripts/download_models.py` and `app/model_files.py` to download fixed revisions, create a SHA-256 manifest, and verify local model directories.
- [x] Create Docker dependency, model, runtime, and test stages. Build the dependency stage while implementing the service.
- [x] Check installed relationships with `pip check`; verify model files after download.

## Task 2: Inference and voiceprint persistence

- [x] Implement bounded FFmpeg decoding in `app/audio.py`.
- [x] Implement normalized cosine matching and atomic JSON storage in `app/voiceprints.py` with model fingerprint validation.
- [x] Implement `SpeechEngine.transcribe`, `SpeechEngine.embedding`, and strict accelerator initialization in `app/engine.py`; reuse the pipeline's speaker model.
- [x] Add meaningful tests for threshold rejection, persistence, model mismatch, audio failure paths, and repeated request options.

## Task 3: HTTP service

- [x] Implement multipart endpoints in `app/main.py`, Bearer authentication, inference gating, readiness, limits, and structured errors.
- [x] Validate ASGI behavior with an injected fake engine; keep real inference tests separate.
- [x] Exercise all endpoint shapes and confirm health probes remain responsive during inference.

## Task 4: Deployment and acceptance

- [x] Add Compose, a direct ECI request renderer, and an ACK virtual-node Pod example; keep credentials external.
- [x] Document build, GPU run, API usage, model versions, NAS ownership, ACR publication, ECI resource sizing and probe settings.
- [x] Build the final image; test actual ASR, a two-speaker recording, voiceprint registration, positive/negative verification and persistence.
- [x] Run real inference inside `--network none`; record image ID, size, hardware, observed results and remaining validation limits in `docs/validation.md`.
