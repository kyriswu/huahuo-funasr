# syntax=docker/dockerfile:1.7
ARG PYTHON_IMAGE=python:3.11.14-slim-bookworm@sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d
ARG BUILD_PROFILE=standard

FROM ${PYTHON_IMAGE} AS base
ARG DEBIAN_MIRROR
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN if [ -n "${DEBIAN_MIRROR}" ]; then \
        sed -i "s|^URIs: http://deb\.debian\.org/|URIs: ${DEBIAN_MIRROR%/}/|" /etc/apt/sources.list.d/debian.sources; \
    fi \
    && apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates ffmpeg libsndfile1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_TRUSTED_HOST

# A release builder can supply this named context with a verified funasr.tar.gz.
FROM scratch AS funasr-upstream-archive
FROM base AS funasr-source
COPY scripts/prepare_funasr.py /tmp/prepare_funasr.py
RUN --mount=from=funasr-upstream-archive,target=/mnt/upstream \
    if [ -e /mnt/upstream/funasr.tar.gz ]; then set -- --archive /mnt/upstream/funasr.tar.gz; fi \
    && python /tmp/prepare_funasr.py --output /opt/funasr-src "$@"

FROM base AS torch-dependencies
ARG BUILD_PROFILE
ARG TORCH_INDEX_URL
ARG TORCH_WHEEL_MIRROR
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN --mount=type=cache,target=/root/.cache/pip \
    case "${BUILD_PROFILE}" in \
        standard) torch_version=2.8.0+cu126; torch_index=https://download.pytorch.org/whl/cu126 ;; \
        a10-2gb) \
            torch_version=2.7.1+cu118; torch_index=https://download.pytorch.org/whl/cu118; \
            torch_sha=a596d91c747d1fa601724e85b9c8797c8d7c62140aa1acf245773e911254bc45; \
            torchaudio_sha=76629345b5e095b5b9dd6abbeb1919e50dd8f3c2082d89b54ad5cd7d25158de9 ;; \
        nvidia-modern) \
            torch_version=2.8.0+cu128; torch_index=https://download.pytorch.org/whl/cu128; \
            torch_sha=039b9dcdd6bdbaa10a8a5cd6be22c4cb3e3589a341e5f904cbb571ca28f55bed; \
            torchaudio_sha=f4409df567d0723a7a3a89d32c7552a17e0ff6f137ea26a0d268c665259b2995 ;; \
        *) echo "Unsupported BUILD_PROFILE: ${BUILD_PROFILE}" >&2; exit 1 ;; \
    esac \
    && python -m pip install pip==25.3 setuptools==80.9.0 wheel==0.45.1 \
    && if [ -n "${TORCH_WHEEL_MIRROR}" ] && [ "${BUILD_PROFILE}" != standard ]; then \
        wheel_dir="${TORCH_WHEEL_MIRROR%/}/${torch_version##*+}"; \
        wheel_suffix="${torch_version%+*}%2B${torch_version##*+}-cp311-cp311-manylinux_2_28_x86_64.whl"; \
        python -m pip install --no-deps \
            "${wheel_dir}/torch-${wheel_suffix}#sha256=${torch_sha}" \
            "${wheel_dir}/torchaudio-${wheel_suffix}#sha256=${torchaudio_sha}"; \
    else \
        python -m pip install "torch==${torch_version}" "torchaudio==${torch_version}" \
            --index-url "${TORCH_INDEX_URL:-${torch_index}}"; \
    fi

FROM torch-dependencies AS dependencies
COPY requirements.txt constraints.txt constraints-cu118.txt constraints-cu128.txt /tmp/
COPY --from=funasr-source /opt/funasr-src /opt/funasr-src
RUN --mount=type=cache,target=/root/.cache/pip \
    case "${BUILD_PROFILE}" in \
        a10-2gb) runtime_constraints=/tmp/constraints-cu118.txt ;; \
        nvidia-modern) runtime_constraints=/tmp/constraints-cu128.txt ;; \
        *) runtime_constraints= ;; \
    esac \
    && if [ -n "${runtime_constraints}" ]; then \
        sed -E '/^(nvidia-|torch==|torchaudio==|triton==)/d' /tmp/constraints.txt > /tmp/constraints-selected.txt \
        && cat "${runtime_constraints}" >> /tmp/constraints-selected.txt; \
    else \
        cp /tmp/constraints.txt /tmp/constraints-selected.txt; \
    fi \
    && sed '/^funasr==/d' /tmp/constraints-selected.txt > /tmp/dependency-lock.txt \
    && python -m pip install --timeout 30 --retries 3 -c /tmp/constraints-selected.txt -r /tmp/requirements.txt -r /tmp/dependency-lock.txt \
    && python -m pip wheel --no-build-isolation --no-deps /opt/funasr-src --wheel-dir /tmp/funasr-wheel \
    && python -m pip install --no-deps /tmp/funasr-wheel/*.whl \
    && python -m pip check \
    && python -m pip freeze > /opt/python-environment.txt \
    && cp /opt/funasr-src/UPSTREAM.json /opt/funasr-upstream.json \
    && rm -rf /opt/funasr-src /tmp/funasr-wheel

# Model acquisition is independent of the GPU wheel download and needs no GPU.
FROM base AS models
ARG BUILD_PROFILE
WORKDIR /build
COPY model-lock.json model-lock-punc-small.json ./
COPY scripts/download_models.py scripts/download_models.py
RUN python - <<'PY'
import json
import os
from pathlib import Path

if os.environ["BUILD_PROFILE"] == "a10-2gb":
    lock_path = Path("model-lock.json")
    lock = json.loads(lock_path.read_text())
    lock["models"]["punc"] = json.loads(Path("model-lock-punc-small.json").read_text())
    lock_path.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n")
PY
RUN --mount=type=cache,id=funasr-models-v1,target=/var/cache/funasr-models,sharing=locked \
    python scripts/download_models.py --lock model-lock.json --output /opt/models --cache-dir /var/cache/funasr-models

FROM base AS runtime
ARG BUILD_PROFILE
ENV PATH=/opt/venv/bin:$PATH \
    DEVICE=cuda:0 FUNASR_BUILD_PROFILE=${BUILD_PROFILE} MODEL_ROOT=/opt/models DATA_DIR=/data \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MODELSCOPE_OFFLINE=1 \
    HF_HOME=/tmp/funasr-cache/huggingface MODELSCOPE_CACHE=/tmp/funasr-cache/modelscope \
    NUMBA_CACHE_DIR=/tmp/funasr-cache/numba OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
COPY --from=dependencies /opt/venv /opt/venv
COPY --from=dependencies /opt/python-environment.txt /opt/python-environment.txt
COPY --from=dependencies /opt/funasr-upstream.json /opt/funasr-upstream.json
COPY --from=models /opt/models /opt/models
RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --no-create-home app \
    && mkdir -p /data/voiceprints /opt/app \
    && chown -R app:app /data /opt/app
WORKDIR /opt/app
COPY --chown=app:app app/ app/
COPY --chown=app:app scripts/ scripts/
COPY --from=models /build/model-lock.json /opt/app/model-lock.json
LABEL org.opencontainers.image.title="FunASR offline GPU service" \
      org.opencontainers.image.description="Offline ASR, speaker diarization, voiceprints and async callbacks with explicit ACK" \
      org.opencontainers.image.version="1.1.0" \
      org.opencontainers.image.source="https://github.com/modelscope/FunASR" \
      io.funasr.upstream.revision="be264e9382b4fef735536f3d278ee14514098a1a"
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=4)"]
STOPSIGNAL SIGTERM
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-graceful-shutdown", "600", "--no-access-log"]

FROM runtime AS test
USER root
COPY requirements-test.txt /tmp/requirements-test.txt
RUN --mount=type=cache,target=/root/.cache/pip python -m pip install -r /tmp/requirements-test.txt
COPY --chown=app:app tests/ tests/
USER 10001:10001
CMD ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"]

# Validate the amd64 ASK candidate inside BuildKit, including on an ARM64 build host.
FROM runtime AS ask-release
ARG BUILD_PROFILE
RUN --network=none python - <<'PY'
import os
from pathlib import Path

import torch

from app.main import app
from app.model_files import verify_models
from app.settings import Settings

profile = os.environ["BUILD_PROFILE"]
expected_cuda = {"a10-2gb": "11.8", "nvidia-modern": "12.8"}.get(profile)
if expected_cuda is None or torch.version.cuda != expected_cuda:
    raise RuntimeError(f"Unexpected CUDA runtime {torch.version.cuda!r} for ASK profile {profile!r}")
required_arches = {"sm_75", "sm_80", "sm_90"}
required_arches.update({"sm_86"} if profile == "a10-2gb" else {"sm_100", "sm_120"})
compiled_arches = set((torch._C._cuda_getArchFlags() or "").split())
if missing_arches := required_arches - compiled_arches:
    raise RuntimeError(f"PyTorch for ASK {profile!r} lacks GPU kernels for {sorted(missing_arches)}")
settings = Settings.from_env()
expected_batch_size = 10 if profile == "a10-2gb" else 120
if settings.device != "cuda:0" or settings.batch_size_s != expected_batch_size:
    raise RuntimeError(f"Unexpected inference defaults for ASK profile {profile!r}")
if not any(route.path == "/readyz" for route in app.routes):
    raise RuntimeError("Missing readiness endpoint")
verify_models(Path("/opt/models"), Path("/opt/app/model-lock.json"))
PY

# Keep the production image as the default build target.
FROM runtime AS production
