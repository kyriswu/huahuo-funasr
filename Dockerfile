# syntax=docker/dockerfile:1.7
ARG PYTHON_IMAGE=python:3.11.14-slim-bookworm@sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d

FROM ${PYTHON_IMAGE} AS base
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates ffmpeg libsndfile1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

FROM base AS funasr-source
COPY scripts/prepare_funasr.py /tmp/prepare_funasr.py
RUN python /tmp/prepare_funasr.py --output /opt/funasr-src

FROM base AS torch-dependencies
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install pip==25.3 setuptools==80.9.0 wheel==0.45.1 \
    && python -m pip install torch==2.8.0+cu126 torchaudio==2.8.0+cu126 --index-url ${TORCH_INDEX_URL}

FROM torch-dependencies AS dependencies
COPY requirements.txt constraints.txt /tmp/
COPY --from=funasr-source /opt/funasr-src /opt/funasr-src
RUN --mount=type=cache,target=/root/.cache/pip \
    sed '/^funasr==/d' /tmp/constraints.txt > /tmp/dependency-lock.txt \
    && python -m pip install --timeout 30 --retries 3 -c /tmp/constraints.txt -r /tmp/requirements.txt -r /tmp/dependency-lock.txt \
    && python -m pip wheel --no-build-isolation --no-deps /opt/funasr-src --wheel-dir /tmp/funasr-wheel \
    && python -m pip install --no-deps /tmp/funasr-wheel/*.whl \
    && python -m pip check \
    && python -m pip freeze > /opt/python-environment.txt \
    && cp /opt/funasr-src/UPSTREAM.json /opt/funasr-upstream.json \
    && rm -rf /opt/funasr-src /tmp/funasr-wheel

# Model acquisition is independent of the GPU wheel download and needs no GPU.
FROM base AS models
WORKDIR /build
COPY model-lock.json .
COPY scripts/download_models.py scripts/download_models.py
RUN python scripts/download_models.py --lock model-lock.json --output /opt/models

FROM base AS runtime
ENV PATH=/opt/venv/bin:$PATH \
    DEVICE=cuda:0 MODEL_ROOT=/opt/models DATA_DIR=/data \
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
COPY model-lock.json /opt/app/model-lock.json
LABEL org.opencontainers.image.title="FunASR ECI offline GPU service" \
      org.opencontainers.image.description="Offline ASR, speaker diarization and voiceprint recognition" \
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

# Keep the production image as the default build target.
FROM runtime AS production
