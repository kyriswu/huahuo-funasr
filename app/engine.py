import copy
import logging
import time

import numpy as np

from app.audio import AudioClip, AudioError, SAMPLE_RATE
from app.model_files import verify_models
from app.settings import Settings
from app.voiceprints import match_voiceprint, normalize

logger = logging.getLogger(__name__)


class SpeechEngine:
    def __init__(self, settings: Settings):
        # Keep GPU imports in initialization so API tests need no model dependencies.
        import torch
        from funasr import AutoModel

        self.settings = settings
        self.torch = torch
        if settings.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable. Attach an NVIDIA GPU to ECI; DEVICE=cpu is an explicit diagnostic option.")
            device = torch.device(settings.device)
            torch.cuda.set_device(device)
            # Check kernel execution as well as driver/device visibility.
            torch.ones(1, device=device).sum().item()
            logger.info("Using GPU: %s; torch=%s CUDA=%s", torch.cuda.get_device_name(device), torch.__version__, torch.version.cuda)
        self.lock, self.model_id = verify_models(settings.model_root, settings.model_lock)
        torch.set_num_threads(settings.cpu_threads)
        self.pipeline = AutoModel(
            model=str(settings.model_root / "asr"),
            vad_model=str(settings.model_root / "vad"),
            punc_model=str(settings.model_root / "punc"),
            spk_model=str(settings.model_root / "speaker"),
            vad_kwargs={"max_single_segment_time": 30000},
            device=settings.device, ncpu=settings.cpu_threads,
            max_num_spks=settings.max_speakers,
            spk_mode="punc_segment", disable_update=True,
            trust_remote_code=False, disable_pbar=True, disable_log=True,
        )
        for name in ("kwargs", "vad_kwargs", "punc_kwargs", "spk_kwargs"):
            actual = getattr(self.pipeline, name)["device"]
            if settings.device.startswith("cuda") and not actual.startswith("cuda"):
                raise RuntimeError(f"{name} unexpectedly loaded on {actual}")
        logger.info("ASR, VAD, punctuation and speaker models loaded from verified local files")

    def embedding(self, clip: AudioClip) -> tuple[np.ndarray, float]:
        """Remove silence and average short CAM++ windows from single-speaker audio."""
        with self.torch.inference_mode():
            vad = self.pipeline.inference(
                clip.samples, model=self.pipeline.vad_model,
                kwargs=copy.deepcopy(self.pipeline.vad_kwargs),
                fs=SAMPLE_RATE, is_final=True,
            )
            intervals = vad[0].get("value", []) if vad else []
            parts = []
            for start, end in intervals:
                left = max(0, int(start * SAMPLE_RATE / 1000))
                right = min(clip.samples.size, int(end * SAMPLE_RATE / 1000))
                if right > left:
                    parts.append(clip.samples[left:right])
            speech_seconds = sum(part.size for part in parts) / SAMPLE_RATE
            if speech_seconds < self.settings.min_voice_seconds:
                raise AudioError(f"Need at least {self.settings.min_voice_seconds:g} seconds of detected speech from one speaker")
            speech = np.concatenate(parts)
            # 3-second windows prevent long enrollment clips from consuming unbounded GPU memory.
            window = SAMPLE_RATE * 3
            vectors, weights = [], []
            for offset in range(0, speech.size, window):
                samples = speech[offset:offset + window]
                if samples.size < SAMPLE_RATE:
                    continue
                result = self.pipeline.inference(
                    samples, model=self.pipeline.spk_model,
                    kwargs=copy.deepcopy(self.pipeline.spk_kwargs), fs=SAMPLE_RATE,
                )
                vector = result[0]["spk_embedding"].detach().cpu().numpy().reshape(-1)
                vectors.append(vector)
                weights.append(samples.size)
            return normalize(np.average(vectors, axis=0, weights=weights)), speech_seconds

    def transcribe(self, clip: AudioClip, speaker_count: int | None, records: list[dict] | None, threshold: float) -> dict:
        started = time.perf_counter()
        # Supply all request-specific options on every call; never reuse an earlier speaker count.
        with self.torch.inference_mode():
            results = self.pipeline.generate(
                input=clip.samples, fs=SAMPLE_RATE,
                batch_size_s=self.settings.batch_size_s,
                preset_spk_num=speaker_count,
                return_spk_res=True, return_spk_center=records is not None,
                sentence_timestamp=True,
            )
        raw = results[0] if results else {}
        centers = raw.get("spk_embedding_center")
        speaker_ids = sorted({int(segment["spk"]) for segment in raw.get("sentence_info", []) if "spk" in segment})
        speakers = []
        for index in speaker_ids:
            speaker = {"speaker_id": f"speaker_{index}"}
            if records is not None:
                if centers is None or index < 0 or index >= len(centers):
                    raise RuntimeError("FunASR returned speaker labels without matching embeddings")
                speaker["voiceprint"] = match_voiceprint(centers[index], records, threshold)
            speakers.append(speaker)
        segments = []
        for item in raw.get("sentence_info", []):
            index = item.get("spk")
            segments.append({
                "start_ms": int(item["start"]), "end_ms": int(item["end"]),
                "text": item.get("text", item.get("sentence", "")),
                "speaker_id": f"speaker_{int(index)}" if index is not None else None,
                "timestamps_ms": [[int(a), int(b)] for a, b in item.get("timestamp", [])],
            })
        elapsed = time.perf_counter() - started
        return {
            "text": raw.get("text", ""), "language": "zh",
            "duration_seconds": round(clip.duration_seconds, 3),
            "processing_seconds": round(elapsed, 3),
            "timestamps_ms": [[int(a), int(b)] for a, b in raw.get("timestamp", [])],
            "segments": segments, "speakers": speakers,
        }
