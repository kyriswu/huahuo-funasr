import threading
from socketserver import ThreadingMixIn
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, ProcessCollector, make_wsgi_app


class ScrapeHandler(WSGIRequestHandler):
    timeout = 5

    def log_message(self, format, *args):
        pass


class ScrapeServer(ThreadingMixIn, WSGIServer):
    daemon_threads = True


class Metrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        ProcessCollector(registry=self.registry)
        self.ready = Gauge("funasr_ready", "Whether FunASR is ready to accept requests.", registry=self.registry)
        self.startup = Gauge("funasr_startup_duration_seconds", "Model and service startup time.", registry=self.registry)
        self.startup_stage = Gauge(
            "funasr_startup_stage_duration_seconds", "Wall time of the last startup by stage, including nested stages.",
            ("stage",), registry=self.registry,
        )
        self.unconsumed = Gauge("funasr_async_unconsumed_tasks", "Tasks holding a capacity slot.", registry=self.registry)
        self.audio = Gauge("funasr_async_audio_bytes", "Bytes reserved for staged task audio.", registry=self.registry)
        self.completed = Counter("funasr_async_tasks_completed", "Terminal async tasks.", ("outcome",), registry=self.registry)
        self.callback_attempts = Counter(
            "funasr_callback_attempts", "Result delivery attempts.", ("outcome",), registry=self.registry,
        )
        self.gpu_oom = Counter("funasr_gpu_oom", "Requests that exhausted GPU memory.", registry=self.registry)
        self.stage = Histogram(
            "funasr_transcription_stage_duration_seconds",
            "Time spent in each transcription stage, including failures.", ("mode", "stage"),
            buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600, 1800), registry=self.registry,
        )

    def attach(self, tasks, engine, device):
        self.unconsumed.set_function(lambda: tasks.capacity_usage()[0])
        self.audio.set_function(lambda: tasks.capacity_usage()[1])
        torch = getattr(engine, "torch", None)
        if device.startswith("cuda") and torch is not None:
            for name, description, sample in (
                ("funasr_cuda_memory_reserved_bytes", "Bytes held by the PyTorch CUDA allocator.", torch.cuda.memory_reserved),
                ("funasr_cuda_peak_memory_reserved_bytes", "Peak bytes held by the PyTorch CUDA allocator.", torch.cuda.max_memory_reserved),
            ):
                Gauge(name, description, registry=self.registry).set_function(lambda sample=sample: sample(device))

    def listen(self, port):
        exposition = make_wsgi_app(registry=self.registry)

        def metrics_only(environ, start_response):
            if environ["PATH_INFO"] == "/metrics":
                return exposition(environ, start_response)
            start_response("404 Not Found", [("Content-Type", "text/plain")])
            return [b"Not Found\n"]

        server = make_server("0.0.0.0", port, metrics_only, server_class=ScrapeServer, handler_class=ScrapeHandler)
        thread = threading.Thread(target=server.serve_forever, name="funasr-metrics", daemon=True)
        thread.start()
        return server
