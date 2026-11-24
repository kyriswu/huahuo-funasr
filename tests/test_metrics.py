from urllib.error import HTTPError
from urllib.request import urlopen

from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families
import pytest

from app.main import create_app
from app.settings import Settings
from test_api import FakeEngine
from test_tasks import acknowledge, submit, wait_status


def test_private_metrics_listener_records_job_lifecycle(tmp_path, audio_bytes):
    settings = Settings(data_dir=tmp_path, metrics_port=0, api_token="configured")
    headers = {"Authorization": "Bearer configured"}
    with TestClient(create_app(settings, FakeEngine)) as client:
        client.headers.update(headers)
        port = client.app.state.metrics_server.server_port
        url = f"http://127.0.0.1:{port}/metrics"

        def scrape():
            with urlopen(url, timeout=2) as response:
                body = response.read().decode()
            samples = [sample for family in text_string_to_metric_families(body) for sample in family.samples]
            return body, {(sample.name, tuple(sorted(sample.labels.items()))): sample.value for sample in samples}

        assert client.get("/metrics").status_code == 404
        task_id = submit(client, audio_bytes)
        wait_status(client, task_id)
        event = client.get(f"/v1/tasks/{task_id}/result").json()
        body, samples = scrape()
        assert samples["funasr_ready", ()] == 1
        assert samples["funasr_startup_duration_seconds", ()] >= 0
        assert samples["funasr_async_unconsumed_tasks", ()] == 1
        assert samples["funasr_async_audio_bytes", ()] == 0
        assert samples["funasr_async_tasks_completed_total", (("outcome", "succeeded"),)] == 1
        assert samples["funasr_transcription_stage_duration_seconds_count", (("mode", "async"), ("stage", "inference"))] == 1
        assert task_id not in body and event["event_id"] not in body and "configured" not in body

        assert acknowledge(client, task_id, event).status_code == 200
        _, samples = scrape()
        assert samples["funasr_async_unconsumed_tasks", ()] == 0

        with pytest.raises(HTTPError) as error:
            urlopen(f"http://127.0.0.1:{port}/v1/tasks", timeout=2)
        assert error.value.code == 404
