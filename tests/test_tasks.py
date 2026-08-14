"""Regression test for #134: the beat schedule fires `purge_queue_messages` with no arguments
against a signature that required one, so every firing raised TypeError."""

from triton_serve.tasks import purge_queue_messages


class RecordingClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def delete(self, path: str):
        self.calls.append(path)
        return type("Response", (), {"text": '{"deleted_messages": 0}'})()


def test_beat_call_without_arguments_succeeds():
    result = purge_queue_messages.apply()
    assert result.successful(), result.result


def test_injected_client_is_called():
    client = RecordingClient()
    result = purge_queue_messages.apply(args=[client])
    assert result.successful(), result.result
    assert client.calls == ["queue/messages"]
