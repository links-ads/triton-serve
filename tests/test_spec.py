from types import SimpleNamespace

from triton_serve.api.services.container import spawn_service_container
from triton_serve.api.services.spec import SPEC_LABEL, container_spec


def _svc(models=("b", "a"), devices=("gpu-2", "gpu-1"), env=None, healthcheck=None, image="img:1"):
    return SimpleNamespace(
        service_name="svc",
        service_image=image,
        image=None,
        models=[SimpleNamespace(model_name=name) for name in models],
        device_allocations=[SimpleNamespace(device_id=uuid) for uuid in devices],
        resources=SimpleNamespace(
            cpu_count=2,
            mem_size=1024,
            shm_size=256,
            environment_variables=env,
            healthcheck=healthcheck,
        ),
    )


def test_spec_reads_the_service_row():
    spec = container_spec(_svc(env={"A": "1"}))
    assert spec.image_ref == "img:1"
    assert spec.models == ("a", "b")
    assert spec.device_uuids == ("gpu-1", "gpu-2")
    assert (spec.cpu_count, spec.mem_size, spec.shm_size) == (2, 1024, 256)
    assert spec.environment == {"A": "1"}


def test_fingerprint_ignores_relationship_order():
    shuffled = container_spec(_svc(models=("a", "b"), devices=("gpu-1", "gpu-2")))
    assert container_spec(_svc()).fingerprint == shuffled.fingerprint


def test_fingerprint_ignores_environment_key_order():
    # jsonb round-trips give arbitrary key order; an unsorted dump would recreate forever
    first = container_spec(_svc(env={"A": "1", "B": "2"}))
    second = container_spec(_svc(env={"B": "2", "A": "1"}))
    assert first.fingerprint == second.fingerprint


def test_null_environment_matches_empty_environment():
    # the column is nullable and the spawn path already coerces None to {}
    assert container_spec(_svc(env=None)).fingerprint == container_spec(_svc(env={})).fingerprint


def test_absent_healthcheck_normalises_to_none():
    assert container_spec(_svc(healthcheck={})).healthcheck is None
    assert container_spec(_svc(healthcheck={})).fingerprint == container_spec(_svc(healthcheck=None)).fingerprint


def test_every_container_affecting_field_changes_the_fingerprint():
    base = container_spec(_svc()).fingerprint
    assert container_spec(_svc(image="img:2")).fingerprint != base
    assert container_spec(_svc(models=("a",))).fingerprint != base
    assert container_spec(_svc(devices=("gpu-1",))).fingerprint != base
    assert container_spec(_svc(env={"A": "1"})).fingerprint != base
    assert container_spec(_svc(healthcheck={"test": ["CMD", "true"]})).fingerprint != base


class _RecordingContainers:
    def __init__(self):
        self.kwargs = None

    def list(self, all=False):
        return []

    def run(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(id="container-1")


def test_spawn_applies_the_spec_to_the_container():
    containers = _RecordingContainers()
    client = SimpleNamespace(containers=containers)
    spec = container_spec(_svc(models=("a", "b"), env={"A": "1"}))

    spawn_service_container(
        client=client,
        image_id="img:1",
        worker_name="svc",
        worker_network="net",
        repository=SimpleNamespace(mounts={}, environment={"STORAGE": "x"}),
        spec=spec,
    )

    kwargs = containers.kwargs
    assert kwargs["labels"] == {SPEC_LABEL: spec.fingerprint}
    assert kwargs["command"] == "--load-model=a --load-model=b"
    assert kwargs["mem_limit"] == "1024m"
    assert kwargs["nano_cpus"] == 2_000_000_000
    # storage wiring wins over the user's environment
    assert kwargs["environment"] == {"A": "1", "STORAGE": "x"}


def test_spawn_attaches_the_devices_named_by_the_spec():
    # the spec is the single reader of the service row: a separate device list could disagree with
    # the fingerprint and label a container as something it was not built from
    containers = _RecordingContainers()
    client = SimpleNamespace(containers=containers)

    spawn_service_container(
        client=client,
        image_id="img:1",
        worker_name="svc",
        worker_network="net",
        repository=SimpleNamespace(mounts={}, environment={}),
        spec=container_spec(_svc(devices=("gpu-2", "gpu-1"))),
    )

    kwargs = containers.kwargs
    assert kwargs["runtime"] == "nvidia"
    assert [request["DeviceIDs"] for request in kwargs["device_requests"]] == [["gpu-1"], ["gpu-2"]]


def test_spawn_without_devices_leaves_the_nvidia_runtime_alone():
    containers = _RecordingContainers()
    client = SimpleNamespace(containers=containers)

    spawn_service_container(
        client=client,
        image_id="img:1",
        worker_name="svc",
        worker_network="net",
        repository=SimpleNamespace(mounts={}, environment={}),
        spec=container_spec(_svc(devices=())),
    )

    assert containers.kwargs["runtime"] is None
    assert containers.kwargs["device_requests"] is None
