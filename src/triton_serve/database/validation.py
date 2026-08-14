import logging

from sqlalchemy.orm import Session

from triton_serve.api.services.resources import get_gpu_info, get_machine_info
from triton_serve.database.model import Device, Machine

LOG = logging.getLogger(__name__)


def check_resources(session: Session):
    """
    Check if the resources saved in the database match the current resources.
    """
    # check if machine exists
    machine = session.query(Machine).first()
    assert machine is not None, "No machine found in the database"
    # check if machine resources match
    _, num_cpus, total_mem = get_machine_info()
    assert num_cpus == machine.num_cpus, f"The cpu count ({num_cpus}) does not match, expected {machine.num_cpus}"
    assert total_mem == machine.total_memory, (
        f"The total memory ({total_mem}) does not match, expected {machine.total_memory}"
    )

    # retrieve devices and check if they match
    saved_devices = session.query(Device).filter(Device.machine.has(host_id=machine.host_id)).all()
    try:
        node_devices = get_gpu_info()
    except Exception as e:
        LOG.warning("Failed to get GPU info: %s", e)
        node_devices = []
    assert len(node_devices) == len(saved_devices), (
        f"Node devices ({len(node_devices)}) do not match the saved devices ({len(saved_devices)})"
    )

    saved_by_uuid = {device.uuid: device for device in saved_devices}
    for node_device in node_devices:
        saved = saved_by_uuid.get(node_device.uuid)
        assert saved is not None, f"Device {node_device.uuid} is on the node but not in the database"
        assert saved.name == node_device.name, f"Device {node_device.uuid} name does not match"
        assert saved.memory == node_device.memory, f"Device {node_device.uuid} memory does not match"
        assert saved.index == node_device.index, f"Device {node_device.uuid} index does not match"
