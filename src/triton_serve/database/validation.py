import logging

from sqlalchemy.orm import Session

from triton_serve.api.services.resources import get_gpu_info, get_machine_info
from triton_serve.database.model import Device, Machine


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
    assert (
        total_mem == machine.total_memory
    ), f"The total memory ({total_mem}) does not match, expected {machine.total_memory}"

    # retrieve devices and check if they match
    host_devices = session.query(Device).filter(Device.machine.has(host_id=machine.host_id)).all()
    try:
        saved_devices = get_gpu_info()
    except Exception as e:
        log = logging.getLogger(__name__)
        log.warning("Failed to get GPU info: %s", e)
        saved_devices = []
    assert len(host_devices) == len(
        saved_devices
    ), f"Node devices ({len(host_devices)}) do not match the saved devices ({len(saved_devices)})"

    device_dict = {device.uuid: device for device in host_devices}
    for device in saved_devices:
        host_device = device_dict.get(device["uuid"])
        assert host_device is not None, f"Device {device['uuid']} not found on the node"
        assert host_device.name == device["name"], f"Device {device['uuid']} name does not match"
        assert host_device.memory == device["memory"], f"Device {device['uuid']} memory does not match"
        assert host_device.index == device["index"], f"Device {device['uuid']} index does not match"
        assert host_device.index == device["index"], f"Device {device['uuid']} index does not match"
