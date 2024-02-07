import csv
import io
import socket
import subprocess

import psutil

from triton_serve.database.schema import DeviceCreateSchema


def get_gpu_info(
    executable: str = "nvidia-smi",
    field_format: str = "csv,noheader,nounits",
    fields: list[str] = None,
) -> list[DeviceCreateSchema]:
    """Get the GPU information using nvidia-smi.

    Args:
        executable (str, optional): The command to run. Defaults to "nvidia-smi".
        field_format (str, optional): what to include in the output. Defaults to "csv,noheader,nounits".
        fields (list[str], optional): which fields to query. Defaults to None.

    Returns:
        list[DeviceCreateSchema]: list of GPUs of this node.
    """
    fields = fields or ["index", "uuid", "memory.total", "name"]
    command = f"{executable} --query-gpu={','.join(fields)} --format={field_format}".split(" ")

    out = subprocess.run(command, capture_output=True, check=True)
    gpus_info = out.stdout.decode(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(gpus_info), fieldnames=fields, skipinitialspace=True)
    gpus = []
    for line in reader:
        gpu = DeviceCreateSchema(
            uuid=line["uuid"],
            name=line["name"],
            memory=int(line["memory.total"]),
            index=int(line["index"]),
        )
        gpus.append(gpu)
    return gpus


def get_machine_info() -> tuple[str, int, int]:
    """Get the machine information.

    Returns:
        tuple[str, int, int]: hostname, number of cpus, total memory.
    """
    num_cpus = psutil.cpu_count(logical=True)
    mem_info = psutil.virtual_memory()
    total_mem = mem_info.total >> 20
    hostname = socket.gethostname()
    return hostname, num_cpus, total_mem
