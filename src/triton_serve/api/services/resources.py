import csv
import io
import socket
import subprocess

import psutil

from triton_serve.api.dto import GPUDevice


def get_gpu_info(
    executable: str = "nvidia-smi",
    field_format: str = "csv,noheader,nounits",
    fields: list[str] = ["index", "uuid", "memory.total", "name"],
):
    output = []
    try:
        command = f"{executable} --query-gpu={','.join(fields)} --format={field_format}".split(" ")

        out = subprocess.run(command, capture_output=True, check=True)
        gpus_info = out.stdout.decode(encoding="utf-8")
        reader = csv.DictReader(io.StringIO(gpus_info), fieldnames=fields, skipinitialspace=True)
        gpus = []
        for line in reader:
            gpu = GPUDevice(
                uuid=line["uuid"],
                name=line["name"],
                memory=int(line["memory.total"]),
                index=int(line["index"]),
            )
            gpus.append(gpu)
        return gpus
    except Exception:
        return output


def get_machine_info():
    num_cpus = psutil.cpu_count(logical=True)
    mem_info = psutil.virtual_memory()
    total_mem = mem_info.total >> 20
    hostname = socket.gethostname()
    return hostname, num_cpus, total_mem
