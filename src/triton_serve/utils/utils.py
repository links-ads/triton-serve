import subprocess
import csv
import io
import psutil
from typing import List
from triton_serve.database.models import Device, Machine


def get_gpu_info(executable: str, fields: List[str], field_format: str):
    output = []
    try:
        command = f"{executable} --query-gpu={','.join(fields)} --format={field_format}".split(" ")

        out = subprocess.run(command, capture_output=True, check=True)
        gpus_info = out.stdout.decode(encoding="utf-8")
        reader = csv.DictReader(io.StringIO(gpus_info), fieldnames=fields, skipinitialspace=True)
        gpus = []
        for line in reader:
            gpu = {
                "uuid": line["uuid"],
                "name": line["name"],
                "memory": int(line["memory.total"]),
                "index": int(line["index"]),
            }
            gpus.append(gpu)
        return gpus
    except Exception as e:
        return output


def get_machine_info():
    num_cpus = psutil.cpu_count(logical=True)
    # retyrn the total memory in bytes
    mem_info = psutil.virtual_memory()
    # convert to MiB
    total_mem = mem_info.total >> 20
    return (num_cpus, total_mem)


def list_gpus():
    gpu_executable: str = "nvidia-smi"
    gpu_format: str = "csv,noheader,nounits"
    gpu_fields: List[str] = ["index", "uuid", "memory.total", "name"]
    return get_gpu_info(gpu_executable, gpu_fields, gpu_format)
