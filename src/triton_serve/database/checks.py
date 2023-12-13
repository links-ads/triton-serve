from typing import List
import subprocess
import csv
import io
import psutil
from triton_serve.database.dto import GPU, Machine
from triton_serve.database.queries.machines import get_machine_resources
from triton_serve.database.queries.devices import get_devices
from triton_serve.database.connection import get_connection


def get_gpu_info(executable: str, fields: List[str], field_format: str):
    output = []
    try:
        command = f"{executable} --query-gpu={','.join(fields)} --format={field_format}".split(" ")

        out = subprocess.run(command, capture_output=True, check=True)
        gpus_info = out.stdout.decode(encoding="utf-8")
        reader = csv.DictReader(io.StringIO(gpus_info), fieldnames=fields, skipinitialspace=True)
        gpus = []
        for line in reader:
            gpu = GPU(line["uuid"], line["name"], int(line["memory.total"]), int(line["index"]))
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


def check_resources():
    connection = get_connection()
    gpu_executable: str = "nvidia-smi"
    gpu_format: str = "csv,noheader,nounits"
    gpu_fields: List[str] = ["index", "uuid", "memory.total", "name"]

    # get the number of cpus and total memory
    num_cpus, total_mem = get_machine_info()
    # get the list of GPUs
    gpus = get_gpu_info(gpu_executable, gpu_fields, gpu_format)
    # create a Resources object
    current_resources = Machine(num_cpus, total_mem, gpus)

    saved_machine = get_machine_resources()
    host_id = saved_machine["host_id"]
    num_cpus = saved_machine["num_cpus"]
    total_mem = saved_machine["total_memory"]

    saved_gpus = get_devices(host_id)
    saved_resources = Machine(num_cpus, total_mem, saved_gpus)

    connection.close()

    return current_resources == saved_resources
