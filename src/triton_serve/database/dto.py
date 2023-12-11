from typing import List


class GPU:
    def __init__(self, device_id: str, name: str, memory: int, index: int, *args, **kwargs):
        """
        :param device_id: UUID of the GPU
        :param name: name of the GPU
        :param memory: total memory of the GPU in MiB
        :param index: index of the GPU

        """
        self.device_id = device_id
        self.name = name
        self.memory = memory
        self.index = index

    # check if two GPUs are the same by comparing all their fields
    def __eq__(self, other):
        """

        :param other: GPU object to compare with

        """
        if isinstance(other, GPU):
            return (
                self.device_id == other.device_id
                and self.name == other.name
                and self.memory == other.memory
                and self.index == other.index
            )
        return False

    # sort by index
    def __lt__(self, other):
        if isinstance(other, GPU):
            return self.index < other.index
        return False

    # print the GPU object as a dictionary
    def __str__(self):
        return str(self.__dict__)


class Machine:
    def __init__(self, num_cpus: int, total_memory: int, gpus: List[GPU]):
        self.num_cpus = num_cpus
        self.total_memory = total_memory
        self.gpus = gpus

    # check if two Machine objects are the same by comparing all their fields
    def __eq__(self, other):
        self.gpus.sort()
        other.gpus.sort()
        if isinstance(other, Machine):
            # check if the two lists have the same elements even if they are in different order
            return (
                self.num_cpus == other.num_cpus and self.total_memory == other.total_memory and self.gpus == other.gpus
            )
        return False
