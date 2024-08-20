from typing import Optional, Sized, Iterator

from torch.utils.data import Sampler
from torch.utils.data.sampler import T_co


# 构建子集顺序采样器
class SubsetSequentialSampler(Sampler[int]):

    def __init__(self, indices) -> None:
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class SubsetTargetSequentialSampler(Sampler[int]):

    def __init__(self, indices,targets) -> None:
        self.indices = indices

    def __iter__(self) -> Iterator[T_co]:
        pass