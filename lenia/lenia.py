import ctypes

ctypes.CDLL("libcufft.so", mode = ctypes.RTLD_GLOBAL)
ctypes.CDLL("libcuda.so", mode = ctypes.RTLD_GLOBAL)
ctypes.CDLL("libcudart.so", mode = ctypes.RTLD_GLOBAL)
liblenia = ctypes.CDLL("lenia/liblenia.so")

liblenia.initialize.argtypes = [ctypes.c_int]
liblenia.initialize.restypes = None

liblenia.finalize.argtypes = []
liblenia.finalize.restypes = None

liblenia.run.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
liblenia.run.restypes = None

import torch
from matplotlib import pyplot as plt
class Lenia:
    def __init__(self, size:int, grid=None) -> None:
        self.size = size
        self.grid: torch.Tensor[torch.float] = grid if grid else torch.zeros(size,size, dtype=torch.float, device="cuda:0")
        liblenia.initialize(size)

    def run(self, steps:int) -> None:
        liblenia.run(steps, ctypes.cast(self.grid.data_ptr(), ctypes.POINTER(ctypes.c_float)))

    def show(self) -> None:
        plt.imshow(self.grid.reshape(self.size,self.size).detach().cpu().numpy())
        plt.show()

    def __del__(self) -> None:
        liblenia.finalize()


