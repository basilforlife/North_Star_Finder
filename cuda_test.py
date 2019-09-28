import numpy as np
from numba import cuda

@cuda.jit
def add(x):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, x.size[0], stride):
        x[i] = x[i]*2*2 


n = 1000000
x = np.ones(n).astype(np.int64)
y = x + 1

d_x = cuda.to_device(x)
d_y = cuda.to_device(y)
d_out = cuda.device_array_like(d_x)

threads_per_block = 128
blocks_per_grid = 30

add[blocks_per_grid, threads_per_block](d_x)
print(d_out.copy_to_host())
