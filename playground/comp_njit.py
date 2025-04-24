import numpy as np
import time
from numba import njit, prange

@njit(parallel=True)
def numba_parallel(arr, out):
    for i in prange(arr.shape[0]):  # Use parallel execution
        out[i] = arr[i] ** 2

arr = np.random.rand(1000000).astype(np.float32)  # Use float32 for speed
out = np.empty_like(arr)  # Pre-allocated memory

# Run multiple times for better benchmarking
start = time.time()
for _ in range(100):
    numba_parallel(arr, out)
print(f"Numba Parallel Time (100 calls): {time.time() - start:.6f} sec")

# NumPy version
start = time.time()
for _ in range(100):
    out = arr ** 2
print(f"NumPy Time (100 calls): {time.time() - start:.6f} sec")
