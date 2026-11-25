## Q1. What is a kernel?
A function that runs on the GPU. The CPU launches it, then thousands of threads run the same function at the same time, each on its own data. The CPU does not wait unless I call synchronize().

## Q2. threadIdx.x, blockIdx.x, blockDim.x
- threadIdx.x = my position inside my block
- blockIdx.x = which block I am in
- blockDim.x = how many threads each block has (I choose this at launch, max 1024)
- Formula: i = blockIdx.x * blockDim.x + threadIdx.x
- Example: 256 threads per block, thread 10 in block 3 -> i = 3*256 + 10 = 778

## Q3. Why if i < n?
Blocks are launched whole, so the thread count rounds up (1,000,000 elements, 256 per block -> 3,907 blocks = 1,000,192 threads, 192 extra). Extra threads would touch memory past the end of the list, giving wrong answers or crashes. The "if" makes them do nothing. It is about safety, not speed.

## Q4. Why vector add matched memory bandwidth
Each element moves 12 bytes for 1 addition -> about 0.08 operations per byte. My T4 needs about 80 operations per byte to stay busy. So the cores wait for memory. Speed = memory speed (about 256 GB/s, about 80% of the 320 GB/s peak). This is memory-bound. About 0.1% of the GPU's calculating power was used.

## Experiment A table
tpb=1: 9.7 GB/s (1 of 32 lanes works, few threads at once)
tpb=32: 237 GB/s (full warps, but only 16 warps per SM)
tpb=128 to 1024: about 256 GB/s (memory bus is the limit)

## Lesson
A kernel that does not cover all elements can silently give wrong answers. Always check correctness before timing.

## Grid-stride loop results (1,048,576 elements)
- 1 block x 1 thread: 133.6 ms
- 1 block x 256 threads: 1.84 ms (only 1 of 40 SMs used)
- 160 blocks x 256 threads: 0.345 ms (all SMs used)
- About 390x faster from launch shape alone, same code.
- Small jobs are dominated by launch overhead, so they never reach full memory bandwidth.
- A grid-stride loop lets any number of threads cover any number of elements: start = my seat number, step = total threads.

Fill all SMs, with many full warps on each. Then memory bandwidth is the only limit.