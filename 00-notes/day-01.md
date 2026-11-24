# Day 1 Revision Notes: GPU Components (T4)

## 1. The kitchen picture
GPU = big kitchen. Many cooks work at the same time on small, similar jobs.

| Word | Meaning | My T4 |
| --- | --- | --- |
| SM | One cooking station. Has cores, tensor cores, registers, shared memory | 40 SMs |
| Core (CUDA core) | One small calculator | 64 per SM = 2,560 total |
| Tensor core | Special machine that multiplies grids of numbers very fast. AI loves it | 8 per SM = 320 total |
| Thread | One cook doing one small job | up to 1,024 per SM |
| Warp | 32 threads doing the same step at the same moment | 32 warps per SM |
| Block | Group of threads (up to 1,024) given to one SM. Split into warps | |
| Grid | All the blocks of one job | |
| RT core | Ray tracing for games. Not used for AI. Ignore it | |

Threads running at once = 40 SMs x 1,024 = 40,960.

## 2. The memory ladder (fast and tiny -> slow and big)

| Memory | Picture | Size on T4 | Who can use it |
| --- | --- | --- | --- |
| Registers | A cook's own hands | 65,536 per SM (~64 per thread when full) | One thread only |
| Shared memory | The counter at one station | 64 KB per SM (48 KB per block) | Threads in one block |
| L2 cache | Pantry for the whole kitchen | 4 MB | All SMs, automatic |
| VRAM | Warehouse across the street | ~15 GB (14,911 MB) | Everything. Models live here |

Rule: the faster the memory, the smaller it is. Most data sits in VRAM, so most data is far away.

## 3. Key numbers I measured
- Calculating speed: ~20 trillion operations/second (Day 0 matmul)
- Memory speed: 243 GB/s (peak ~320, so I got ~76%)
- Balance: 20 trillion / 243 billion = about 80 calculations per byte

## 4. The big ideas
- Compute-bound: the maths units are the limit. Memory-bound: waiting for data is the limit.
- Arithmetic intensity = calculations per byte fetched.
  - Below ~80 -> memory-bound. Above ~80 -> compute-bound.
  - Adding two lists: ~0.17 -> memory-bound.
  - Big matrix multiply (4096): ~1,365 -> compute-bound.
- LLM text generation (decode) reads all weights for each token and does only ~1-2 calculations per weight -> memory-bound.
- Batching helps: many requests share one read of the weights.
- Three things slow a program: compute, memory bandwidth, overhead (CPU/Python sending work too slowly).
- Occupancy = how busy the SMs are. If a thread needs too many registers, fewer threads fit.
- Warp divergence: if threads in one warp take different if/else paths, the warp does both paths one after the other, so it is slower.

## 5. Habits and warnings
- Always use torch.cuda.synchronize() or CUDA events when timing GPU code. The GPU runs in the background, so plain time.time() lies.
- Run one warm-up first. The first GPU run is slow.
- nvidia-smi "GPU-Util" only says some job was running. It does not say how much of the chip is busy.
- T4 is compute capability 7.5 (Turing): no BF16, so use float16.

## 6. One-line memory trick
Cooks (cores) are fast. The warehouse (VRAM) is far. Most AI speed problems = cooks waiting for ingredients.