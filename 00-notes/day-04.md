# Day 4 Notes: How an LLM writes text (prefill, decode, KV cache)

## 1. The loop in one line
Prompt -> **prefill** (read the whole prompt at once) -> **decode** (write 1 token, add it, repeat).
A token is a small piece of text. The model writes one token at a time.

## 2. Prefill vs decode

| | Prefill | Decode |
| --- | --- | --- |
| What happens | Model reads the whole prompt at once | Model writes one token per step |
| Tokens per weight read | Many (e.g. 1,024) | 1 per user |
| Intensity | About 1,000 (above T4 bar of 80) | About 1 (far below 80) |
| Limit | Compute-bound | Memory-bound |
| Why | One read of the weights serves all the tokens | Every token re-reads all the weights for one token of work |

My T4 bar (ridge point) = about 80 calculations per byte. 80 is a fact about the GPU. Intensity is a fact about the job.
Intensity = calculations done / bytes moved (NOT divided by memory speed).
Batching: with B users decoding together, intensity is about B. About 80 users are needed to become compute-bound. Real limit is KV cache memory.

## 3. KV cache
- Attention needs K (keys) and V (values) of all earlier tokens.
- Without a cache, every step recomputes K and V for the whole text so far. Repeated work.
- With a cache, K and V are saved in VRAM. Each step computes them only for the new token.
- Saves: compute (no repeated work).
- Costs: VRAM. It grows with every token and with every user. Long prompts or many users can overflow it.
- Analogy: a student's notes. Without notes, re-listen to the whole lecture before every answer.

## 4. My results (Qwen2.5-0.5B, fp16, T4)
- Model size: about 1 GB. Ideal decode step (weights / 256 GB/s): 3.9 ms.
- Prefill: 1,024 tokens in 163.5 ms = 6,263 tokens/s (about 6 TFLOP/s, about 30% of the 20 TFLOP/s I measured on Day 0).
- Decode: 32.1 ms per token = 31 tokens/s. About 31 GB/s = only 12% of memory speed. About 0.15% of calculating speed.
- Real decode is 8.3x slower than the memory-only ideal.
- Prefill handles about 200x more tokens/s than decode.
- Cache vs no cache on short texts (32 to 512 new tokens): speedup only 1.1x to 1.2x. Time per token was flat: about 32.5 ms with cache, about 36-38 ms without.
- Cache shape (1, 2, 1057, 64) = 1 user, 2 KV heads, 1,057 tokens, head size 64.
- Cache size: about 12 KB per token (2 x 2 heads x 64 x 2 bytes x 24 layers). 1,057 tokens = about 13 MB. Tiny next to the 1 GB of weights.
- Long-prompt test (1,500-token prompt, cache vs no cache): (paste my numbers here)

## 5. Questions and answers

**Q1. Which is compute-bound and which is memory-bound? Why?**
Prefill is compute-bound. The weights are read once and serve many tokens, so calculations per byte is about 1,000, above the T4 bar of 80. Decode is memory-bound. Each weight read serves only one token, so intensity is about 1, far below 80. The cooks wait for ingredients.

**Q2. What does the KV cache save and what does it cost?**
It saves recomputing K and V for all earlier tokens at every step. It costs VRAM. The cache grows with every token and with every user, so long prompts or many users can use up the memory.

**Q3. Why does the cache speedup grow with longer text?**
With the cache, each new step processes 1 token. Without it, each step re-processes every token so far, like a mini prefill each time. Example: 1,500-token prompt and 32 new tokens = about 1,532 token-passes with cache vs about 48,500 without, roughly 32x more work. The longer the text, the more wasted work without the cache, so the speedup grows. In my small test it did not show (only 1.1x to 1.2x) because about 32 ms of fixed overhead per step hid the saving. Note: the no-cache version keeps the GPU busier, but that busy work is repeated work, so busy does not mean faster.

**Q4. Why was real decode (32.1 ms) slower than the ideal (3.9 ms)?**
Overhead. One decode step is hundreds of tiny GPU launches (24 layers x many small operations), driven one by one from Python. The GPU finishes each tiny job fast and then waits for the next instruction. Proof: only 12% of memory speed and about 0.15% of calculating speed were used, so neither memory nor calculating was the limit. This is the third slowdown (compute, memory, overhead).
Fixes: bigger batches (share the overhead), kernel fusion (fewer launches), CUDA graphs (record the step once and replay it).

## 6. Big lessons
1. Prefill = many tokens per weight read = compute-bound. Decode = one token per weight read = memory-bound.
2. Batching raises decode intensity (about equal to the number of users). KV cache memory is what limits batch size.
3. KV cache saves compute but costs VRAM.
4. On a small model, overhead (waiting for Python) can matter more than memory or compute.
5. Speed ideal for decode = memory speed / model size in bytes. Real speed is lower because of overhead and KV-cache reads.
6. Always measure: my guess that the cache speedup would grow was not visible in a small test.

## 7. One-line memory trick
Prefill = reading the whole book at once (many pages, one trip). Decode = writing one word at a time (one trip per word). The KV cache is the notes that stop you re-reading the book every time.