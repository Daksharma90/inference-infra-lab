## First, the one rule

For any job, work out one number:

```
intensity = calculations done ÷ bytes moved from VRAM
```

Then compare it with your T4's ridge point, **about 80**:
- **Below 80:** the cooks wait for ingredients. The job is **memory-bound**.
- **Above 80:** the road is wide enough and the cooks are the limit. The job is **compute-bound**.

Every exercise is the same two steps: count calculations, count bytes.

---

## Exercise 1(a): Adding two lists (fp16)

Take one element:
- Read a number from list 1: 2 bytes (fp16 = 2 bytes)
- Read a number from list 2: 2 bytes
- Write the result: 2 bytes
- Total: **6 bytes**
- Calculations: **1** (one addition)

Intensity = 1 ÷ 6 ≈ **0.17**

0.17 is far below 80, so this is **memory-bound**. It's the same story as your Day 2 kernel.

## Exercise 1(b): Big matrix multiply (4096 × 4096, fp16)

This is the opposite case, where the same numbers get used over and over.

**Calculations:** each of the 4096 × 4096 output numbers needs 4096 multiplies and 4096 adds.
- 4096 × 4096 × (2 × 4096) = about **137 billion** calculations.

**Bytes:** read matrix A, read matrix B, write matrix C. Each is 4096 × 4096 = 16.8 million numbers × 2 bytes = 33.5 MB.
- 3 × 33.5 MB = about **100 MB**.

Intensity = 137 billion ÷ 100 million ≈ **1,365**

Why so high? Each number we fetch from memory gets used **thousands of times** inside the chip (in registers and shared memory, the fast parts of your Day 1 ladder). We pay for the trip to the warehouse once and then cook with those ingredients thousands of times.

1,365 is far above 80, so this is **compute-bound**.

## Exercise 1(c): Matrix times one vector

A **vector** is just one column of numbers. Multiplying a big matrix by one vector means each matrix number meets the vector only **once**.

Think of 4096 × 4096 = 16.8 million matrix numbers:
- Each gets used in **1 multiply and 1 add** = 2 calculations, then it's never used again.
- Each is 2 bytes (fp16) and must be fetched from VRAM.
- (The vector is only 4096 numbers, which is tiny and stays in fast memory.)

Intensity = 2 calculations ÷ 2 bytes = **1**

This is **memory-bound**. It's the opposite of (b): big matrix times big matrix reuses every number thousands of times, while big matrix times one vector uses each number once.

| Job | Intensity | vs 80 | Verdict |
| --- | --- | --- | --- |
| (a) add two lists | 0.17 | far below | memory-bound |
| (b) matrix × matrix | 1,365 | far above | compute-bound |
| (c) matrix × vector | 1 | far below | memory-bound |

---

## Exercise 2: Intensity of (c) and its zone

We just worked it out: intensity = **1**.

- The T4 needs **80** to stay busy.
- The job gives **1**. That's **80 times too little**.

**How much speed do we get?** Use the roofline formula:

```
speed = memory speed × intensity = 256 GB/s × 1 = 256 billion calculations/second
```

The calculator can do about 20 trillion per second (your Day 0 number), so we use 256 billion ÷ 20,000 billion = about **1.3%** of the GPU's calculating power.

That means 98.7% of the cooks stand idle waiting for ingredients. That's the cost of a memory-bound job.

---

## Exercise 3: The link to LLMs (the most important part)

### Why is LLM text generation "matrix × vector"?

An LLM is made of big weight matrices (the model's numbers). To generate **one token** for **one user**:
- The current token becomes **one vector** of numbers.
- That vector gets multiplied by each big weight matrix, one after another.

So it's exactly case (c): a huge matrix times one vector. Every weight is read from VRAM, used once, then thrown away. Then the next token needs **all the weights read again**.

Intensity ≈ **1**, so decoding for one user is **memory-bound**.

### What changes with 8 users?

Say 8 users are all waiting for their next token. We can **stack their 8 vectors side by side** into one small matrix (4096 × 8) and multiply once.

- **Bytes moved:** the same big weight matrix, read **once**. No extra weight traffic.
- **Calculations:** 8 times more, because each weight is now used for 8 users.

Intensity goes from 1 to **8**, so we get about **8x more tokens per second** for roughly the same memory traffic.

**This is why inference servers batch many requests together.** One read of the weights serves many users. The road stays the same size, but each trip carries 8 orders instead of 1.

### How many users before it stops being memory-bound?

```
users needed ≈ ridge ÷ intensity per user = 80 ÷ 1 = about 80 users
```

So at around 80 users in one batch, the T4's cooks become the limit. Below that, adding users is almost free.

**Real-world caveat:** each user also needs **KV cache** (the memory of the conversation so far, which we'll learn on Day 4). That eats VRAM and limits how many users fit. So real batch sizes are often smaller than the ideal 80. This tension, between batching for speed and running out of memory, is the heart of what inference engines like vLLM manage.

### A quick real-world number

A model's whole weights are read every step, so for one user:

```
max tokens per second ≈ memory speed ÷ model size in bytes
```

For a small 1-billion-parameter model in fp16 (about 2 GB):
- 256 GB/s ÷ 2 GB = about **128 tokens/second**, even if nothing else is slow.

With 8 users in a batch, the **total** across everyone is about 8x that (about 1,000 tokens/second), because the weights are still read once. Real systems land below these numbers because of overhead and KV-cache reads, but the pattern holds.

---

## One-line summary

> Matrix × matrix reuses each number thousands of times, so it's compute-bound. Matrix × one vector uses each number once, so it's memory-bound. LLM decoding is matrix × vector, which is why servers batch many users: one read of the weights serves everyone.
