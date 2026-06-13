# LLM inference verification

Cheaply verify that an untrusted **GPU** computed an LLM forward pass correctly, from a trusted **CPU** ("TEE") process, using **approximate matrix-multiplication checking**.

## Reproduce the benchmarks

### Preparation

```bash
cd flossy
conda create -n flossy python=3.10 -y
conda activate flossy

# any Llama/Qwen3-family causal LM works — swap the name below
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507
M=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/)
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1
```

### Inference only

Pure GPU inference floor (CUDA-graph decode, no verification); prints `ms/step`.

```bash
python bench_gpu_sweep.py --model "$M" --batch 16 --ns 64
```

- `--model` — local snapshot path of the model to run.
- `--batch` — batch size (tokens emitted per step); at batch 16 a "step" is 16 tokens.
- `--ns` — output lengths N to sweep, e.g. `--ns 64 256 512` benchmarks all three.

### End to end inference + verification

Prover on GPU + 4-worker verifier on CPU; prints the verifier `verdict : ACCEPT`
and the prover `per-request : … ms/tok`. The prover always decodes as a CUDA graph
(the one-time graph capture is excluded from the reported per-step time).

```bash
python run_e2e_spawn.py "$M" -n 64 --batch 16 --workers 4 --vthreads 8
```

- `-n` — number of tokens to generate and verify per request.
- `--batch` — batch size (the batch shares one prompt).
- `--workers` — verifier worker processes, each owning a layer slice; **4** is the sweet spot at batch 16.
- `--vthreads` — CPU threads per verifier worker.
