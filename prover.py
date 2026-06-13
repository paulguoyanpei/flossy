"""FLOSSY prover process (GPU, untrusted).

Loads the model on CUDA (fp16, matching FLOSSY's HalfTensor), runs greedy
generation, and streams every matmul output of every decode step to the
verifier over the shared-memory ring.

Transfer is pipelined (mirrors FLOSSY's ``DataTransfer.copy_thread``):
  * main thread: at each step, concatenate the recorded tensors into one fp32
    GPU buffer and launch an async copy to a pinned staging buffer on a side
    CUDA stream, then hand the work to a copy thread. Generation of step t+1
    overlaps the copy of step t.
  * copy thread: waits for the copy event, writes the pinned buffer into the
    shared-memory ring slot, and sends the step header (handling slot
    backpressure via the verifier's acks).
"""

from __future__ import annotations

import argparse
import os
import queue
import threading
import time

import torch

from capture import FlossyContext, StepBuffer, instrument
from common import add_common_args, build_prompt_ids, load_model
from transport import ControlChannel, Cursors, SharedRing, estimate_slot_elems, plan_grouped


def connect_with_retry(host: str, port: int, timeout: float = 60.0) -> ControlChannel:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            return ControlChannel.connect(host, port)
        except (ConnectionRefusedError, OSError) as e:
            last = e
            time.sleep(0.25)
    raise TimeoutError(f"could not connect to verifier at {host}:{port}: {last}")


def read_port(args) -> int:
    if args.port_file:
        deadline = time.time() + 120
        while time.time() < deadline:
            if os.path.exists(args.port_file) and os.path.getsize(args.port_file) > 0:
                with open(args.port_file) as f:
                    return int(f.read().strip())
            time.sleep(0.25)
        raise TimeoutError(f"port file {args.port_file} never appeared")
    return args.port


class ProverPipeline:
    """Overlaps GPU->CPU copy and slot-write/header-send with generation."""

    def __init__(self, ring: SharedRing, ctrl: ControlChannel, n_slots: int, slot_elems: int,
                 use_gpu_staging: bool = True, async_gpu_stage: bool = False, direct_dma: bool = False,
                 cursors: "Cursors | None" = None):
        self.ring = ring
        self.ctrl = ctrl
        self.cursors = cursors  # shm cursors (backpressure + readiness); None => no verifier
        self.cursor_mode = False  # set True for the cuda-graph path: publish `produced`
        # via the shm cursor instead of a per-step socket header (set in run()).
        self.n_slots = n_slots
        self.device = torch.cuda.current_device()
        self.async_gpu_stage = async_gpu_stage
        self.stream = torch.cuda.Stream()
        # direct_dma: page-lock the shm so the GPU DMAs straight into the slot,
        # skipping the pinned->shm CPU memcpy (the big cost at large batch).
        self.direct_dma = bool(direct_dma) and ring.register_cuda()
        # pinned staging only needed for the non-direct (staged memcpy) path
        self.pinned = (None if self.direct_dma
                       else [torch.empty(slot_elems, dtype=ring.torch_dtype, pin_memory=True) for _ in range(n_slots)])
        self.gpu_staging = (
            [torch.empty(slot_elems, dtype=ring.torch_dtype, device="cuda") for _ in range(n_slots)]
            if use_gpu_staging
            else None
        )
        # per-slot device shadow buffers for the CUDA-graph path: the in-graph cat
        # writes one fixed `staging` buffer, so the slow GPU->host DMA out of it
        # would block the next replay. Instead we do a fast on-device copy
        # staging -> dev_shadow[slot] (HBM ~2 TB/s) on the replay stream, then DMA
        # the shadow -> shm on a side stream so the PCIe transfer overlaps the next
        # replay. Allocated lazily (needs the captured staging size). Slot reuse is
        # already gated by _await_slot (ack => DMA done), so no extra event needed.
        self.dev_shadow: list[torch.Tensor] | None = None
        self.free_q: queue.Queue[int] = queue.Queue()
        for i in range(n_slots):
            self.free_q.put(i)
        self.work_q: queue.Queue = queue.Queue()
        self.acked = -1
        self.ack_thread: threading.Thread | None = None  # dedicated ack receiver (graph path)
        self.capture_wall = 0.0  # one-time CUDA-graph capture time (excluded from per-request cost)
        self._plan_sig = None  # cached grouping plan signature (layout is constant across decode steps)
        self._plan = None
        self.copy_s = 0.0  # cumulative copy-thread wall time
        self.t_sync = self.t_ack = self.t_slot = self.t_send = 0.0
        self.stage_q: queue.Queue | None = None
        self.stage_thread: threading.Thread | None = None
        if self.async_gpu_stage:
            self.stage_q = queue.Queue()
            self.stage_thread = threading.Thread(target=self._stage_loop, daemon=True)
            self.stage_thread.start()
        self.thread = threading.Thread(target=self._copy_loop, daemon=True)
        self.thread.start()

    def _stage_loop(self) -> None:
        torch.cuda.set_device(self.device)
        while True:
            item = self.stage_q.get()
            if item is None:
                break
            ready_event, pinned_idx, slot_idx, n, header, ordered_names, buf = item
            with torch.cuda.stream(self.stream):
                self.stream.wait_event(ready_event)
                pieces = [buf.fetch(name).reshape(-1).to(self.ring.torch_dtype) for name in ordered_names]
                if self.gpu_staging is None:
                    flat = torch.cat(pieces)
                else:
                    flat = self.gpu_staging[pinned_idx][:n]
                    torch.cat(pieces, out=flat)
                self.pinned[pinned_idx][:n].copy_(flat, non_blocking=True)
                for name in ordered_names:
                    buf.fetch(name).record_stream(self.stream)
                event = torch.cuda.Event()
                event.record(self.stream)
            self.work_q.put((event, pinned_idx, slot_idx, n, header))

    def _copy_loop(self) -> None:
        if self.direct_dma:
            return self._copy_loop_direct()
        while True:
            item = self.work_q.get()
            if item is None:
                break
            event, pinned_idx, slot_idx, n, header = item
            t0 = time.time()
            event.synchronize()  # wait for GPU -> pinned
            t1 = time.time()
            # backpressure: this slot last held step (step - n_slots)
            while header["step"] - self.n_slots > self._consumed():
                time.sleep(5e-5)
            t2 = time.time()
            self.ring.slot(slot_idx)[:n].copy_(self.pinned[pinned_idx][:n])
            t3 = time.time()
            self.ctrl.send(header)
            self.free_q.put(pinned_idx)
            t4 = time.time()
            self.t_sync += t1 - t0
            self.t_ack += t2 - t1
            self.t_slot += t3 - t2
            self.t_send += t4 - t3
            self.copy_s += t4 - t0

    def _copy_loop_direct(self) -> None:
        # direct DMA: the GPU already wrote the slot; wait for the DMA, then signal
        # readiness. cursor_mode -> bump the shm `produced` cursor (a memory write,
        # no socket); else -> send the per-step header over the socket.
        while True:
            item = self.work_q.get()
            if item is None:
                break
            event, slot_idx, n, step, header = item
            t0 = time.time()
            event.synchronize()  # wait for GPU -> shm DMA
            t1 = time.time()
            if self.cursor_mode:
                if self.cursors is not None:
                    self.cursors.produced = step
            else:
                self.ctrl.send(header)
            t2 = time.time()
            self.t_sync += t1 - t0
            self.t_send += t2 - t1
            self.copy_s += t2 - t0

    # --- backpressure via the shm `consumed` cursor (no ack thread) --------- #
    # The verifier writes its `consumed` step into shared memory; the prover reads
    # it here for slot reuse. No socket recv on the prover at all -> the replay
    # loop never blocks waiting on the verifier (it only blocks if it would lap
    # the ring, which it won't while the verifier keeps up).
    def start_ack_loop(self) -> None:  # kept for call-site compatibility; no-op now
        pass

    def finish_ack_loop(self) -> None:
        pass

    def _consumed(self) -> int:
        return self.cursors.consumed if self.cursors is not None else (1 << 30)

    def _await_slot(self, step: int) -> None:
        """Backpressure: spin on the verifier's shm `consumed` cursor (a memory
        read, no socket) until the slot this step will reuse has been consumed."""
        while step - self.n_slots > self._consumed():
            time.sleep(5e-5)

    def graph_sink(self, step: int, staging: torch.Tensor, n: int, header: dict | None = None) -> None:
        """Transfer for the CUDA-graph decode path: the grouped data is already
        gathered (by the cat captured inside the graph) in `staging` (GPU).

        To overlap the GPU->host transfer with the next replay (it would otherwise
        block it -- the next cat reuses the single `staging` buffer), we:
          1. copy `staging` -> `dev_shadow[slot]` on the replay stream (device-to-
             device, HBM bandwidth, ~cheap), which frees `staging` immediately;
          2. DMA `dev_shadow[slot]` -> shm on the side stream, so the slow PCIe
             transfer runs concurrently with subsequent replays.
        The copy thread waits on the side-stream event to send the header.
        Backpressure (`_await_slot`) gates slot/shadow reuse on the ack counter, so
        the main replay thread never blocks in a socket recv. Requires direct_dma."""
        self._await_slot(step)
        slot = step % self.n_slots
        if self.dev_shadow is None:
            self.dev_shadow = [torch.empty_like(staging) for _ in range(self.n_slots)]
        shadow = self.dev_shadow[slot]
        cur = torch.cuda.current_stream()
        shadow[:n].copy_(staging[:n])  # d2d on the replay stream: frees `staging` fast
        self.stream.wait_stream(cur)   # side stream waits only for the d2d, not the next replay
        with torch.cuda.stream(self.stream):
            self.ring.slot(slot)[:n].copy_(shadow[:n], non_blocking=True)  # GPU -> registered shm (PCIe)
            event = torch.cuda.Event()
            event.record(self.stream)
        self.work_q.put((event, slot, n, step, header))

    def sink(self, step: int, buf: StepBuffer) -> None:
        # Group tensors by projection so each group is contiguous in the slot.
        # The grouping plan AND header are constant across decode steps, so cache
        # them (keyed by a cheap layout signature) and send a compact header on
        # hits; only the prefill / first-decode steps recompute + send full.
        first = buf.fetch(buf.order[0])
        sig = (len(buf.order), tuple(first.shape))
        slot = step % self.n_slots
        if sig == self._plan_sig:
            ordered_names, n = self._plan
            header = {"type": "step", "step": step, "slot": slot, "total_len": n, "same": True}
        else:
            header, ordered_names, n = plan_grouped(buf, step, self.n_slots, step)
            self._plan_sig = sig
            self._plan = (ordered_names, n)
        if self.direct_dma:
            # backpressure: wait until this slot (last held by step-n_slots) is
            # consumed (shm cursor), then DMA straight from GPU into the shm slot.
            while step - self.n_slots > self._consumed():
                time.sleep(5e-5)
            cur = torch.cuda.current_stream()
            self.stream.wait_stream(cur)
            with torch.cuda.stream(self.stream):
                pieces = [buf.fetch(name).reshape(-1) for name in ordered_names]
                if self.gpu_staging is None:
                    flat = torch.cat(pieces)
                else:
                    flat = self.gpu_staging[slot][:n]
                    torch.cat(pieces, out=flat)
                self.ring.slot(slot)[:n].copy_(flat, non_blocking=True)  # GPU -> shm DMA
                event = torch.cuda.Event()
                event.record(self.stream)
            self.work_q.put((event, slot, n, step, header))
            return
        pinned_idx = self.free_q.get()  # blocks -> bounds lookahead to n_slots
        cur = torch.cuda.current_stream()
        if self.async_gpu_stage:
            ready_event = torch.cuda.Event()
            ready_event.record(cur)
            self.stage_q.put((ready_event, pinned_idx, step, n, header, ordered_names, buf))
        else:
            self.stream.wait_stream(cur)
            with torch.cuda.stream(self.stream):
                # recorded tensors are already the ring dtype (fp16), so skip the
                # per-tensor .to() (a no-op that still costs ~290 dispatches/step)
                pieces = [buf.fetch(name).reshape(-1) for name in ordered_names]
                if self.gpu_staging is None:
                    flat = torch.cat(pieces)
                else:
                    flat = self.gpu_staging[pinned_idx][:n]
                    torch.cat(pieces, out=flat)
                self.pinned[pinned_idx][:n].copy_(flat, non_blocking=True)
                event = torch.cuda.Event()
                event.record(self.stream)
            self.work_q.put((event, pinned_idx, step, n, header))

    def finish(self) -> None:
        if self.async_gpu_stage:
            self.stage_q.put(None)
            self.stage_thread.join()
        self.work_q.put(None)
        self.thread.join()


@torch.inference_mode()
def run_graph(model, input_ids, args, pipe: "ProverPipeline", captured: dict, transfer: bool = True):
    """Manual static-cache decode with a CUDA graph: prefill eager (transferred
    like the generate path), then capture ONE graph of (decode forward + grouped
    cat -> staging) and replay it per step. The captured cat reads the recorded
    intermediates each replay, so the transfer's gather is inside the graph (no
    per-step CPU launch). Returns (out_ids, n_steps). Fixed length (no EOS);
    greedy; lockstep via lm_head.

    The transfer is cheap (a microbench puts the in-graph cat at ~0.01 ms and the
    device->host DMA at ~0.05 ms): the decode step is GPU-bound at ~10 ms for a
    4B model, so the gather+DMA hide under it. Acks are drained on a separate
    thread (``start_ack_loop``) and backpressure spins on the ack counter, so the
    replay loop -- which also issues ``graph.replay()`` -- never blocks in a
    socket recv and never starves the GPU."""
    from transformers import StaticCache

    B, P = input_ids.shape
    N = args.max_new_tokens
    dev = input_ids.device
    n_slots = pipe.n_slots
    cache = StaticCache(config=model.config, max_batch_size=B, max_cache_len=P + N,
                        device=dev, dtype=torch.float16)

    def fwd(cur, pos):
        return model(input_ids=cur, cache_position=pos, past_key_values=cache,
                     use_cache=True, logits_to_keep=1).logits

    pipe.start_ack_loop()  # drain acks off the main (replay) thread

    # ---- prefill (eager) -> transport step 0 ----
    logits = fwd(input_ids, torch.arange(P, device=dev))
    buf = captured["buf"]
    if transfer:
        header_pf, ordered_pf, n_pf = plan_grouped(buf, 0, n_slots, 0)
        if pipe.cursor_mode:  # send the prefill grouping plan once (no per-step header)
            pipe.ctrl.send({"type": "plan", "which": "prefill", "groups": header_pf["groups"]})
        pf_staging = torch.cat([buf.fetch(nm).reshape(-1) for nm in ordered_pf])
        pipe.graph_sink(0, pf_staging, n_pf, header_pf)
    cur_tok = logits[:, -1:].argmax(-1)  # (B,1) first generated token
    out_ids = [input_ids, cur_tok]

    # ---- capture decode graph (one-time, prompt-independent -> can move to a
    # server-startup warmup; we time it separately so the per-request cost can be
    # reported with the capture excluded) ----
    static_tok = cur_tok.clone()
    static_pos = torch.zeros(1, dtype=torch.long, device=dev)
    static_pos.fill_(P)
    torch.cuda.synchronize()
    t_cap0 = time.time()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):  # warmup for allocator stability
        for _ in range(3):
            fwd(static_tok, static_pos)
    torch.cuda.current_stream().wait_stream(side)

    dec_buf = captured["buf"]
    if transfer:
        header_dec, ordered_dec, n_dec = plan_grouped(dec_buf, 1, n_slots, 1)
        staging = torch.empty(n_dec, dtype=torch.float16, device=dev)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_logits = fwd(static_tok, static_pos)
        if transfer:
            gbuf = captured["buf"]
            torch.cat([gbuf.fetch(nm).reshape(-1) for nm in ordered_dec], out=staging)
    torch.cuda.synchronize()
    pipe.capture_wall = time.time() - t_cap0
    if transfer and pipe.cursor_mode:  # send the decode plan + capture cost once
        pipe.ctrl.send({"type": "plan", "which": "decode", "groups": header_dec["groups"],
                        "capture_wall": pipe.capture_wall})

    # ---- replay loop: decode steps 1..N-1 ----
    for k in range(N - 1):
        step = k + 1
        static_pos.fill_(P + k)
        graph.replay()  # forward [+ cat -> staging] (default stream)
        if transfer:
            header = header_dec if step == 1 else {"type": "step", "step": step,
                                                   "slot": step % n_slots, "total_len": n_dec, "same": True}
            pipe.graph_sink(step, staging, n_dec, header)
        nxt = static_logits[:, -1:].argmax(-1)
        static_tok.copy_(nxt)
        out_ids.append(nxt.clone())

    return torch.cat(out_ids, dim=1), N


def main() -> int:
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--threads", type=int, default=4,
                    help="prover CPU threads (kept low so the GPU-side memcpy does not contend with the verifier)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    if args.threads > 0:
        torch.set_num_threads(min(args.threads, torch.get_num_threads()))

    tokenizer, model = load_model(args.model, "cuda:0")
    input_ids = build_prompt_ids(tokenizer, args.prompt, "cuda:0")
    if args.batch > 1:
        input_ids = input_ids.repeat(args.batch, 1)
    batch, prompt_len = input_ids.shape

    port = read_port(args)
    ctrl = connect_with_retry(args.host, port)

    slot_elems = estimate_slot_elems(model.config, batch, prompt_len, args.max_new_tokens,
                                     transfer_scores=args.transfer_scores)
    ring = SharedRing.create(args.shm_name, args.slots, slot_elems, dtype="float16")
    cur_name = args.shm_name + "_cur"
    cursors = Cursors.create(cur_name)
    ctrl.send({"type": "hello", "shm_name": args.shm_name, "n_slots": args.slots,
               "slot_elems": slot_elems, "dtype": "float16", "cursor_name": cur_name,
               "cursor_mode": True})

    # wait for the verifier to finish its one-time Br precompute so the timing
    # below reflects the steady-state pipeline, not setup
    ready = ctrl.recv()
    assert ready["type"] == "ready"

    pipe = ProverPipeline(ring, ctrl, args.slots, slot_elems,
                          direct_dma=True, cursors=cursors)
    pipe.cursor_mode = True  # graph path: publish `produced` via the shm cursor
    captured: dict = {}
    assert pipe.direct_dma, "cuda-graph requires direct DMA (cudaHostRegister failed?)"
    ctx = FlossyContext(role="prover", s=args.s, tol=args.tol,
                        record_scores=args.transfer_scores,
                        sink=lambda st, b: captured.__setitem__("buf", b))
    instrument(model, ctx)

    torch.cuda.synchronize()
    t0 = time.time()
    out, n_steps_graph = run_graph(model, input_ids, args, pipe, captured)
    pipe.finish()  # drain the copy thread (all headers sent)
    total_wall = time.time() - t0

    # share the one-time graph-capture cost so the verifier can report an
    # end-to-end-excluding-setup wall (its verify wall spans the prover's capture)
    cursors.done = 1  # end-of-stream via the shm cursor (capture_wall already sent)
    while True:
        msg = ctrl.recv()
        if msg["type"] == "bye":
            break

    tokens = out[0].tolist()
    n_steps = n_steps_graph
    cap = pipe.capture_wall
    # per-request cost with the one-time graph capture excluded (i.e. if capture
    # were done once at server startup and reused across requests)
    req_wall = total_wall - cap
    new_tokens = len(tokens) - prompt_len
    # single authoritative total: per-request wall, one-time graph capture excluded
    print(f"per-request : {req_wall:.2f}s  ({new_tokens} tokens, {n_steps / req_wall:.1f} tok/s, "
          f"{req_wall / n_steps * 1000:.2f} ms/tok)", flush=True)

    ring.close()
    cursors.close()
    ctrl.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
