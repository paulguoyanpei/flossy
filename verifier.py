"""FLOSSY verifier process (CPU "TEE").

Verifies every matmul of every decode step against the tensors streamed by the
prover, without running the model forward: the residual stream is rebuilt by
prefix-summing the transferred block outputs (`lean_verifier.LeanVerifier`), and
tokens come from the transferred `lm_head` argmax (lockstep with the GPU).

With ``--workers N`` (N>1) it runs a coordinator + N worker processes, each owning
a contiguous slice of layers and reading the same shared-memory ring; the lean
verifier shards the residual (each worker rebuilds only the boundary prefix below
its slice), so only a per-step barrier is needed. A single torch process plateaus
at ~8-16 threads on the batched checks, so at large batch sharding across processes
uses the otherwise-idle cores. ``--workers 1`` runs the checks inline.
"""

from __future__ import annotations

import argparse
import gc
import queue
import time
import traceback

import torch
import torch.multiprocessing as mp

from common import add_common_args, build_prompt_ids, load_model
from lean_verifier import LeanVerifier
from transport import ControlChannel, Cursors, SharedRing, read_grouped

# seconds to wait for a worker ready/done/result before aborting instead of hanging
WORKER_TIMEOUT = 600.0


def _layer_ranges(n_layers: int, n_workers: int) -> list[tuple[int, int]]:
    base, extra = divmod(n_layers, n_workers)
    ranges, lo = [], 0
    for w in range(n_workers):
        hi = lo + base + (1 if w < extra else 0)
        ranges.append((lo, hi))
        lo = hi
    return ranges


# --------------------------------------------------------------------------- #
# Single-process verification loop
# --------------------------------------------------------------------------- #
def _recv_plans(ctrl) -> dict:
    """Cursor mode: the prover sends the prefill + decode grouping plans (and the
    capture cost) ONCE at setup, instead of a header per step."""
    plans = {"capture_wall": 0.0}
    for _ in range(2):
        msg = ctrl.recv()
        assert msg["type"] == "plan", f"expected plan, got {msg.get('type')}"
        plans[msg["which"]] = msg["groups"]
        if "capture_wall" in msg:
            plans["capture_wall"] = msg["capture_wall"]
    return plans


def run_single(args, ctrl, ring, input_ids, gen, cursors, cursor_mode) -> "LeanVerifier":
    lv = LeanVerifier(model=args._model, s=args.s, tol=args.tol, generator=gen)
    lv.set_prompt(input_ids[0])
    del args._model, input_ids
    gc.collect()
    ctrl.send({"type": "ready"})
    src_t = compute_t = 0.0
    cap = 0.0
    if cursor_mode:
        plans = _recv_plans(ctrl)            # blocks for the prover's prefill+capture+setup
        t0 = time.time()                     # start the clock only once the prover is producing
        last = -1
        while True:
            p = cursors.produced
            if last >= p:                       # caught up: stop if done, else poll
                if cursors.done:
                    break
                time.sleep(2e-5)
                continue
            ts = time.time()
            step = last + 1
            groups_meta = plans["prefill"] if step == 0 else plans["decode"]
            groups = read_grouped(ring, step, groups_meta)
            src_t += time.time() - ts
            tc = time.time()
            lv.verify_step(groups)
            compute_t += time.time() - tc
            cursors.consumed = step             # free the slot (shm, no socket)
            last = step
    else:
        last_groups = None
        t0 = time.time()
        while True:
            ts = time.time()
            msg = ctrl.recv()
            if msg["type"] == "done":
                cap = msg.get("capture_wall", 0.0)
                break
            if msg.get("same"):
                groups_meta = last_groups
            else:
                last_groups = groups_meta = msg["groups"]
            groups = read_grouped(ring, msg["slot"], groups_meta)
            src_t += time.time() - ts
            tc = time.time()
            lv.verify_step(groups)
            compute_t += time.time() - tc
            cursors.consumed = msg["step"]      # free the slot (shm, no socket ack)
    dt = time.time() - t0
    lv._timing = (dt, src_t, compute_t)
    lv._capture_wall = cap
    return lv


# --------------------------------------------------------------------------- #
# Multi-process worker (one contiguous slice of layers)
# --------------------------------------------------------------------------- #
def _worker(wid, lo, hi, checks_lmhead, args, hello, task_q, done_q, result_q):
    ring = None
    try:
        if args.threads > 0:
            torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        gen = torch.Generator().manual_seed(args.seed)
        tokenizer, model = load_model(args.model, "cpu")
        input_ids = build_prompt_ids(tokenizer, args.prompt, "cpu")
        lv = LeanVerifier(model, s=args.s, tol=args.tol, generator=gen,
                          layers=range(lo, hi), checks_lmhead=checks_lmhead)
        lv.set_prompt(input_ids[0])
        del model, tokenizer, input_ids
        gc.collect()
        ring = SharedRing.attach(hello["shm_name"], hello["n_slots"], hello["slot_elems"],
                                 dtype=hello.get("dtype", "float16"))
        last_groups = None
        t_wait = t_read = t_verify = 0.0
        done_q.put(("ready", wid))
        while True:
            tw = time.time()
            task = task_q.get()
            t_wait += time.time() - tw
            if task is None:
                break
            step, slot, gmeta = task
            if gmeta is not None:
                last_groups = gmeta
            tr = time.time()
            groups = read_grouped(ring, slot, last_groups)
            t_read += time.time() - tr
            tv = time.time()
            lv.verify_step(groups)
            t_verify += time.time() - tv
            done_q.put(("done", wid, step))
        result_q.put((wid, lv.n_checks, lv.n_failures, lv.worst_ratio, lv.worst_name,
                      lv.failures[:20], t_wait, t_read, t_verify))
    except BaseException:
        tb = traceback.format_exc()
        try:
            done_q.put(("error", wid, tb))
        except Exception:
            pass
        try:
            result_q.put(("error", wid, tb))
        except Exception:
            pass
        raise
    finally:
        if ring is not None:
            ring.close()


def _stop_workers(task_qs, procs, join_timeout: float = 5.0) -> None:
    for q in task_qs:
        try:
            q.put_nowait(None)
        except Exception:
            pass
    for p in procs:
        p.join(timeout=join_timeout)
    for p in procs:
        if p.is_alive():
            p.terminate()
    for p in procs:
        p.join(timeout=join_timeout)


def _live_worker_summary(procs) -> str:
    return ", ".join(
        f"{wid}:pid={p.pid}:{'alive' if p.is_alive() else f'exit={p.exitcode}'}"
        for wid, p in enumerate(procs)
    )


def _get_worker_event(done_q, procs, timeout: float, phase: str):
    deadline = time.time() + timeout
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError(f"timed out after {timeout:.0f}s waiting for worker {phase}; "
                               f"workers: {_live_worker_summary(procs)}")
        try:
            item = done_q.get(timeout=min(1.0, remaining))
        except queue.Empty:
            for wid, proc in enumerate(procs):
                if proc.exitcode not in (None, 0):
                    raise RuntimeError(f"worker {wid} (pid {proc.pid}) exited with code "
                                       f"{proc.exitcode} while waiting for {phase}")
            continue
        if item[0] == "error":
            raise RuntimeError(f"worker {item[1]} failed while waiting for {phase}:\n{item[2]}")
        return item


def _get_worker_result(result_q, procs, timeout: float):
    deadline = time.time() + timeout
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError(f"timed out after {timeout:.0f}s waiting for worker result; "
                               f"workers: {_live_worker_summary(procs)}")
        try:
            item = result_q.get(timeout=min(1.0, remaining))
        except queue.Empty:
            for wid, proc in enumerate(procs):
                if proc.exitcode not in (None, 0):
                    raise RuntimeError(f"worker {wid} (pid {proc.pid}) exited with code "
                                       f"{proc.exitcode} before returning its result")
            continue
        if item[0] == "error":
            raise RuntimeError(f"worker {item[1]} failed before returning its result:\n{item[2]}")
        return item


def run_multi(args, ctrl, hello, cursors, cursor_mode) -> dict:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(args.model)
    ranges = _layer_ranges(cfg.num_hidden_layers, args.workers)

    ctx = mp.get_context("spawn")
    task_qs = [ctx.Queue() for _ in range(args.workers)]
    done_q = ctx.Queue()
    result_q = ctx.Queue()
    procs = []
    for w, (lo, hi) in enumerate(ranges):
        p = ctx.Process(target=_worker,
                        args=(w, lo, hi, w == args.workers - 1, args, hello, task_qs[w], done_q, result_q))
        p.start()
        procs.append(p)

    finished = False
    try:
        for _ in range(args.workers):  # wait until all workers loaded + attached
            tag, wid, *_ = _get_worker_event(done_q, procs, WORKER_TIMEOUT, "startup")
            if tag != "ready":
                raise RuntimeError(f"unexpected worker event during startup: {(tag, wid)}")
        ctrl.send({"type": "ready"})

        src_t = compute_t = 0.0
        cap = 0.0
        steps = 0
        last_groups = None

        def barrier(step):  # wait until all workers finished `step`
            for _ in range(args.workers):
                tag, wid, *rest = _get_worker_event(done_q, procs, WORKER_TIMEOUT, f"step {step}")
                if tag != "done":
                    raise RuntimeError(f"unexpected worker event during step {step}: {(tag, wid)}")

        n_slots = hello["n_slots"]
        if cursor_mode:
            # Prover is decoupled (runs ahead, publishing `produced`); the coordinator
            # polls it, feeds each new step to the workers, barriers, and frees the
            # slot via `consumed`. No socket in the hot loop -> no per-step round-trip.
            plans = _recv_plans(ctrl)
            cap = plans["capture_wall"]
            last = -1
            t0 = time.time()
            while True:
                ts = time.time()
                while last >= cursors.produced and not cursors.done:
                    time.sleep(2e-5)
                src_t += time.time() - ts
                if last >= cursors.produced:   # caught up and done
                    break
                step = last + 1
                gmeta = plans["prefill"] if step == 0 else (plans["decode"] if step == 1 else None)
                for q in task_qs:
                    q.put((step, step % n_slots, gmeta))
                tc = time.time()
                barrier(step)
                compute_t += time.time() - tc
                cursors.consumed = step
                last = step
            steps = last + 1
            dt = time.time() - t0
        else:
            # socket header mode (non-graph prover): recv a header per step.
            prev = None
            t0 = time.time()
            while True:
                ts = time.time()
                msg = ctrl.recv()
                src_t += time.time() - ts
                done = msg["type"] == "done"
                if prev is not None:           # finish the step issued before this recv
                    tc = time.time()
                    barrier(prev)
                    compute_t += time.time() - tc
                    cursors.consumed = prev
                    prev = None
                if done:
                    cap = msg.get("capture_wall", 0.0)
                    break
                gmeta = None
                if not msg.get("same"):
                    last_groups = gmeta = msg["groups"]
                for q in task_qs:
                    q.put((msg["step"], msg["slot"], gmeta))
                prev = msg["step"]
                steps += 1
            dt = time.time() - t0

        for q in task_qs:
            q.put(None)
        agg = {"n_checks": 0, "n_failures": 0, "worst_ratio": 0.0, "worst_name": "", "failures": [],
               "timing": (dt, src_t, compute_t), "capture_wall": cap, "steps": steps}
        for _ in range(args.workers):
            wid, nc, nf, wr, wn, fails, *_ = _get_worker_result(result_q, procs, WORKER_TIMEOUT)
            agg["n_checks"] += nc
            agg["n_failures"] += nf
            if wr > agg["worst_ratio"]:
                agg["worst_ratio"], agg["worst_name"] = wr, wn
            agg["failures"].extend(fails)
        finished = True
        for p in procs:
            p.join(timeout=5.0)
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5.0)
        return agg
    finally:
        if not finished:
            _stop_workers(task_qs, procs)


def main() -> int:
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--workers", type=int, default=1,
                    help="verifier worker processes (shard layers); 1 = inline single process")
    ap.add_argument("--threads", type=int, default=16,
                    help="CPU threads per process (batched checks peak ~8-16; use fewer per worker)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    # control socket: bind, advertise the chosen port, accept the prover
    pending, port = ControlChannel.listen(args.host, args.port)
    if args.port_file:
        with open(args.port_file, "w") as f:
            f.write(str(port))
    ctrl = pending.accept()
    hello = ctrl.recv()
    assert hello["type"] == "hello"

    cursors = Cursors.attach(hello["cursor_name"])  # write `consumed` instead of socket acks
    cursor_mode = hello.get("cursor_mode", False)   # readiness via shm `produced` (no per-step recv)
    if args.workers > 1:
        agg = run_multi(args, ctrl, hello, cursors, cursor_mode)
    else:
        if args.threads > 0:
            torch.set_num_threads(min(args.threads, torch.get_num_threads()))
        tokenizer, model = load_model(args.model, "cpu")
        args._model = model
        input_ids = build_prompt_ids(tokenizer, args.prompt, "cpu")
        del model, tokenizer
        ring = SharedRing.attach(hello["shm_name"], hello["n_slots"], hello["slot_elems"],
                                 dtype=hello.get("dtype", "float16"))
        lv = run_single(args, ctrl, ring, input_ids, gen, cursors, cursor_mode)
        agg = {"n_checks": lv.n_checks, "n_failures": lv.n_failures, "worst_ratio": lv.worst_ratio,
               "worst_name": lv.worst_name, "failures": lv.failures[:20], "timing": lv._timing,
               "capture_wall": getattr(lv, "_capture_wall", 0.0), "steps": lv.step + 1}
        ring.close()

    cursors.close()
    ctrl.send({"type": "bye"})

    # the prover prints the single authoritative timing; here just the verdict
    print(f"verdict : {'ACCEPT' if agg['n_failures'] == 0 else 'REJECT'}", flush=True)

    ctrl.close()
    return 0 if agg["n_failures"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
