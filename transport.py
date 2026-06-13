"""Cross-process transport for FLOSSY: shared-memory data plane + socket control.

The prover (GPU, untrusted) and verifier (CPU "TEE") run as two independent OS
processes. Bulk per-step tensors travel through a named POSIX shared-memory ring
(no serialization); small JSON headers and acks travel over a localhost socket.
This boundary is the seam that a real TEE deployment would replace.

Wire protocol (length-prefixed JSON on the control socket):
  verifier listens, accepts the prover.
  prover  -> verifier : {"type":"hello", shm_name, n_slots, slot_elems, dtype}
  prover  -> verifier : {"type":"step", step, slot, total_len, segments:[...]}
  verifier-> prover    : {"type":"ack", step}            # frees the slot
  prover  -> verifier : {"type":"done"}                  # end of stream

A segment is {"name", "offset", "numel", "shape"} into the flat fp32 slot.
"""

from __future__ import annotations

import json
import math
import socket
import struct
from multiprocessing import shared_memory

import numpy as np
import torch

from capture import StepBuffer


# --------------------------------------------------------------------------- #
# Control socket
# --------------------------------------------------------------------------- #
class ControlChannel:
    def __init__(self, conn: socket.socket) -> None:
        self.conn = conn

    @classmethod
    def listen(cls, host: str, port: int) -> tuple["ControlChannel", int]:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(1)
        actual_port = srv.getsockname()[1]
        return _PendingServer(srv), actual_port

    @classmethod
    def connect(cls, host: str, port: int) -> "ControlChannel":
        conn = socket.create_connection((host, port))
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return cls(conn)

    def send(self, obj: dict) -> None:
        data = json.dumps(obj).encode("utf-8")
        self.conn.sendall(struct.pack("!I", len(data)) + data)

    def recv(self) -> dict:
        raw = self._recvn(4)
        (length,) = struct.unpack("!I", raw)
        return json.loads(self._recvn(length).decode("utf-8"))

    def _recvn(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.conn.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("control channel closed")
            buf.extend(chunk)
        return bytes(buf)

    def close(self) -> None:
        try:
            self.conn.close()
        except OSError:
            pass


class _PendingServer(ControlChannel):
    """Wraps a listening socket; ``accept()`` returns the connected channel."""

    def __init__(self, srv: socket.socket) -> None:
        self.srv = srv
        self.conn = None  # type: ignore[assignment]

    def accept(self) -> ControlChannel:
        conn, _addr = self.srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.srv.close()
        return ControlChannel(conn)


# --------------------------------------------------------------------------- #
# Shared-memory ring (fp32 slots)
# --------------------------------------------------------------------------- #
# transferred-tensor dtype: fp16 halves bandwidth and avoids a per-tensor GPU
# upcast (the verifier casts to fp32 inside the checks anyway).
_NP = {"float16": np.float16, "float32": np.float32}
_TORCH = {"float16": torch.float16, "float32": torch.float32}


class SharedRing:
    def __init__(self, shm: shared_memory.SharedMemory, n_slots: int, slot_elems: int, owner: bool,
                 dtype: str = "float16") -> None:
        self.shm = shm
        self.n_slots = n_slots
        self.slot_elems = slot_elems
        self.owner = owner
        self.dtype_str = dtype
        self.torch_dtype = _TORCH[dtype]
        arr = np.ndarray((n_slots, slot_elems), dtype=_NP[dtype], buffer=shm.buf)
        self.slots = torch.from_numpy(arr)  # (n_slots, slot_elems), shares memory

    @property
    def itemsize(self) -> int:
        return 2 if self.dtype_str == "float16" else 4

    @classmethod
    def create(cls, name: str, n_slots: int, slot_elems: int, dtype: str = "float16") -> "SharedRing":
        size = n_slots * slot_elems * (2 if dtype == "float16" else 4)
        shm = shared_memory.SharedMemory(create=True, size=size, name=name)
        return cls(shm, n_slots, slot_elems, owner=True, dtype=dtype)

    @classmethod
    def attach(cls, name: str, n_slots: int, slot_elems: int, dtype: str = "float16") -> "SharedRing":
        shm = shared_memory.SharedMemory(name=name)
        # The attaching (non-owner) process must not unlink at shutdown. Python's
        # resource_tracker otherwise tries to and prints spurious leak warnings,
        # so unregister it here; only the owner unlinks.
        try:
            from multiprocessing import resource_tracker

            resource_tracker.unregister(shm._name, "shared_memory")  # type: ignore[attr-defined]
        except Exception:
            pass
        return cls(shm, n_slots, slot_elems, owner=False, dtype=dtype)

    def slot(self, i: int) -> torch.Tensor:
        return self.slots[i % self.n_slots]

    def register_cuda(self) -> bool:
        """Page-lock the whole ring with cudaHostRegister so the GPU can DMA
        directly into the shared-memory slots (skipping a pinned->shm memcpy).
        Returns True on success. Only the prover (writer) needs this."""
        self._registered = False
        try:
            ptr = self.slots.data_ptr()
            nbytes = self.n_slots * self.slot_elems * self.itemsize
            err = torch.cuda.cudart().cudaHostRegister(ptr, nbytes, 0)  # 0 = cudaHostRegisterDefault
            ok = (int(err) == 0) if not isinstance(err, tuple) else (int(err[0]) == 0)
            self._registered = ok
            self._reg_ptr = ptr
            return ok
        except Exception:  # pragma: no cover
            return False

    def close(self) -> None:
        if getattr(self, "_registered", False):
            try:
                torch.cuda.cudart().cudaHostUnregister(self._reg_ptr)
            except Exception:
                pass
            self._registered = False
        del self.slots
        self.shm.close()
        if self.owner:
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass


class Cursors:
    """Lock-free producer/consumer cursors in a tiny shared-memory region,
    replacing the per-step socket ack with a memory read/write (no round-trip).

    ``consumed`` is written by the verifier (the highest step it has fully read)
    and read by the prover for slot-reuse backpressure -- so the prover never
    blocks in a socket recv waiting on the verifier. ``produced`` is written by
    the prover (highest step DMA-complete); reserved for also dropping the
    per-step header send. int64, single-writer-per-field, monotonic, naturally
    atomic on x86 (no lock needed)."""

    SIZE = 64  # bytes (cache-line)
    PRODUCED, CONSUMED, DONE = 0, 1, 2

    def __init__(self, shm: shared_memory.SharedMemory, owner: bool) -> None:
        self.shm = shm
        self.owner = owner
        self.arr = np.ndarray((8,), dtype=np.int64, buffer=shm.buf)

    @classmethod
    def create(cls, name: str) -> "Cursors":
        shm = shared_memory.SharedMemory(create=True, size=cls.SIZE, name=name)
        c = cls(shm, owner=True)
        c.arr[:] = -1
        c.arr[cls.DONE] = 0
        return c

    @classmethod
    def attach(cls, name: str) -> "Cursors":
        shm = shared_memory.SharedMemory(name=name)
        try:
            from multiprocessing import resource_tracker

            resource_tracker.unregister(shm._name, "shared_memory")  # type: ignore[attr-defined]
        except Exception:
            pass
        return cls(shm, owner=False)

    @property
    def produced(self) -> int:
        return int(self.arr[self.PRODUCED])

    @produced.setter
    def produced(self, v: int) -> None:
        self.arr[self.PRODUCED] = v

    @property
    def consumed(self) -> int:
        return int(self.arr[self.CONSUMED])

    @consumed.setter
    def consumed(self, v: int) -> None:
        self.arr[self.CONSUMED] = v

    @property
    def done(self) -> int:
        return int(self.arr[self.DONE])

    @done.setter
    def done(self, v: int) -> None:
        self.arr[self.DONE] = v

    def close(self) -> None:
        try:
            del self.arr
            self.shm.close()
            if self.owner:
                self.shm.unlink()
        except (FileNotFoundError, BufferError):
            pass


def estimate_slot_elems(config, batch: int, prompt_len: int, max_new_tokens: int, margin: float = 1.25,
                        transfer_scores: bool = True) -> int:
    """Upper bound on the flattened fp32 element count for a single decode step.

    Worst case is the prefill (q = kv = prompt_len) for the linear and score
    terms; the last decode (q = 1, kv = prompt_len + max_new_tokens) is also
    considered. Margin covers rounding / chat-template growth. When the prover
    does not ship raw QK^T scores (``transfer_scores=False``) the ``attn_scores``
    term -- the dominant prefill cost -- is dropped, shrinking the slot.
    """
    h = config.hidden_size
    inter = config.intermediate_size
    n_layers = config.num_hidden_layers
    n_q = config.num_attention_heads
    n_kv = config.num_key_value_heads
    d = getattr(config, "head_dim", None) or h // n_q
    vocab = config.vocab_size

    def step_elems(q: int, kv: int) -> int:
        per_layer = batch * q * (n_q * d + 2 * n_kv * d + h + 2 * inter)  # projections
        if transfer_scores:
            per_layer += batch * n_q * q * kv  # attn_scores
        per_layer += batch * n_q * q * d  # attn_out
        return n_layers * per_layer + batch * 1 * vocab  # lm_head keeps 1 position

    prefill = step_elems(prompt_len, prompt_len)
    last_decode = step_elems(1, prompt_len + max_new_tokens)
    return int(max(prefill, last_decode) * margin) + 1


# --------------------------------------------------------------------------- #
# Pack / unpack a StepBuffer into a ring slot
# --------------------------------------------------------------------------- #
def pack_step(ring: SharedRing, slot_idx: int, step: int, buf: StepBuffer) -> dict:
    """Flatten ``buf`` (fp32) into ring slot ``slot_idx``; return the step header."""
    slot = ring.slot(slot_idx)
    segments = []
    offset = 0
    for name in buf.order:
        t = buf.fetch(name).detach().to("cpu", ring.torch_dtype).reshape(-1)
        numel = t.numel()
        if offset + numel > ring.slot_elems:
            raise RuntimeError(
                f"step {step} needs {offset + numel} elems > slot_elems {ring.slot_elems}; "
                "increase the slot-size estimate margin"
            )
        slot[offset : offset + numel] = t
        segments.append({"name": name, "offset": offset, "numel": numel, "shape": list(buf.fetch(name).shape)})
        offset += numel
    return {"type": "step", "step": step, "slot": slot_idx % ring.n_slots, "total_len": offset, "segments": segments}


def plan_step(buf: StepBuffer, slot_idx: int, n_slots: int, step: int) -> tuple[dict, int]:
    """Compute the step header (segment offsets/shapes) from tensor shapes only,
    without moving any data. Returns (header, total_len). The caller is
    responsible for writing ``total_len`` fp32 elements into the slot in
    ``buf.order``.
    """
    segments = []
    offset = 0
    for name in buf.order:
        shape = list(buf.fetch(name).shape)
        numel = int(buf.fetch(name).numel())
        segments.append({"name": name, "offset": offset, "numel": numel, "shape": shape})
        offset += numel
    header = {"type": "step", "step": step, "slot": slot_idx % n_slots, "total_len": offset, "segments": segments}
    return header, offset


def plan_grouped(buf: StepBuffer, slot_idx: int, n_slots: int, step: int):
    """Group the step's tensors by projection (q_proj, k_proj, ... contiguous
    across layers) so the verifier reads ~one slice per group instead of one
    view per (layer, proj). Returns (header, ordered_names, total_len); the
    caller writes the slot in ``ordered_names`` order. Within a group all layers
    share a shape, so a group is a contiguous ``(count, *shape)`` block.
    """
    tmp: dict[str, list[tuple[int, str]]] = {}
    for name in buf.order:
        if "." in name:
            li, proj = name.split(".", 1)
            tmp.setdefault(proj, []).append((int(li), name))
        else:
            tmp.setdefault(name, []).append((-1, name))
    groups, ordered_names, offset = [], [], 0
    for proj in sorted(tmp):
        items = sorted(tmp[proj])
        shape = list(buf.fetch(items[0][1]).shape)
        count = len(items)
        groups.append({"name": proj, "offset": offset, "count": count, "shape": shape})
        ordered_names.extend(nm for _, nm in items)
        offset += count * math.prod(shape)
    header = {"type": "step", "step": step, "slot": slot_idx % n_slots, "total_len": offset, "groups": groups}
    return header, ordered_names, offset


def read_grouped(ring: SharedRing, slot_idx: int, groups: list[dict], copy: bool = False) -> dict[str, torch.Tensor]:
    """Read each projection group as one ``(count, *shape)`` view into the slot.

    ``copy=True``: first bulk-clone the used span of the slot into regular (cached)
    memory. The ring is ``cudaHostRegister``'d (pinned/write-combining) so the GPU
    can DMA into it; CPU reads of WC memory are slow and get much worse when many
    threads do *scattered* reads under contention. One *sequential* bulk clone here
    (then all downstream reads hit cached memory) is far more robust on a busy node.
    """
    slot = ring.slot(slot_idx)
    if copy:
        used = max((g["offset"] + g["count"] * math.prod(g["shape"]) for g in groups), default=0)
        slot = slot[:used].clone()
    out = {}
    for g in groups:
        n = g["count"] * math.prod(g["shape"])
        out[g["name"]] = slot[g["offset"] : g["offset"] + n].view(g["count"], *g["shape"])
    return out


def unpack_step(ring: SharedRing, header: dict) -> StepBuffer:
    """View the ring slot named by ``header`` back into a StepBuffer.

    Returns zero-copy views into the shared-memory slot (no clone): the slot is
    protected by backpressure -- the prover does not reuse it until the verifier
    acks the step, which happens only after all checks for the step are done.
    """
    slot = ring.slot(header["slot"])
    buf = StepBuffer()
    for seg in header["segments"]:
        flat = slot[seg["offset"] : seg["offset"] + seg["numel"]]
        buf.record(seg["name"], flat.view(*seg["shape"]))
    return buf


# --------------------------------------------------------------------------- #
# Round-trip self-test (two processes, random tensors)
# --------------------------------------------------------------------------- #
def _verifier_proc(host, n_slots, slot_elems, n_steps, port_q, result_q):
    pending, port = ControlChannel.listen(host, 0)
    port_q.put(port)
    ctrl = pending.accept()
    hello = ctrl.recv()
    ring = SharedRing.attach(hello["shm_name"], hello["n_slots"], hello["slot_elems"], dtype="float32")
    checksums = []
    while True:
        msg = ctrl.recv()
        if msg["type"] == "done":
            break
        buf = unpack_step(ring, msg)
        checksums.append({name: float(buf.fetch(name).sum()) for name in buf.order})
        ctrl.send({"type": "ack", "step": msg["step"]})
    ring.close()
    ctrl.close()
    result_q.put(checksums)


def _selftest() -> int:
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    host = "127.0.0.1"
    n_slots, slot_elems, n_steps = 3, 4096, 6
    shm_name = f"flossy_test_{torch.randint(0, 1_000_000, (1,)).item()}"

    port_q: mp.Queue = mp.Queue()
    result_q: mp.Queue = mp.Queue()
    proc = mp.Process(target=_verifier_proc, args=(host, n_slots, slot_elems, n_steps, port_q, result_q))
    proc.start()

    port = port_q.get(timeout=10)
    ctrl = ControlChannel.connect(host, port)
    ring = SharedRing.create(shm_name, n_slots, slot_elems, dtype="float32")
    ctrl.send({"type": "hello", "shm_name": shm_name, "n_slots": n_slots, "slot_elems": slot_elems, "dtype": "float32"})

    torch.manual_seed(0)
    expected = []
    acked = -1
    for step in range(n_steps):
        buf = StepBuffer()
        buf.record("a", torch.randn(2, 8))
        buf.record("b", torch.randn(3, 5))
        expected.append({name: float(buf.fetch(name).sum()) for name in buf.order})
        # backpressure: do not overwrite an unacked slot
        while step - acked >= n_slots:
            m = ctrl.recv()
            if m["type"] == "ack":
                acked = max(acked, m["step"])
        header = pack_step(ring, step, step, buf)
        ctrl.send(header)
    ctrl.send({"type": "done"})

    got = result_q.get(timeout=10)
    proc.join(timeout=10)
    ring.close()
    ctrl.close()

    ok = len(got) == n_steps and all(
        all(abs(got[i][k] - expected[i][k]) < 1e-3 for k in expected[i]) for i in range(n_steps)
    )
    print(f"round-trip {n_steps} steps: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print("usage: python transport.py --selftest")
