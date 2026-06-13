"""Run the prover+verifier pipeline as two mp.spawn children of ONE python
process (the only multi-process pattern this sandbox tolerates). Same network
namespace -> the localhost socket connects.

  python run_e2e_spawn.py <model> [-n N] [--vthreads T] [extra prover flags...]

-n sets the output token count; --vthreads sets the verifier CPU thread count;
all other extra flags (e.g. --batch) go to the prover.
"""
import multiprocessing as mp
import os
import sys
import time

PORT = "29823"
SHM = "e2espawn"


def _run(modname, argv):
    sys.argv = [modname + ".py"] + argv
    mod = __import__(modname)
    raise SystemExit(mod.main())


def main():
    model = sys.argv[1]
    rest = sys.argv[2:]
    # pull verifier knobs out of the extra args; the remainder go to the prover
    workers, vthreads, ntok, sval, extra_prover = "1", "16", "64", "10", []
    it = iter(rest)
    for a in it:
        if a == "--workers":
            workers = next(it)
        elif a == "--vthreads":
            vthreads = next(it)
        elif a in ("-n", "--max-new-tokens"):
            ntok = next(it)
        elif a == "--s":
            sval = next(it)
        else:
            extra_prover.append(a)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    common = ["--model", model, "--prompt", "essay about computing history", "--s", sval,
              "-n", ntok, "--min-new-tokens", ntok, "--host", "127.0.0.1", "--port", PORT, "--shm-name", SHM]
    vargs = ["--workers", workers, "--threads", vthreads]
    ctx = mp.get_context("spawn")
    v = ctx.Process(target=_run, args=("verifier", common + vargs))
    v.start()
    time.sleep(25)  # let the verifier bind + load its CPU model(s)
    p = ctx.Process(target=_run, args=("prover", common + ["--threads", "4"] + extra_prover))
    p.start()
    p.join()
    v.join()


if __name__ == "__main__":
    main()
