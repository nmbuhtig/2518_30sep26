#!/usr/bin/env python3

import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
TAUS = "0,0.25,0.5,1,1.5,2"
GRIDSEED = 20260910
COMMON = [
    "--w0-um", "5", "--eta-col", "0.10",
    "--input-p", "0", "--input-l", "0", "--input-sigma", "1",
    "--ntraj", "1000", "--resample-step", "0.25", "--output-tau", TAUS,
    "--lmin-measure", "-60", "--lmax-measure", "60",
    "--nqr-measure", "100", "--nphi-measure", "256",
    "--measurement-order", "3",
]
REQUIRED = [
    "sphere_glmt_fixed_collection_v3.py", "sphere_glmt_stage3_flux.py",
    "sphere_glmt_stage3b_oam.py", "sphere_glmt_stage4b_trajectories_v2.py",
    "sweep_eta_col_rate.py", "compare_grids.py", "summarize_ensembles.py",
    "witness_branching.py",
]


def run(args):
    print("\n>>> " + " ".join(args), flush=True)
    t0 = time.time()
    subprocess.run([PY] + args, check=True)
    print(f"<<< finished in {time.time() - t0:.0f} s", flush=True)


def trajectories(phi_tag, phi, nside, seed, ntraj=1000):
    tag = "" if ntraj == 1000 else f"_t{ntraj}"
    prefix = f"s5b_phi{phi_tag}_n{nside}_seed{seed}{tag}"
    args = list(COMMON)
    args[args.index("--ntraj") + 1] = str(ntraj)
    run(["sphere_glmt_stage4b_trajectories_v2.py", *args,
         "--phi", phi, "--nside", str(nside), "--seed", str(seed),
         "--prefix", prefix])
    return f"{prefix}_FULL_RESULTS.json"


def grid_study(phi_tag, phi):
    files = {n: trajectories(phi_tag, phi, n, GRIDSEED) for n in (41, 51, 71)}
    run(["compare_grids.py", files[41], files[51],
         "--prefix", f"s5b_phi{phi_tag}_grid_41_51"])
    run(["compare_grids.py", files[51], files[71],
         "--prefix", f"s5b_phi{phi_tag}_grid_51_71"])
    return files


def job1():
    grid_study("0042", "0.0042")


def job2():
    files = grid_study("0084", "0.0084")
    ens = [files[51]] + [trajectories("0084", "0.0084", 51, s)
                         for s in (20260911, 20260912)]
    run(["summarize_ensembles.py", *ens, "--prefix", "s5b_phi0084_n51_ensembles"])


def job3():
    run(["sweep_eta_col_rate.py", "--prefix", "s5b_eta_sweep"])
    run(["sweep_eta_col_rate.py", "--input-in-disk",
         "--prefix", "s5b_eta_sweep_indisk"])
    run(["sweep_eta_col_rate.py", "--eta-col", "0.1,0.5,0.9",
         "--ntheta", "320", "--nphi", "512", "--nq-in", "96",
         "--prefix", "s5b_eta_sweep_check"])


def job4():
    files = {n: trajectories("0042", "0.0042", n, GRIDSEED, 4000)
             for n in (41, 51, 71)}
    run(["compare_grids.py", files[41], files[51],
         "--prefix", "s5b_phi0042_t4000_grid_41_51"])
    run(["compare_grids.py", files[51], files[71],
         "--prefix", "s5b_phi0042_t4000_grid_51_71"])
    # Independent ensemble on the same grid: sampling-noise reference.
    noise = trajectories("0042", "0.0042", 51, 20260911, 4000)
    run(["compare_grids.py", files[51], noise,
         "--prefix", "s5b_phi0042_t4000_noise_n51_seeds10_11"])


def job5():
    for n in (41, 51, 71):
        run(["witness_branching.py", "--nside", str(n),
             "--prefix", f"s5b_witness_n{n}"])


def main():
    jobs = {"job1": [job1], "job2": [job2], "job3": [job3],
            "job4": [job4], "job5": [job5],
            "all": [job3, job1, job2]}
    if len(sys.argv) != 2 or sys.argv[1] not in jobs:
        print(__doc__)
        sys.exit(1)
    missing = [f for f in REQUIRED if not Path(f).exists()]
    if missing:
        print("Missing in the current folder:\n  " + "\n  ".join(missing))
        sys.exit(1)
    print(f"Python: {PY}")
    for job in jobs[sys.argv[1]]:
        job()


if __name__ == "__main__":
    main()
