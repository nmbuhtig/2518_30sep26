#!/usr/bin/env python3

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


def load_module(filename, module_name):
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(module_name, here / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


s4 = load_module("sphere_glmt_stage4b_trajectories_v2.py", "stage4b_witness")
base = s4.base


def branching(state, jumps, grid, meas, lmin, lmax, order, rel_cut):
    qprob = s4.retained_q_probabilities(state, jumps)
    total = float(np.sum(qprob))
    qids = np.nonzero(qprob > rel_cut * qprob.max())[0]
    acc = None
    measured, used = 0.0, 0.0
    for qid in qids:
        new, _ = s4.apply_retained_jump(state, jumps, int(qid))
        res = s4.state_to_oam(state=new, grid=grid, measurement=meas,
                              lmin=lmin, lmax=lmax,
                              interpolation_order=order)
        w = float(qprob[qid])
        acc = w * res["power_spin"] if acc is None else acc + w * res["power_spin"]
        measured += w * res["measured_total"]
        used += w
    return {
        "rate_tau_units": total,
        "neglected_rate_fraction": 1.0 - used / total,
        "capture": measured / used,
        "l_values": res["l_values"].tolist(),
        "P_plus": (acc[0] / used).tolist(),
        "P_minus": (acc[1] / used).tolist(),
    }


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--nside", type=int, default=51)
    ap.add_argument("--input-p", default="0,1,2")
    ap.add_argument("--w0-um", type=float, default=5.0)
    ap.add_argument("--eta-col", type=float, default=0.10)
    ap.add_argument("--wavelength-um", type=float, default=0.633)
    ap.add_argument("--radius-um", type=float, default=2.0)
    ap.add_argument("--n-medium", type=float, default=1.0)
    ap.add_argument("--n-sphere-real", type=float, default=1.33)
    ap.add_argument("--n-sphere-imag", type=float, default=0.0)
    ap.add_argument("--lmin", type=int, default=-20)
    ap.add_argument("--lmax", type=int, default=20)
    ap.add_argument("--nqr-measure", type=int, default=100)
    ap.add_argument("--nphi-measure", type=int, default=256)
    ap.add_argument("--measurement-order", type=int, default=3)
    ap.add_argument("--rel-cut", type=float, default=1e-14,
                    help="Skip Q channels below this fraction of the largest.")
    ap.add_argument("--prefix", default="witness")
    args = ap.parse_args()

    t0 = time.time()
    k = 2.0 * math.pi * args.n_medium / args.wavelength_um
    m_rel = complex(args.n_sphere_real, args.n_sphere_imag) / args.n_medium
    mie_a, mie_b, qsca, _ = base.mie_coefficients(m_rel, k * args.radius_um)
    sigma_s = qsca * math.pi * args.radius_um**2

    grid = s4.make_cartesian_disk(args.eta_col * k, args.nside, k)
    jumps = s4.precompute_pair_spin_matrices(
        grid=grid, mie_a=mie_a, mie_b=mie_b, k=k, sigma_s=sigma_s)
    meas = s4.make_measurement_grid(args.eta_col * k, args.nqr_measure,
                                    args.nphi_measure)
    full_rate_diag = 1.0 / grid["cos_theta"]
    print(f"nside={args.nside}, points={grid['npoints']}, "
          f"setup {time.time() - t0:.0f} s")

    out = {"parameters": vars(args), "sigma_s_um2": sigma_s, "inputs": {}}
    for p in [int(v) for v in args.input_p.split(",")]:
        state, raw_norm = s4.initial_lg_state(
            grid=grid, w0_um=args.w0_um, p=p, l=0, sigma=+1)
        b = branching(state, jumps, grid, meas, args.lmin, args.lmax,
                      args.measurement_order, args.rel_cut)
        pop = np.sum(np.abs(state) ** 2, axis=1)
        b["sigma_col_um2"] = sigma_s * b["rate_tau_units"]
        b["sigma_full_um2"] = sigma_s * float(np.sum(pop * full_rate_diag))
        b["raw_flux_norm_in_disk"] = raw_norm
        out["inputs"][str(p)] = b
        print(f"p={p}: sigma_col={b['sigma_col_um2']:.6f} um^2, "
              f"sigma_full={b['sigma_full_um2']:.6f} um^2, "
              f"capture={b['capture']:.8f}, "
              f"P(0,+)={b['P_plus'][b['l_values'].index(0)]:.6f}, "
              f"P_sf={sum(b['P_minus']):.6e}  [{time.time() - t0:.0f} s]")

    def dist(b):
        return np.concatenate([b["P_plus"], b["P_minus"]])

    keys = list(out["inputs"])
    out["tv_distance"] = {}
    for kk in keys[1:]:
        a, c = dist(out["inputs"][keys[0]]), dist(out["inputs"][kk])
        a, c = a / a.sum(), c / c.sum()
        tv = 0.5 * float(np.sum(np.abs(a - c)))
        out["tv_distance"][f"p{kk}_vs_p{keys[0]}"] = tv
        print(f"D_TV(p={kk}, p={keys[0]}) = {tv:.6f}")

    out["runtime_s"] = time.time() - t0
    with open(f"{args.prefix}_RESULTS.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"Wrote {args.prefix}_RESULTS.json")


if __name__ == "__main__":
    main()
