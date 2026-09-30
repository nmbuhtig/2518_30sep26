#!/usr/bin/env python3

import argparse
import json
import math

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--prefix", default="ensemble_summary")
    args = ap.parse_args()

    runs = [json.load(open(p)) for p in args.files]
    par0 = runs[0]["parameters"]
    for r in runs[1:]:
        for key in ("phi", "nside", "eta_col", "input_p", "output_tau", "ntraj"):
            if r["parameters"][key] != par0[key]:
                print(f"WARNING: '{key}' differs between ensembles")
    seeds = [r["parameters"]["seed"] for r in runs]
    if len(set(seeds)) != len(seeds):
        print("WARNING: repeated seeds; ensembles are not independent")

    gv = runs[0]["grid_validation"]
    loss_rate = 1.0 - (gv["initial_collected_cross_section_um2"]
                       / gv["initial_full_cross_section_um2"])
    n = len(runs)
    taus = [o["tau"] for o in runs[0]["outputs"]]
    lvals = sorted(int(l) for l in runs[0]["outputs"][0]["oam_conditional"])

    rows = []
    for i, tau in enumerate(taus):
        T = np.array([r["outputs"][i]["transmittance"] for r in runs])
        sf = np.array([r["outputs"][i]["spin_flip_conditional"] for r in runs])
        oam = np.array([[r["outputs"][i]["oam_conditional"][str(l)]
                         for l in lvals] for r in runs])
        sd = T.std(ddof=1) if n > 1 else float("nan")
        se = sd / math.sqrt(n)
        tref = math.exp(-loss_rate * tau)
        rows.append({
            "tau": tau,
            "T_mean": T.mean(), "T_sd": sd, "T_se": se,
            "T_ref": tref,
            "ref_minus_mean_in_se": (tref - T.mean()) / se if se > 0 else None,
            "Psf_mean": sf.mean(), "Psf_sd": sf.std(ddof=1) if n > 1 else None,
            "P_l_mean": dict(zip(lvals, oam.mean(0).tolist())),
            "P_l_sd": dict(zip(lvals, (oam.std(0, ddof=1) if n > 1
                                       else np.zeros(len(lvals))).tolist())),
            "ESS_min": min(r["outputs"][i]["effective_sample_size"] for r in runs),
        })

    phys = runs[0]["physics"]
    print(f"phi={par0['phi']}, nside={par0['nside']}, ensembles={n}, "
          f"seeds={seeds}")
    print(f"ell_s={phys['ell_s_um']:.4f} um, ell_s/zR={phys['ell_s_over_zR']:.4f}, "
          f"reference loss rate={loss_rate:.5f}")
    print(f"{'tau':>5} {'T_mean':>9} {'T_sd':>9} {'T_ref':>9} {'dev/SE':>7} "
          f"{'P0':>8} {'P+-1':>9} {'P+-2':>9} {'Psf':>9} {'ESSmin':>7}")
    for r in rows:
        p = r["P_l_mean"]
        dev = r["ref_minus_mean_in_se"]
        print(f"{r['tau']:5.2f} {r['T_mean']:9.6f} {r['T_sd']:9.6f} "
              f"{r['T_ref']:9.6f} {dev if dev is not None else float('nan'):7.2f} "
              f"{p[0]:8.5f} {p[1] + p[-1]:9.2e} {p[2] + p[-2]:9.2e} "
              f"{r['Psf_mean']:9.2e} {r['ESS_min']:7.1f}")

    with open(f"{args.prefix}.json", "w") as fh:
        json.dump({"files": args.files, "seeds": seeds,
                   "parameters": par0, "physics": phys,
                   "reference_loss_rate": loss_rate, "rows": rows}, fh, indent=1)
    print(f"Wrote {args.prefix}.json")


if __name__ == "__main__":
    main()
