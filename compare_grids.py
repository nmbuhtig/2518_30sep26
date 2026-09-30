#!/usr/bin/env python3

import argparse
import csv
import json

MATCH_KEYS = ("phi", "eta_col", "seed", "input_p", "input_l", "input_sigma",
              "w0_um", "ntraj", "output_tau", "resample_step")


def load(path):
    with open(path) as fh:
        return json.load(fh)


def tv_distance(a, b):
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("coarse")
    ap.add_argument("fine")
    ap.add_argument("--prefix", default="grid_comparison")
    ap.add_argument("--cmin", type=float, default=0.99)
    ap.add_argument("--tvmax", type=float, default=0.01)
    ap.add_argument("--dtmax", type=float, default=0.01)
    ap.add_argument("--dsfmax", type=float, default=1e-4)
    args = ap.parse_args()

    c, f = load(args.coarse), load(args.fine)
    for key in MATCH_KEYS:
        if c["parameters"].get(key) != f["parameters"].get(key):
            print(f"WARNING: parameter '{key}' differs: "
                  f"{c['parameters'].get(key)} vs {f['parameters'].get(key)}")

    nc, nf = c["parameters"]["nside"], f["parameters"]["nside"]
    fo = {round(o["tau"], 10): o for o in f["outputs"]}
    rows = []
    for oc in c["outputs"]:
        tau = round(oc["tau"], 10)
        of = fo.get(tau)
        if of is None:
            continue
        tc, tf = oc["transmittance"], of["transmittance"]
        row = {
            "tau": tau,
            f"T_{nc}": tc,
            f"T_{nf}": tf,
            "dT_over_T": abs(tf - tc) / tf,
            f"C_{nc}": oc["measurement_captured_weight"],
            f"C_{nf}": of["measurement_captured_weight"],
            "D_TV": tv_distance(oc["oam_conditional"], of["oam_conditional"]),
            "dP_sf": abs(oc["spin_flip_conditional"] - of["spin_flip_conditional"]),
            f"ESS_{nc}": oc["effective_sample_size"],
            f"ESS_{nf}": of["effective_sample_size"],
        }
        ok = (min(row[f"C_{nc}"], row[f"C_{nf}"]) >= args.cmin
              and row["D_TV"] <= args.tvmax
              and row["dT_over_T"] <= args.dtmax
              and row["dP_sf"] <= args.dsfmax)
        row["status"] = "pass" if ok else "fail"
        rows.append(row)

    print(f"phi={c['parameters']['phi']}, seed={c['parameters']['seed']}, "
          f"grids {nc} vs {nf}")
    print(f"{'tau':>5} {'T_c':>9} {'T_f':>9} {'dT/T':>9} {'C_c':>9} "
          f"{'C_f':>9} {'D_TV':>8} {'dPsf':>9} status")
    for r in rows:
        print(f"{r['tau']:5.2f} {r[f'T_{nc}']:9.6f} {r[f'T_{nf}']:9.6f} "
              f"{100 * r['dT_over_T']:8.4f}% {r[f'C_{nc}']:9.6f} "
              f"{r[f'C_{nf}']:9.6f} {r['D_TV']:8.5f} {r['dP_sf']:9.2e} "
              f"{r['status']}")

    with open(f"{args.prefix}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(f"{args.prefix}.json", "w") as fh:
        json.dump({"coarse": args.coarse, "fine": args.fine,
                   "criteria": {"cmin": args.cmin, "tvmax": args.tvmax,
                                "dtmax": args.dtmax, "dsfmax": args.dsfmax},
                   "rows": rows}, fh, indent=1)
    print(f"Wrote {args.prefix}.csv and {args.prefix}.json")


if __name__ == "__main__":
    main()
