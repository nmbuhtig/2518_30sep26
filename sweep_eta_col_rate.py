#!/usr/bin/env python3

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


def load_module(filename, module_name):
    here = Path(__file__).resolve().parent
    candidate = here / filename
    if not candidate.exists():
        raise FileNotFoundError(f"{filename} must be in the same folder.")
    spec = importlib.util.spec_from_file_location(module_name, candidate)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_module(
    "sphere_glmt_fixed_collection_v3.py", "sphere_glmt_fixed_collection_v3_sweep"
)
flux = load_module("sphere_glmt_stage3_flux.py", "sphere_glmt_stage3_flux_sweep")
stage3b = load_module("sphere_glmt_stage3b_oam.py", "sphere_glmt_stage3b_oam_sweep")


def parse_list(text, cast=float):
    return [cast(v) for v in text.split(",") if v.strip()]


def khat_from_angles(theta, phi):
    s = np.sin(theta)
    return np.column_stack((s * np.cos(phi), s * np.sin(phi), np.cos(theta)))


def rcol_at_inputs(q_in, eta_col, k, mie_a, mie_b, ntheta, nphi):
    theta_col = math.asin(eta_col)
    xg, wg = np.polynomial.legendre.leggauss(ntheta)
    theta = 0.5 * theta_col * (xg + 1.0)
    wtheta = 0.5 * theta_col * wg
    phi = 2.0 * math.pi * np.arange(nphi) / nphi
    wphi = 2.0 * math.pi / nphi

    th2, ph2 = np.meshgrid(theta, phi, indexing="ij")
    khat_out = khat_from_angles(th2.ravel(), ph2.ravel())
    # d^2q_o / c_o = k^2 sin(theta) dtheta dphi
    w_out = (k**2) * (np.sin(th2) * wtheta[:, None] * wphi).ravel()

    e_out = {s: stage3b.local_circular(khat_out, s) for s in (+1, -1)}

    theta_in = np.arcsin(np.clip(q_in / k, 0.0, 1.0))
    khat_in = khat_from_angles(theta_in, np.zeros_like(theta_in))
    c_in = np.cos(theta_in)
    e_in = stage3b.local_circular(khat_in, +1)

    r_keep = np.zeros(len(q_in))
    r_flip = np.zeros(len(q_in))
    for j in range(len(q_in)):
        scat = flux.scattering_vectors(
            khat_out, khat_in[j : j + 1], e_in[j : j + 1], mie_a, mie_b
        )[:, 0, :]
        for s, target in ((+1, r_keep), (-1, r_flip)):
            amp = np.einsum("ok,ok->o", e_out[s].conj(), scat)
            target[j] = np.sum(w_out * np.abs(amp) ** 2) / (k**4 * c_in[j])
    return r_keep, r_flip


def radial_weights(q, w0, p, k):
    U = base.lg_angular_spectrum(q, np.zeros_like(q), w0, p, 0)
    c = np.sqrt(np.maximum(0.0, 1.0 - (q / k) ** 2))
    return 2.0 * math.pi * q * c * np.abs(U) ** 2


def execute(args):
    t0 = time.time()
    k = 2.0 * math.pi * args.n_medium / args.wavelength_um
    m_rel = complex(args.n_sphere_real, args.n_sphere_imag) / args.n_medium
    x = k * args.radius_um
    mie_a, mie_b, qsca, nmax = base.mie_coefficients(m_rel, x)
    sigma_s = qsca * math.pi * args.radius_um**2

    eta_list = parse_list(args.eta_col)
    p_list = parse_list(args.input_p, int)

    # Input radial quadrature (complete beam, not truncated at eta_col).
    qin_max = args.qin_max_over_k * k
    xg, wg = np.polynomial.legendre.leggauss(args.nq_in)
    q_in = 0.5 * qin_max * (xg + 1.0)
    w_in = 0.5 * qin_max * wg
    c_in = np.sqrt(1.0 - (q_in / k) ** 2)

    def make_dens(q, w):
        c = np.sqrt(1.0 - (q / k) ** 2)
        out = {}
        for p in p_list:
            d = w * radial_weights(q, args.w0_um, p, k)
            norm = float(np.sum(d))
            d = d / norm
            out[p] = (d, norm, sigma_s * float(np.sum(d / c)))
        return out

    dens = make_dens(q_in, w_in)

    def make_dens_fraction(p, eta):
        qc = min(qin_max, eta * k)
        qd = 0.5 * qc * (xg + 1.0)
        wd = 0.5 * qc * wg
        inside = float(np.sum(wd * radial_weights(qd, args.w0_um, p, k)))
        return inside / dens[p][1]

    print(f"x = {x:.7f}, n_max = {nmax}, sigma_s = {sigma_s:.8f} um^2")
    for p in p_list:
        print(f"p={p}: flux norm on input quadrature = {dens[p][1]:.12f}, "
              f"sigma_full = {dens[p][2]:.8f} um^2")

    rows, rcol_tables = [], {}
    for eta in eta_list:
        r_keep, r_flip = rcol_at_inputs(
            q_in, eta, k, mie_a, mie_b, args.ntheta, args.nphi
        )
        r_tot = r_keep + r_flip
        if args.input_in_disk:
            # Inputs restricted to the collection disk and renormalized.
            qd = 0.5 * min(qin_max, eta * k) * (xg + 1.0)
            wd = 0.5 * min(qin_max, eta * k) * wg
            rk_d, rf_d = rcol_at_inputs(
                qd, eta, k, mie_a, mie_b, args.ntheta, args.nphi
            )
            r_eval, dens_eval = rk_d + rf_d, make_dens(qd, wd)
        else:
            r_eval, dens_eval = r_tot, dens
        row = {"eta_col": eta}
        for p in p_list:
            d, _, full = dens_eval[p]
            col = float(np.sum(d * r_eval))
            outside = 1.0 - make_dens_fraction(p, eta)
            row[f"sigma_col_p{p}_um2"] = col
            row[f"sigma_full_p{p}_um2"] = full
            row[f"input_flux_outside_disk_p{p}"] = outside
        p0 = p_list[0]
        for p in p_list[1:]:
            dcol = row[f"sigma_col_p{p0}_um2"] - row[f"sigma_col_p{p}_um2"]
            dfull = row[f"sigma_full_p{p}_um2"] - row[f"sigma_full_p{p0}_um2"]
            rel_col = dcol / row[f"sigma_col_p{p0}_um2"]
            rel_full = dfull / row[f"sigma_full_p{p0}_um2"]
            row[f"dcol_p{p0}_p{p}_um2"] = dcol
            row[f"dfull_p{p}_p{p0}_um2"] = dfull
            row[f"rel_dcol_p{p0}_p{p}"] = rel_col
            row[f"rel_dfull_p{p}_p{p0}"] = rel_full
            row[f"abs_ratio_p{p}"] = dcol / dfull
            row[f"rel_ratio_p{p}"] = rel_col / rel_full
        rows.append(row)
        rcol_tables[eta] = (r_keep / sigma_s, r_flip / sigma_s)
        msg = ", ".join(
            f"sigma_col(p={p})={row[f'sigma_col_p{p}_um2']:.6f}" for p in p_list
        )
        print(f"eta_col={eta:.3f}: {msg}  [{time.time() - t0:.0f} s]")

    prefix = args.prefix
    with open(f"{prefix}_rates.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(f"{prefix}_rcol.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        header = ["q_um_inv", "eta"]
        for eta in eta_list:
            header += [f"rkeep_over_sigma_s_eta{eta:g}", f"rflip_over_sigma_s_eta{eta:g}"]
        w.writerow(header)
        for j in range(len(q_in)):
            line = [q_in[j], q_in[j] / k]
            for eta in eta_list:
                line += [rcol_tables[eta][0][j], rcol_tables[eta][1][j]]
            w.writerow(line)

    result = {
        "script": "sweep_eta_col_rate.py",
        "parameters": vars(args),
        "physics": {
            "k_um_inv": k,
            "size_parameter": x,
            "mie_nmax": nmax,
            "Qsca": qsca,
            "sigma_s_um2": sigma_s,
        },
        "input_quadrature": {
            "q_um_inv": q_in.tolist(),
            "flux_norm": {str(p): dens[p][1] for p in p_list},
        },
        "rates": rows,
        "rcol_over_sigma_s": {
            f"{eta:g}": {"keep": rcol_tables[eta][0].tolist(),
                         "flip": rcol_tables[eta][1].tolist()}
            for eta in eta_list
        },
        "runtime_s": time.time() - t0,
    }
    with open(f"{prefix}_RESULTS.json", "w") as fh:
        json.dump(result, fh, indent=1)
    print(f"Wrote {prefix}_RESULTS.json, {prefix}_rates.csv, {prefix}_rcol.csv")


def parse_args():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--eta-col", default="0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--input-p", default="0,1,2",
                    help="Radial orders; the first is the reference.")
    ap.add_argument("--w0-um", type=float, default=5.0)
    ap.add_argument("--wavelength-um", type=float, default=0.633)
    ap.add_argument("--radius-um", type=float, default=2.0)
    ap.add_argument("--n-medium", type=float, default=1.0)
    ap.add_argument("--n-sphere-real", type=float, default=1.33)
    ap.add_argument("--n-sphere-imag", type=float, default=0.0)
    ap.add_argument("--nq-in", type=int, default=64)
    ap.add_argument("--qin-max-over-k", type=float, default=0.25)
    ap.add_argument("--ntheta", type=int, default=160)
    ap.add_argument("--nphi", type=int, default=256)
    ap.add_argument("--input-in-disk", action="store_true",
                    help="Restrict inputs to |q|<=eta_col k and renormalize "
                         "(state on the retained space); default uses the "
                         "complete LG beam.")
    ap.add_argument("--prefix", default="eta_sweep")
    return ap.parse_args()


if __name__ == "__main__":
    execute(parse_args())
