#!/usr/bin/env python3


from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np


def load_module(filename, module_name):
    here = Path(__file__).resolve().parent
    candidate = here / filename

    if not candidate.exists():
        raise FileNotFoundError(
            f"{filename} must be in the same folder."
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        candidate,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load {filename}."
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


base = load_module(
    "sphere_glmt_fixed_collection_v3.py",
    "sphere_glmt_fixed_collection_v3_stage3b",
)

flux = load_module(
    "sphere_glmt_stage3_flux.py",
    "sphere_glmt_stage3_flux_stage3b",
)


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def write_json(data, path):
    path = Path(path)
    path.write_text(
        json.dumps(
            data,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return path


def args_dict(args):
    return {
        key: value
        for key, value in vars(args).items()
        if isinstance(
            value,
            (str, int, float, bool, type(None)),
        )
    }


def trapezoid(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def local_circular(khat, sigma):
    if sigma not in (-1, +1):
        raise ValueError("sigma must be +/-1.")

    khat = np.asarray(
        khat,
        dtype=float,
    )

    x = khat[:, 0]
    y = khat[:, 1]
    z = khat[:, 2]

    phi = np.arctan2(y, x)

    sin_theta = np.sqrt(
        np.maximum(
            0.0,
            x**2 + y**2,
        )
    )

    cos_theta = z

    e_theta = np.column_stack(
        (
            cos_theta * np.cos(phi),
            cos_theta * np.sin(phi),
            -sin_theta,
        )
    )

    e_phi = np.column_stack(
        (
            -np.sin(phi),
            np.cos(phi),
            np.zeros_like(phi),
        )
    )

    phase = np.exp(
        1j * sigma * phi
    )

    result = (
        phase[:, None]
        * (
            e_theta
            + 1j * sigma * e_phi
        )
        / math.sqrt(2.0)
    )

    return result


def polarization_diagnostics(khat):
    ep = local_circular(
        khat,
        +1,
    )

    em = local_circular(
        khat,
        -1,
    )

    gram_pp = np.sum(
        ep.conj() * ep,
        axis=1,
    )

    gram_mm = np.sum(
        em.conj() * em,
        axis=1,
    )

    gram_pm = np.sum(
        ep.conj() * em,
        axis=1,
    )

    trans_p = np.sum(
        ep.conj() * khat,
        axis=1,
    )

    trans_m = np.sum(
        em.conj() * khat,
        axis=1,
    )

    pgp = base.projected_global_circular(
        khat,
        +1,
    )

    pgm = base.projected_global_circular(
        khat,
        -1,
    )

    return {
        "max_norm_error": float(
            max(
                np.max(
                    np.abs(
                        gram_pp - 1.0
                    )
                ),
                np.max(
                    np.abs(
                        gram_mm - 1.0
                    )
                ),
            )
        ),
        "max_cross_overlap": float(
            np.max(
                np.abs(gram_pm)
            )
        ),
        "max_transversality_error": float(
            max(
                np.max(
                    np.abs(trans_p)
                ),
                np.max(
                    np.abs(trans_m)
                ),
            )
        ),
        "max_change_from_projected_global": float(
            max(
                np.max(
                    np.linalg.norm(
                        ep - pgp,
                        axis=1,
                    )
                ),
                np.max(
                    np.linalg.norm(
                        em - pgm,
                        axis=1,
                    )
                ),
            )
        ),
    }


def oam_indices(nphi):
    raw = np.fft.fftfreq(
        nphi,
        d=1.0 / nphi,
    )

    return raw.astype(int)


def compute_b_resolved(
    *,
    input_grid,
    output_grid,
    input_U,
    input_sigma,
    b_um,
    mie_a,
    mie_b,
    k,
    lmin_out,
    lmax_out,
    block_size,
):
    khat_in = input_grid.khat

    e_in = local_circular(
        khat_in,
        input_sigma,
    )

    cos_in = khat_in[:, 2]

    transverse_norm = float(
        np.sum(
            input_grid.weights
            * np.abs(input_U) ** 2
        )
    )

    flux_norm = float(
        np.sum(
            input_grid.weights
            * np.abs(input_U) ** 2
            * cos_in
        )
    )

    qx_in = (
        k * khat_in[:, 0]
    )

    phase_in = np.exp(
        1j
        * qx_in[:, None]
        * b_um[None, :]
    )

    incident_matrix = (
        input_U[:, None]
        * phase_in
        * input_grid.weights[:, None]
    )

    nq_out = int(
        len(output_grid.q)
        // len(
            np.unique(
                np.round(
                    output_grid.phi,
                    14,
                )
            )
        )
    )

    # Grid was constructed as nq x nphi.
    # Infer nphi from repeated q nodes.
    first_q = output_grid.q[0]

    nphi_out = int(
        np.sum(
            np.isclose(
                output_grid.q,
                first_q,
                rtol=0.0,
                atol=1e-14,
            )
        )
    )

    nq_out = int(
        len(output_grid.q)
        // nphi_out
    )

    q2d = output_grid.q.reshape(
        nq_out,
        nphi_out,
    )

    phi2d = output_grid.phi.reshape(
        nq_out,
        nphi_out,
    )

    w2d = output_grid.weights.reshape(
        nq_out,
        nphi_out,
    )

    q_nodes = q2d[:, 0]
    phi_nodes = phi2d[0, :]

    dphi = (
        2.0 * math.pi / nphi_out
    )

    q_wq = (
        w2d[:, 0] / dphi
    )

    cos_out = np.sqrt(
        1.0
        - (q_nodes / k) ** 2
    )

    radial_omega_weights = (
        q_wq
        / (
            k**2 * cos_out
        )
    )

    fft_l = oam_indices(
        nphi_out
    )

    selected = (
        (fft_l >= lmin_out)
        & (fft_l <= lmax_out)
    )

    selected_l = fft_l[
        selected
    ]

    nl = len(selected_l)
    nb = len(b_um)

    powers = {
        +1: np.zeros(
            (
                nl,
                nb,
            ),
            dtype=float,
        ),
        -1: np.zeros(
            (
                nl,
                nb,
            ),
            dtype=float,
        ),
    }

    total_direct_integral = np.zeros(
        nb,
        dtype=float,
    )

    total_fft_integral = np.zeros(
        nb,
        dtype=float,
    )

    qx_all = (
        k
        * output_grid.khat[:, 0]
    ).reshape(
        nq_out,
        nphi_out,
    )

    for iq in range(nq_out):
        j0 = (
            iq * nphi_out
        )
        j1 = (
            j0 + nphi_out
        )

        ks = output_grid.khat[
            j0:j1
        ]

        kernel = flux.scattering_vectors(
            ks,
            khat_in,
            e_in,
            mie_a,
            mie_b,
        )

        field = np.empty(
            (
                nphi_out,
                nb,
                3,
            ),
            dtype=np.complex128,
        )

        for c in range(3):
            field[:, :, c] = (
                kernel[:, :, c]
                @ incident_matrix
            )

        phase_out = np.exp(
            -1j
            * qx_all[iq, :, None]
            * b_um[None, :]
        )

        field *= phase_out[
            :,
            :,
            None,
        ]

        direct_phi_integral = (
            dphi
            * np.sum(
                np.sum(
                    np.abs(field) ** 2,
                    axis=2,
                ),
                axis=0,
            )
        )

        total_direct_integral += (
            radial_omega_weights[iq]
            * direct_phi_integral
        )

        spin_sum_fft = np.zeros(
            nb,
            dtype=float,
        )

        for sigma_out in (
            +1,
            -1,
        ):
            eout = local_circular(
                ks,
                sigma_out,
            )

            scalar = np.einsum(
                "pk,pbk->pb",
                eout.conj(),
                field,
            )

            coeff = (
                math.sqrt(
                    2.0 * math.pi
                )
                / nphi_out
                * np.fft.fft(
                    scalar,
                    axis=0,
                )
            )

            spectral_power_all = (
                np.abs(coeff) ** 2
            )

            spin_sum_fft += np.sum(
                spectral_power_all,
                axis=0,
            )

            powers[sigma_out] += (
                radial_omega_weights[iq]
                * spectral_power_all[
                    selected,
                    :
                ]
            )

        total_fft_integral += (
            radial_omega_weights[iq]
            * spin_sum_fft
        )

    prefactor = (
        1.0
        / (
            4.0
            * math.pi**2
            * k**2
            * flux_norm
        )
    )

    for sigma_out in (
        +1,
        -1,
    ):
        powers[sigma_out] *= (
            prefactor
        )

    total_direct = (
        prefactor
        * total_direct_integral
    )

    total_fft = (
        prefactor
        * total_fft_integral
    )

    selected_sum = (
        np.sum(
            powers[+1],
            axis=0,
        )
        + np.sum(
            powers[-1],
            axis=0,
        )
    )

    return {
        "selected_l": (
            selected_l
        ),
        "powers": powers,
        "total_direct": (
            total_direct
        ),
        "total_fft": total_fft,
        "selected_sum": (
            selected_sum
        ),
        "flux_norm": flux_norm,
        "transverse_norm": (
            transverse_norm
        ),
        "radial_omega_weights": (
            radial_omega_weights
        ),
        "q_nodes": q_nodes,
    }


def radial_cross_section(
    b_um,
    probability,
):
    return float(
        2.0
        * math.pi
        * trapezoid(
            b_um * probability,
            b_um,
        )
    )


def channel_cross_sections(
    *,
    b_um,
    selected_l,
    powers,
    input_l,
    input_sigma,
):
    rows = []

    j_in = (
        input_l + input_sigma
    )

    total_selected = 0.0
    jbreak = 0.0
    spinflip = 0.0
    compensated = 0.0

    for sigma_out in (
        +1,
        -1,
    ):
        for il, lout in enumerate(
            selected_l
        ):
            sigma_ch = (
                radial_cross_section(
                    b_um,
                    powers[
                        sigma_out
                    ][il],
                )
            )

            total_selected += (
                sigma_ch
            )

            is_flip = (
                sigma_out
                != input_sigma
            )

            is_jbreak = (
                (
                    int(lout)
                    + sigma_out
                )
                != j_in
            )

            is_comp = (
                is_flip
                and not is_jbreak
            )

            if is_jbreak:
                jbreak += sigma_ch

            if is_flip:
                spinflip += sigma_ch

            if is_comp:
                compensated += (
                    sigma_ch
                )

            rows.append(
                {
                    "l_out": int(lout),
                    "sigma_out": int(
                        sigma_out
                    ),
                    "j_out": int(
                        int(lout)
                        + sigma_out
                    ),
                    "cross_section_um2": float(
                        sigma_ch
                    ),
                    "is_spinflip": bool(
                        is_flip
                    ),
                    "is_jbreak": bool(
                        is_jbreak
                    ),
                    "is_compensated_so": bool(
                        is_comp
                    ),
                }
            )

    return {
        "rows": rows,
        "total_selected_um2": float(
            total_selected
        ),
        "jbreak_um2": float(
            jbreak
        ),
        "spinflip_um2": float(
            spinflip
        ),
        "compensated_so_um2": float(
            compensated
        ),
    }


def write_channel_csv(
    rows,
    sigma_total,
    sigma_collected,
    path,
):
    enriched = []

    for row in rows:
        copy = dict(row)

        copy[
            "fraction_all_scattering"
        ] = (
            row[
                "cross_section_um2"
            ]
            / sigma_total
        )

        copy[
            "fraction_collected"
        ] = (
            row[
                "cross_section_um2"
            ]
            / sigma_collected
        )

        enriched.append(copy)

    with Path(path).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                enriched[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            enriched
        )

    return enriched


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
        description=(
            "Stage 3B absolute flux-normalized "
            "SAM-OAM scattering branching."
        ),
    )

    p.add_argument(
        "--w0-um",
        type=float,
        required=True,
    )

    p.add_argument(
        "--eta-col",
        type=float,
        default=0.10,
    )

    p.add_argument(
        "--input-l",
        type=int,
        default=0,
    )

    p.add_argument(
        "--input-sigma",
        type=int,
        choices=(-1, +1),
        default=+1,
    )

    p.add_argument(
        "--lmin-out",
        type=int,
        default=-12,
    )

    p.add_argument(
        "--lmax-out",
        type=int,
        default=12,
    )

    p.add_argument(
        "--bmax-over-w0",
        type=float,
        default=3.0,
    )

    p.add_argument(
        "--nb",
        type=int,
        default=121,
    )

    p.add_argument(
        "--wavelength-um",
        type=float,
        default=0.633,
    )

    p.add_argument(
        "--radius-um",
        type=float,
        default=2.0,
    )

    p.add_argument(
        "--n-medium",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--n-sphere-real",
        type=float,
        default=1.33,
    )

    p.add_argument(
        "--n-sphere-imag",
        type=float,
        default=0.0,
    )

    p.add_argument(
        "--nq-in",
        type=int,
        default=48,
    )

    p.add_argument(
        "--nphi-in",
        type=int,
        default=64,
    )

    p.add_argument(
        "--nq-out",
        type=int,
        default=56,
    )

    p.add_argument(
        "--nphi-out",
        type=int,
        default=96,
    )

    p.add_argument(
        "--input-tail-tolerance",
        type=float,
        default=1e-10,
    )

    p.add_argument(
        "--input-qcap-fraction",
        type=float,
        default=0.40,
    )

    p.add_argument(
        "--block-size",
        type=int,
        default=32,
    )

    p.add_argument(
        "--prefix",
        default="sphere_glmt_stage3b_oam",
    )

    return p.parse_args()


def execute(args):
    k = (
        2.0
        * math.pi
        * args.n_medium
        / args.wavelength_um
    )

    n_sphere = complex(
        args.n_sphere_real,
        args.n_sphere_imag,
    )

    m_rel = (
        n_sphere
        / args.n_medium
    )

    x = (
        k * args.radius_um
    )

    (
        mie_a,
        mie_b,
        qsca,
        mie_nmax,
    ) = base.mie_coefficients(
        m_rel,
        x,
    )

    sigma_s = (
        qsca
        * math.pi
        * args.radius_um**2
    )

    q_in_max, input_coverage = (
        base.choose_input_qmax(
            k=k,
            w0=args.w0_um,
            l_input=args.input_l,
            tail_tolerance=args.input_tail_tolerance,
            qcap_fraction=args.input_qcap_fraction,
        )
    )

    input_grid = base.make_grid(
        q_in_max,
        k,
        args.nq_in,
        args.nphi_in,
    )

    output_grid = base.make_grid(
        args.eta_col * k,
        k,
        args.nq_out,
        args.nphi_out,
    )

    U = base.lg_angular_spectrum(
        input_grid.q,
        input_grid.phi,
        args.w0_um,
        0,
        args.input_l,
    )

    b_over_w0 = np.linspace(
        0.0,
        args.bmax_over_w0,
        args.nb,
    )

    b_um = (
        b_over_w0
        * args.w0_um
    )

    calc = compute_b_resolved(
        input_grid=input_grid,
        output_grid=output_grid,
        input_U=U,
        input_sigma=args.input_sigma,
        b_um=b_um,
        mie_a=mie_a,
        mie_b=mie_b,
        k=k,
        lmin_out=args.lmin_out,
        lmax_out=args.lmax_out,
        block_size=args.block_size,
    )

    selected_l = calc[
        "selected_l"
    ]

    channels = channel_cross_sections(
        b_um=b_um,
        selected_l=selected_l,
        powers=calc["powers"],
        input_l=args.input_l,
        input_sigma=args.input_sigma,
    )

    sigma_col_direct = (
        radial_cross_section(
            b_um,
            calc["total_direct"],
        )
    )

    sigma_col_fft = (
        radial_cross_section(
            b_um,
            calc["total_fft"],
        )
    )

    sigma_col_selected = (
        channels[
            "total_selected_um2"
        ]
    )

    sigma_z_expected = (
        sigma_s
        * calc["transverse_norm"]
        / calc["flux_norm"]
    )

    channel_csv = (
        f"{args.prefix}_channels.csv"
    )

    channel_rows = write_channel_csv(
        channels["rows"],
        sigma_z_expected,
        sigma_col_direct,
        channel_csv,
    )

    b_csv = (
        f"{args.prefix}_b_scan.csv"
    )

    with Path(b_csv).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        header = [
            "b_over_w0",
            "b_um",
            "p_col_direct",
            "p_col_fft_all_l",
            "p_col_selected_l",
        ]

        for sigma_out in (
            +1,
            -1,
        ):
            for lout in selected_l:
                header.append(
                    f"p_l{int(lout)}_s{sigma_out:+d}"
                )

        writer.writerow(header)

        for ib in range(
            len(b_um)
        ):
            row = [
                float(
                    b_over_w0[ib]
                ),
                float(b_um[ib]),
                float(
                    calc[
                        "total_direct"
                    ][ib]
                ),
                float(
                    calc[
                        "total_fft"
                    ][ib]
                ),
                float(
                    calc[
                        "selected_sum"
                    ][ib]
                ),
            ]

            for sigma_out in (
                +1,
                -1,
            ):
                for il in range(
                    len(selected_l)
                ):
                    row.append(
                        float(
                            calc[
                                "powers"
                            ][sigma_out][
                                il,
                                ib,
                            ]
                        )
                    )

            writer.writerow(row)

    jbreak = (
        channels[
            "jbreak_um2"
        ]
    )

    spinflip = (
        channels[
            "spinflip_um2"
        ]
    )

    comp = (
        channels[
            "compensated_so_um2"
        ]
    )

    centered_selected = float(
        calc["selected_sum"][0]
    )

    centered_jbreak = 0.0

    j_in = (
        args.input_l
        + args.input_sigma
    )

    for sigma_out in (
        +1,
        -1,
    ):
        for il, lout in enumerate(
            selected_l
        ):
            if (
                int(lout)
                + sigma_out
            ) != j_in:
                centered_jbreak += float(
                    calc[
                        "powers"
                    ][sigma_out][
                        il,
                        0,
                    ]
                )

    pol_in_diag = (
        polarization_diagnostics(
            input_grid.khat
        )
    )

    pol_out_diag = (
        polarization_diagnostics(
            output_grid.khat
        )
    )

    summary = {
        "mie": {
            "size_parameter": float(x),
            "mie_nmax": int(
                mie_nmax
            ),
            "Qsca": float(qsca),
            "sigma_s_um2": float(
                sigma_s
            ),
        },
        "input": {
            "l": int(
                args.input_l
            ),
            "sigma": int(
                args.input_sigma
            ),
            "j": int(
                j_in
            ),
            "input_eta_max": float(
                q_in_max / k
            ),
            "input_coverage": float(
                input_coverage
            ),
            "transverse_norm": float(
                calc[
                    "transverse_norm"
                ]
            ),
            "flux_norm": float(
                calc["flux_norm"]
            ),
            "sigma_z_expected_um2": float(
                sigma_z_expected
            ),
            "rate_correction_factor": float(
                calc[
                    "transverse_norm"
                ]
                / calc[
                    "flux_norm"
                ]
            ),
        },
        "collection": {
            "eta_col": float(
                args.eta_col
            ),
            "NA_col": float(
                args.n_medium
                * args.eta_col
            ),
            "theta_col_deg": float(
                math.degrees(
                    math.asin(
                        args.eta_col
                    )
                )
            ),
            "sigma_col_direct_um2": float(
                sigma_col_direct
            ),
            "sigma_col_fft_um2": float(
                sigma_col_fft
            ),
            "fft_parseval_relative_error": float(
                abs(
                    sigma_col_fft
                    - sigma_col_direct
                )
                / sigma_col_direct
            ),
            "sigma_col_selected_l_um2": float(
                sigma_col_selected
            ),
            "selected_l_capture_fraction": float(
                sigma_col_selected
                / sigma_col_direct
            ),
            "absolute_collected_fraction": float(
                sigma_col_direct
                / sigma_z_expected
            ),
        },
        "branching": {
            "jbreak_um2": float(
                jbreak
            ),
            "spinflip_um2": float(
                spinflip
            ),
            "compensated_so_um2": float(
                comp
            ),
            "jbreak_fraction_all_scattering": float(
                jbreak
                / sigma_z_expected
            ),
            "spinflip_fraction_all_scattering": float(
                spinflip
                / sigma_z_expected
            ),
            "compensated_so_fraction_all_scattering": float(
                comp
                / sigma_z_expected
            ),
            "jbreak_fraction_collected": float(
                jbreak
                / sigma_col_direct
            ),
            "spinflip_fraction_collected": float(
                spinflip
                / sigma_col_direct
            ),
            "compensated_so_fraction_collected": float(
                comp
                / sigma_col_direct
            ),
            "centered_selected_probability": float(
                centered_selected
            ),
            "centered_jbreak_probability": float(
                centered_jbreak
            ),
            "centered_jbreak_fraction": float(
                centered_jbreak
                / centered_selected
                if centered_selected > 0
                else 0.0
            ),
        },
        "polarization_diagnostics": {
            "input_grid": (
                pol_in_diag
            ),
            "output_grid": (
                pol_out_diag
            ),
        },
        "channels": (
            channel_rows
        ),
        "b_scan": {
            "b_over_w0": (
                b_over_w0.tolist()
            ),
            "b_um": (
                b_um.tolist()
            ),
            "p_col_direct": (
                calc[
                    "total_direct"
                ].tolist()
            ),
            "p_col_fft_all_l": (
                calc[
                    "total_fft"
                ].tolist()
            ),
            "p_col_selected_l": (
                calc[
                    "selected_sum"
                ].tolist()
            ),
        },
    }

    full = {
        "script_version": 1,
        "scope": (
            "Stage-3B absolute flux-normalized "
            "SAM-OAM branching with continuous radial trace"
        ),
        "parameters": args_dict(args),
        "summary": summary,
        "notes": {
            "radial_output": (
                "continuous q traced by quadrature; "
                "no LG radial truncation"
            ),
            "polarization": (
                "exact transverse paraxial-gauge circular basis"
            ),
            "uniform_medium": (
                "2 pi integral b db for an incident Jz eigenstate"
            ),
            "stage_limit": (
                "basis-state branching only; "
                "coherence superoperator deferred to Stage 3C"
            ),
        },
    }

    full_path = write_json(
        full,
        f"{args.prefix}_FULL_RESULTS.json",
    )

    manifest = {
        "primary_review_file": str(
            full_path
        ),
        "channels_csv": (
            channel_csv
        ),
        "b_scan_csv": b_csv,
        "console_log": (
            f"{args.prefix}_console.log"
        ),
    }

    manifest_path = write_json(
        manifest,
        f"{args.prefix}_manifest.json",
    )

    print()
    print("=" * 92)
    print("STAGE 3B ABSOLUTE SAM-OAM BRANCHING")
    print("=" * 92)

    print(
        "Expected total scattering cross section [um^2] = "
        f"{sigma_z_expected:.12g}"
    )

    print(
        "Collected cross section [um^2] = "
        f"{sigma_col_direct:.12g}"
    )

    print(
        "Absolute collected fraction = "
        f"{summary['collection']['absolute_collected_fraction']:.9f}"
    )

    print(
        "Selected OAM capture within collection = "
        f"{summary['collection']['selected_l_capture_fraction']:.12f}"
    )

    print(
        "FFT Parseval relative error = "
        f"{summary['collection']['fft_parseval_relative_error']:.6e}"
    )

    print(
        "j-break fraction of all scattering = "
        f"{summary['branching']['jbreak_fraction_all_scattering']:.9f}"
    )

    print(
        "j-break fraction within collection = "
        f"{summary['branching']['jbreak_fraction_collected']:.9f}"
    )

    print(
        "spin-flip fraction of all scattering = "
        f"{summary['branching']['spinflip_fraction_all_scattering']:.9e}"
    )

    print(
        "spin-flip fraction within collection = "
        f"{summary['branching']['spinflip_fraction_collected']:.9e}"
    )

    print(
        "centered j-break fraction = "
        f"{summary['branching']['centered_jbreak_fraction']:.6e}"
    )

    print()
    print(f"Primary review file: {full_path}")
    print(f"Manifest: {manifest_path}")


def main():
    args = parse_args()

    log_path = Path(
        f"{args.prefix}_console.log"
    )

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as log_file:
        sys.stdout = TeeStream(
            original_stdout,
            log_file,
        )

        sys.stderr = TeeStream(
            original_stderr,
            log_file,
        )

        try:
            print(
                "Complete console transcript "
                f"is saved to {log_path}"
            )

            execute(args)

        except Exception:
            print()
            print("=" * 92)
            print("RUN FAILED")
            print("=" * 92)
            traceback.print_exc()
            raise

        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    print(
        f"Console transcript saved: {log_path}"
    )


if __name__ == "__main__":
    main()
