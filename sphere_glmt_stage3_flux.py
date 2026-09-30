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
from numpy.polynomial.legendre import leggauss


def load_base_module():
    here = Path(__file__).resolve().parent
    candidate = here / "sphere_glmt_fixed_collection_v3.py"

    if not candidate.exists():
        raise FileNotFoundError(
            "sphere_glmt_fixed_collection_v3.py must be in the same folder."
        )

    spec = importlib.util.spec_from_file_location(
        "sphere_glmt_fixed_collection_v3_stage3",
        candidate,
    )

    if spec is None or spec.loader is None:
        raise ImportError("Could not load fixed-collection module.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


base = load_base_module()


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


def hermitian_part(a):
    return 0.5 * (a + a.conj().T)


def inv_sqrt_hermitian(a, floor=1e-14):
    vals, vecs = np.linalg.eigh(
        hermitian_part(a)
    )

    if np.min(vals) <= floor:
        raise RuntimeError(
            "Gram matrix is not positive definite."
        )

    inv_root = (
        vecs
        @ np.diag(vals ** -0.5)
        @ vecs.conj().T
    )

    return inv_root, vals


def make_full_sphere_grid(nmu, nphi):
    mu, wmu = leggauss(nmu)

    phi = np.linspace(
        0.0,
        2.0 * math.pi,
        nphi,
        endpoint=False,
    )

    dphi = 2.0 * math.pi / nphi

    mu2d, phi2d = np.meshgrid(
        mu,
        phi,
        indexing="ij",
    )

    wmu2d, _ = np.meshgrid(
        wmu,
        phi,
        indexing="ij",
    )

    st = np.sqrt(
        np.maximum(
            0.0,
            1.0 - mu2d**2,
        )
    )

    khat = np.column_stack(
        (
            (
                st * np.cos(phi2d)
            ).ravel(),
            (
                st * np.sin(phi2d)
            ).ravel(),
            mu2d.ravel(),
        )
    )

    weights = (
        wmu2d * dphi
    ).ravel()

    return khat, weights


def scattering_vectors(
    khat_out,
    khat_in,
    e_in,
    mie_a,
    mie_b,
):
    cross = np.cross(
        khat_in[None, :, :],
        khat_out[:, None, :],
    )

    cross_norm = np.linalg.norm(
        cross,
        axis=2,
    )

    mu = np.clip(
        khat_out @ khat_in.T,
        -1.0,
        1.0,
    )

    s1, s2 = base.mie_amplitudes(
        mu,
        mie_a,
        mie_b,
    )

    n_perp = np.zeros_like(cross)

    regular = cross_norm > 1e-12

    n_perp[regular] = (
        cross[regular]
        / cross_norm[regular, None]
    )

    epar_in = np.cross(
        n_perp,
        khat_in[None, :, :],
    )

    epar_out = np.cross(
        n_perp,
        khat_out[:, None, :],
    )

    inc_perp = np.einsum(
        "oik,ik->oi",
        n_perp,
        e_in,
    )

    inc_par = np.einsum(
        "oik,ik->oi",
        epar_in,
        e_in,
    )

    e_scat = (
        s1[:, :, None]
        * inc_perp[:, :, None]
        * n_perp
        +
        s2[:, :, None]
        * inc_par[:, :, None]
        * epar_out
    )

    degenerate = ~regular

    if np.any(degenerate):
        oo, ii = np.nonzero(
            degenerate
        )

        e_scat[oo, ii, :] = (
            s1[oo, ii, None]
            * e_in[ii, :]
        )

    return e_scat


def orthonormal_transverse_pair(khat):
    ep = base.projected_global_circular(
        khat,
        +1,
    )

    em = base.projected_global_circular(
        khat,
        -1,
    )

    result_plus = np.empty_like(ep)
    result_minus = np.empty_like(em)

    max_change = 0.0
    max_overlap_before = 0.0

    for i in range(len(khat)):
        e = np.column_stack(
            (
                ep[i],
                em[i],
            )
        )

        gram = e.conj().T @ e

        max_overlap_before = max(
            max_overlap_before,
            abs(gram[0, 1]),
        )

        vals, vecs = np.linalg.eigh(
            hermitian_part(gram)
        )

        inv_root = (
            vecs
            @ np.diag(vals ** -0.5)
            @ vecs.conj().T
        )

        f = e @ inv_root

        result_plus[i] = f[:, 0]
        result_minus[i] = f[:, 1]

        max_change = max(
            max_change,
            np.linalg.norm(
                f[:, 0] - ep[i]
            ),
            np.linalg.norm(
                f[:, 1] - em[i]
            ),
        )

    return (
        result_plus,
        result_minus,
        float(max_overlap_before),
        float(max_change),
    )


def numerical_plane_wave_cross_section(
    *,
    khat_out,
    omega_weights,
    khat_in,
    e_in,
    mie_a,
    mie_b,
    k,
):
    e_scat = scattering_vectors(
        khat_out,
        khat_in[None, :],
        e_in[None, :],
        mie_a,
        mie_b,
    )[:, 0, :]

    integral = np.sum(
        omega_weights
        * np.sum(
            np.abs(e_scat) ** 2,
            axis=1,
        )
    )

    return float(
        integral / k**2
    )


def build_input_basis(
    *,
    input_grid,
    w0_um,
    lmin,
    lmax,
):
    khat = input_grid.khat

    ep = base.projected_global_circular(
        khat,
        +1,
    )

    em = base.projected_global_circular(
        khat,
        -1,
    )

    labels = []
    columns = []

    for sigma, pol in (
        (+1, ep),
        (-1, em),
    ):
        for l in range(lmin, lmax + 1):
            U = base.lg_angular_spectrum(
                input_grid.q,
                input_grid.phi,
                w0_um,
                0,
                l,
            )

            columns.append(
                U[:, None] * pol
            )

            labels.append(
                {
                    "l": int(l),
                    "sigma": int(sigma),
                }
            )

    modes = np.stack(
        columns,
        axis=-1,
    )
    # Shape: Nq, 3, Nbasis.

    w = input_grid.weights
    cos_theta = khat[:, 2]

    gram_transverse = np.einsum(
        "n,nci,ncj->ij",
        w,
        modes.conj(),
        modes,
    )

    gram_flux = np.einsum(
        "n,n,nci,ncj->ij",
        w,
        cos_theta,
        modes.conj(),
        modes,
    )

    gram_transverse = hermitian_part(
        gram_transverse
    )

    gram_flux = hermitian_part(
        gram_flux
    )

    inv_root_flux, flux_eigs = (
        inv_sqrt_hermitian(
            gram_flux
        )
    )

    rate_shape = hermitian_part(
        inv_root_flux.conj().T
        @ gram_transverse
        @ inv_root_flux
    )

    rate_eigs = np.linalg.eigvalsh(
        rate_shape
    )

    return {
        "labels": labels,
        "gram_transverse": gram_transverse,
        "gram_flux": gram_flux,
        "flux_eigenvalues": flux_eigs,
        "rate_shape": rate_shape,
        "rate_shape_eigenvalues": rate_eigs,
    }


def beam_power_scan(
    *,
    input_grid,
    input_U,
    input_sigma,
    b_over_w0,
    w0_um,
    full_khat,
    full_weights,
    collection_grid,
    mie_a,
    mie_b,
    k,
    block_size,
):
    khat_in = input_grid.khat

    e_in = base.projected_global_circular(
        khat_in,
        input_sigma,
    )

    flux_norm = float(
        np.sum(
            input_grid.weights
            * np.abs(input_U) ** 2
            * khat_in[:, 2]
        )
    )

    transverse_norm = float(
        np.sum(
            input_grid.weights
            * np.abs(input_U) ** 2
        )
    )

    b_um = (
        np.asarray(
            b_over_w0,
            dtype=float,
        )
        * w0_um
    )

    qx_in = (
        k * khat_in[:, 0]
    )

    phase = np.exp(
        1j
        * qx_in[:, None]
        * b_um[None, :]
    )

    incident_matrix = (
        input_U[:, None]
        * phase
        * input_grid.weights[:, None]
    )

    nb = len(b_um)

    full_acc = np.zeros(
        nb,
        dtype=float,
    )

    col_acc = np.zeros(
        nb,
        dtype=float,
    )

    # Full sphere.
    for j0 in range(
        0,
        len(full_khat),
        block_size,
    ):
        j1 = min(
            j0 + block_size,
            len(full_khat),
        )

        ks = full_khat[j0:j1]

        kernel = scattering_vectors(
            ks,
            khat_in,
            e_in,
            mie_a,
            mie_b,
        )

        field = np.empty(
            (
                len(ks),
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

        full_acc += np.sum(
            full_weights[j0:j1, None]
            * np.sum(
                np.abs(field) ** 2,
                axis=2,
            ),
            axis=0,
        ).real

    # Collection aperture.
    khat_col = collection_grid.khat

    cos_col = khat_col[:, 2]

    omega_col_weights = (
        collection_grid.weights
        / (
            k**2
            * cos_col
        )
    )

    for j0 in range(
        0,
        len(khat_col),
        block_size,
    ):
        j1 = min(
            j0 + block_size,
            len(khat_col),
        )

        ks = khat_col[j0:j1]

        kernel = scattering_vectors(
            ks,
            khat_in,
            e_in,
            mie_a,
            mie_b,
        )

        field = np.empty(
            (
                len(ks),
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

        col_acc += np.sum(
            omega_col_weights[
                j0:j1,
                None,
            ]
            * np.sum(
                np.abs(field) ** 2,
                axis=2,
            ),
            axis=0,
        ).real

    prefactor = (
        1.0
        / (
            4.0
            * math.pi**2
            * k**2
            * flux_norm
        )
    )

    p_full = prefactor * full_acc
    p_col = prefactor * col_acc

    return {
        "flux_norm": flux_norm,
        "transverse_norm": transverse_norm,
        "b_um": b_um,
        "p_scat_full": p_full,
        "p_scat_collected": p_col,
    }


def radial_cross_section(
    b_um,
    probability,
    max_b_um=None,
):
    b_um = np.asarray(
        b_um,
        dtype=float,
    )

    probability = np.asarray(
        probability,
        dtype=float,
    )

    if max_b_um is not None:
        mask = b_um <= (
            max_b_um + 1e-12
        )
        b_um = b_um[mask]
        probability = probability[mask]

    if len(b_um) < 2:
        return float("nan")

    return float(
        2.0
        * math.pi
        * trapezoid(
            b_um * probability,
            b_um,
        )
    )


def write_scan_csv(scan, path):
    b_um = scan["b_um"]
    p_full = scan["p_scat_full"]
    p_col = scan["p_scat_collected"]

    with Path(path).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "b_um",
                "b_over_w0",
                "p_scat_full",
                "p_scat_collected",
                "collected_over_full",
            ]
        )

        for b, pf, pc in zip(
            b_um,
            p_full,
            p_col,
        ):
            writer.writerow(
                [
                    float(b),
                    float(b / b_um[-1])
                    if False
                    else "",
                    float(pf),
                    float(pc),
                    float(pc / pf)
                    if pf > 0
                    else float("nan"),
                ]
            )


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
        description=(
            "Stage 3A physical flux normalization "
            "for spherical Mie scattering."
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
        "--basis-lmin",
        type=int,
        default=-6,
    )

    p.add_argument(
        "--basis-lmax",
        type=int,
        default=6,
    )

    p.add_argument(
        "--scan-input-l",
        type=int,
        default=0,
    )

    p.add_argument(
        "--scan-input-sigma",
        type=int,
        choices=(-1, 1),
        default=1,
    )

    p.add_argument(
        "--bmax-over-w0",
        type=float,
        default=4.0,
    )

    p.add_argument(
        "--nb",
        type=int,
        default=41,
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
        "--nq-col",
        type=int,
        default=56,
    )

    p.add_argument(
        "--nphi-col",
        type=int,
        default=64,
    )

    p.add_argument(
        "--nmu-full",
        type=int,
        default=48,
    )

    p.add_argument(
        "--nphi-full",
        type=int,
        default=64,
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
        default="sphere_glmt_stage3_flux",
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

    sigma_s_mie = (
        qsca
        * math.pi
        * args.radius_um**2
    )

    # Optical theorem from the same Mie amplitudes.
    s1_forward, s2_forward = (
        base.mie_amplitudes(
            np.array([1.0]),
            mie_a,
            mie_b,
        )
    )

    s_forward = 0.5 * (
        s1_forward[0]
        + s2_forward[0]
    )

    sigma_ext_optical = (
        4.0
        * math.pi
        / k**2
        * np.real(s_forward)
    )

    # Full sphere quadrature.
    full_khat, full_weights = (
        make_full_sphere_grid(
            args.nmu_full,
            args.nphi_full,
        )
    )

    khat_incident = np.array(
        [0.0, 0.0, 1.0],
        dtype=float,
    )

    e_x = np.array(
        [1.0, 0.0, 0.0],
        dtype=np.complex128,
    )

    e_circ = np.array(
        [
            1.0,
            1j,
            0.0,
        ],
        dtype=np.complex128,
    ) / math.sqrt(2.0)

    sigma_num_x = (
        numerical_plane_wave_cross_section(
            khat_out=full_khat,
            omega_weights=full_weights,
            khat_in=khat_incident,
            e_in=e_x,
            mie_a=mie_a,
            mie_b=mie_b,
            k=k,
        )
    )

    sigma_num_circ = (
        numerical_plane_wave_cross_section(
            khat_out=full_khat,
            omega_weights=full_weights,
            khat_in=khat_incident,
            e_in=e_circ,
            mie_a=mie_a,
            mie_b=mie_b,
            k=k,
        )
    )

    # Input grid must cover every l used in rate-operator validation.
    l_for_qmax = max(
        abs(args.basis_lmin),
        abs(args.basis_lmax),
        abs(args.scan_input_l),
    )

    q_in_max, input_coverage_worst = (
        base.choose_input_qmax(
            k=k,
            w0=args.w0_um,
            l_input=l_for_qmax,
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

    collection_grid = base.make_grid(
        args.eta_col * k,
        k,
        args.nq_col,
        args.nphi_col,
    )

    basis = build_input_basis(
        input_grid=input_grid,
        w0_um=args.w0_um,
        lmin=args.basis_lmin,
        lmax=args.basis_lmax,
    )

    rate_eigs = basis[
        "rate_shape_eigenvalues"
    ]

    rate_deviation_max = float(
        np.max(
            np.abs(
                rate_eigs - 1.0
            )
        )
    )

    rate_deviation_fro = float(
        np.linalg.norm(
            basis["rate_shape"]
            - np.eye(
                len(rate_eigs)
            )
        )
    )

    # Polarization-basis nonorthogonality within collection.
    (
        _ep_col,
        _em_col,
        max_pol_overlap,
        max_pol_lowdin_change,
    ) = orthonormal_transverse_pair(
        collection_grid.khat
    )

    # Absolute beam scattering scan.
    U_scan = base.lg_angular_spectrum(
        input_grid.q,
        input_grid.phi,
        args.w0_um,
        0,
        args.scan_input_l,
    )

    b_over_w0 = np.linspace(
        0.0,
        args.bmax_over_w0,
        args.nb,
    )

    scan = beam_power_scan(
        input_grid=input_grid,
        input_U=U_scan,
        input_sigma=args.scan_input_sigma,
        b_over_w0=b_over_w0,
        w0_um=args.w0_um,
        full_khat=full_khat,
        full_weights=full_weights,
        collection_grid=collection_grid,
        mie_a=mie_a,
        mie_b=mie_b,
        k=k,
        block_size=args.block_size,
    )

    scan["b_over_w0"] = (
        b_over_w0
    )

    sigma_full_numeric = (
        radial_cross_section(
            scan["b_um"],
            scan["p_scat_full"],
        )
    )

    sigma_col_numeric = (
        radial_cross_section(
            scan["b_um"],
            scan["p_scat_collected"],
        )
    )

    # Analytic translation-invariance expectation.
    expected_sigma_z = (
        sigma_s_mie
        * scan["transverse_norm"]
        / scan["flux_norm"]
    )

    cumulative = {}

    for scale in (
        2.0,
        3.0,
        args.bmax_over_w0,
    ):
        if scale <= (
            args.bmax_over_w0 + 1e-12
        ):
            cutoff = (
                scale * args.w0_um
            )

            cumulative[str(scale)] = {
                "sigma_full_um2": (
                    radial_cross_section(
                        scan["b_um"],
                        scan["p_scat_full"],
                        cutoff,
                    )
                ),
                "sigma_collected_um2": (
                    radial_cross_section(
                        scan["b_um"],
                        scan[
                            "p_scat_collected"
                        ],
                        cutoff,
                    )
                ),
            }

    scan_csv = (
        f"{args.prefix}_b_scan.csv"
    )

    with Path(scan_csv).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "b_over_w0",
                "b_um",
                "p_scat_full",
                "p_scat_collected",
                "collected_over_full",
            ]
        )

        for u, b, pf, pc in zip(
            b_over_w0,
            scan["b_um"],
            scan["p_scat_full"],
            scan["p_scat_collected"],
        ):
            writer.writerow(
                [
                    float(u),
                    float(b),
                    float(pf),
                    float(pc),
                    (
                        float(pc / pf)
                        if pf > 0
                        else float("nan")
                    ),
                ]
            )

    summary = {
        "mie": {
            "size_parameter": float(x),
            "mie_nmax": int(mie_nmax),
            "Qsca": float(qsca),
            "sigma_s_mie_um2": float(
                sigma_s_mie
            ),
            "sigma_ext_optical_um2": float(
                sigma_ext_optical
            ),
            "optical_theorem_relative_error": float(
                abs(
                    sigma_ext_optical
                    - sigma_s_mie
                )
                / sigma_s_mie
            ),
            "sigma_s_fullsphere_x_um2": float(
                sigma_num_x
            ),
            "sigma_s_fullsphere_circular_um2": float(
                sigma_num_circ
            ),
            "fullsphere_x_relative_error": float(
                abs(
                    sigma_num_x
                    - sigma_s_mie
                )
                / sigma_s_mie
            ),
            "fullsphere_circular_relative_error": float(
                abs(
                    sigma_num_circ
                    - sigma_s_mie
                )
                / sigma_s_mie
            ),
        },
        "input_basis": {
            "lmin": int(
                args.basis_lmin
            ),
            "lmax": int(
                args.basis_lmax
            ),
            "dimension": int(
                len(
                    basis["labels"]
                )
            ),
            "labels": basis["labels"],
            "input_eta_max": float(
                q_in_max / k
            ),
            "worst_mode_input_coverage": float(
                input_coverage_worst
            ),
            "flux_gram_eigenvalue_min": float(
                np.min(
                    basis[
                        "flux_eigenvalues"
                    ]
                )
            ),
            "flux_gram_eigenvalue_max": float(
                np.max(
                    basis[
                        "flux_eigenvalues"
                    ]
                )
            ),
            "rate_shape_eigenvalue_min": float(
                np.min(rate_eigs)
            ),
            "rate_shape_eigenvalue_max": float(
                np.max(rate_eigs)
            ),
            "max_rate_scalar_deviation": (
                rate_deviation_max
            ),
            "rate_shape_frobenius_deviation": (
                rate_deviation_fro
            ),
        },
        "collection_polarization": {
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
            "max_projected_circular_overlap": (
                max_pol_overlap
            ),
            "max_lowdin_vector_change": (
                max_pol_lowdin_change
            ),
        },
        "beam_scan": {
            "input_l": int(
                args.scan_input_l
            ),
            "input_sigma": int(
                args.scan_input_sigma
            ),
            "flux_norm": float(
                scan["flux_norm"]
            ),
            "transverse_norm": float(
                scan[
                    "transverse_norm"
                ]
            ),
            "expected_sigma_z_um2": float(
                expected_sigma_z
            ),
            "sigma_full_numeric_um2": float(
                sigma_full_numeric
            ),
            "sigma_full_relative_error": float(
                abs(
                    sigma_full_numeric
                    - expected_sigma_z
                )
                / expected_sigma_z
            ),
            "sigma_collected_numeric_um2": float(
                sigma_col_numeric
            ),
            "collected_cross_section_fraction": float(
                sigma_col_numeric
                / sigma_full_numeric
            ),
            "max_single_particle_scattered_fraction": float(
                np.max(
                    scan["p_scat_full"]
                )
            ),
            "max_single_particle_collected_fraction": float(
                np.max(
                    scan[
                        "p_scat_collected"
                    ]
                )
            ),
            "cumulative_cross_sections": (
                cumulative
            ),
            "b_over_w0": (
                b_over_w0.tolist()
            ),
            "b_um": (
                scan["b_um"].tolist()
            ),
            "p_scat_full": (
                scan[
                    "p_scat_full"
                ].tolist()
            ),
            "p_scat_collected": (
                scan[
                    "p_scat_collected"
                ].tolist()
            ),
        },
    }

    full = {
        "script_version": 1,
        "scope": (
            "Stage-3A physical flux normalization "
            "and homogeneous event-rate validation"
        ),
        "parameters": args_dict(args),
        "summary": summary,
        "notes": {
            "event_picture": (
                "scattering jumps only; coherent forward "
                "mean-field phase is not yet included"
            ),
            "uniform_medium_measure": (
                "n_v f(a) da d^2b dz"
            ),
            "generator_target": (
                "n_v integral d^2b "
                "[J_col rho J_col^dagger "
                "- 1/2{J_full^dagger J_full,rho}]"
            ),
            "radial_trace": (
                "deferred to Stage 3B after absolute "
                "normalization passes"
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
        "b_scan_csv": scan_csv,
        "console_log": (
            f"{args.prefix}_console.log"
        ),
    }

    manifest_path = write_json(
        manifest,
        f"{args.prefix}_manifest.json",
    )

    print()
    print("=" * 88)
    print("STAGE 3A FLUX NORMALIZATION")
    print("=" * 88)

    print(
        "Mie sigma_s [um^2] = "
        f"{sigma_s_mie:.12g}"
    )

    print(
        "Full-sphere x-pol relative error = "
        f"{summary['mie']['fullsphere_x_relative_error']:.6e}"
    )

    print(
        "Optical-theorem relative error = "
        f"{summary['mie']['optical_theorem_relative_error']:.6e}"
    )

    print(
        "Rate-shape eigenvalue range = "
        f"[{np.min(rate_eigs):.9f}, "
        f"{np.max(rate_eigs):.9f}]"
    )

    print(
        "Maximum scalar-rate deviation = "
        f"{rate_deviation_max:.6e}"
    )

    print(
        "Expected homogeneous b-integral [um^2] = "
        f"{expected_sigma_z:.12g}"
    )

    print(
        "Numerical homogeneous b-integral [um^2] = "
        f"{sigma_full_numeric:.12g}"
    )

    print(
        "b-integral relative error = "
        f"{summary['beam_scan']['sigma_full_relative_error']:.6e}"
    )

    print(
        "Collected cross-section fraction = "
        f"{summary['beam_scan']['collected_cross_section_fraction']:.9f}"
    )

    print(
        "Maximum one-particle scattered fraction = "
        f"{summary['beam_scan']['max_single_particle_scattered_fraction']:.9f}"
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
            print("=" * 88)
            print("RUN FAILED")
            print("=" * 88)
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
