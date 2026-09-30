#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import (
    eval_genlaguerre,
    gammainc,
    gammaincinv,
    gammaln,
    spherical_jn,
    spherical_yn,
)

class AngularGrid:
    q: np.ndarray
    phi: np.ndarray
    weights: np.ndarray
    khat: np.ndarray
    qmax: float


@dataclass
class Results:
    pmax_values: np.ndarray
    b_over_w0: np.ndarray

    sampled_power: np.ndarray
    retained_power: np.ndarray
    capture_fraction: np.ndarray

    conversion_fraction: np.ndarray
    radial_fraction: np.ndarray
    spinflip_fraction: np.ndarray
    compensated_so_fraction: np.ndarray
    jbreak_fraction: np.ndarray

    phi_mass_covered: float
    phi_capture_fraction: np.ndarray
    phi_conversion_fraction: np.ndarray
    phi_radial_fraction: np.ndarray
    phi_spinflip_fraction: np.ndarray
    phi_compensated_so_fraction: np.ndarray
    phi_jbreak_fraction: np.ndarray
    onaxis_jbreak_fraction: np.ndarray

    qsca: float
    mie_nmax: int

    eta_col: float
    na_col: float
    theta_col_deg: float
    paraxial_phase_error_fraction: float

    eta_in_max: float
    input_coverage: float

    min_output_mode_aperture_fraction: float
    max_output_mode_aperture_fraction: float
    max_output_mode_norm_quadrature_error: float



def mie_coefficients(m_rel: complex, x: float, nmax: int | None = None):
    if x <= 0:
        raise ValueError("Mie size parameter x must be positive.")

    if nmax is None:
        nmax = int(math.ceil(x + 4.0 * x ** (1.0 / 3.0) + 2.0))

    n = np.arange(1, nmax + 1)
    mx = m_rel * x

    jx = spherical_jn(n, x)
    jx_p = spherical_jn(n, x, derivative=True)
    yx = spherical_yn(n, x)
    yx_p = spherical_yn(n, x, derivative=True)

    jmx = spherical_jn(n, mx)
    jmx_p = spherical_jn(n, mx, derivative=True)

    psi_x = x * jx
    psi_x_p = jx + x * jx_p

    psi_mx = mx * jmx
    psi_mx_p = jmx + mx * jmx_p

    xi_x = x * (jx + 1j * yx)
    xi_x_p = (jx + 1j * yx) + x * (jx_p + 1j * yx_p)

    a_num = m_rel * psi_mx * psi_x_p - psi_x * psi_mx_p
    a_den = m_rel * psi_mx * xi_x_p - xi_x * psi_mx_p

    b_num = psi_mx * psi_x_p - m_rel * psi_x * psi_mx_p
    b_den = psi_mx * xi_x_p - m_rel * xi_x * psi_mx_p

    a = a_num / a_den
    b = b_num / b_den

    qsca = (2.0 / x**2) * np.sum(
        (2.0 * n + 1.0) * (np.abs(a) ** 2 + np.abs(b) ** 2)
    )

    return a, b, float(np.real(qsca)), nmax


def mie_amplitudes(mu: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Bohren-Huffman Mie amplitude functions S1 and S2."""
    mu = np.asarray(mu, dtype=float)

    s1 = np.zeros(mu.shape, dtype=np.complex128)
    s2 = np.zeros(mu.shape, dtype=np.complex128)

    pi_nm2 = np.zeros(mu.shape, dtype=float)
    pi_nm1 = np.ones(mu.shape, dtype=float)

    for idx in range(1, len(a) + 1):
        n = idx

        if n == 1:
            pi_n = pi_nm1
            pi_prev = pi_nm2
        else:
            pi_n = (
                ((2.0 * n - 1.0) / (n - 1.0)) * mu * pi_nm1
                - (n / (n - 1.0)) * pi_nm2
            )
            pi_prev = pi_nm1

        tau_n = n * mu * pi_n - (n + 1.0) * pi_prev
        c = (2.0 * n + 1.0) / (n * (n + 1.0))

        s1 += c * (a[n - 1] * pi_n + b[n - 1] * tau_n)
        s2 += c * (a[n - 1] * tau_n + b[n - 1] * pi_n)

        if n >= 2:
            pi_nm2, pi_nm1 = pi_nm1, pi_n

    return s1, s2



def lg_angular_spectrum(
    q: np.ndarray,
    phi: np.ndarray,
    w0: float,
    p: int,
    l: int,
):
    if p < 0:
        raise ValueError("p must be nonnegative.")

    n = abs(l)
    t = 0.5 * (w0 * q) ** 2

    log_norm = (
        math.log(w0)
        - 0.5 * math.log(2.0 * math.pi)
        + 0.5 * (gammaln(p + 1.0) - gammaln(p + n + 1.0))
    )

    radial = (
        math.exp(log_norm)
        * np.power(t, 0.5 * n)
        * eval_genlaguerre(p, n, t)
        * np.exp(-0.5 * t)
    )

    phase = ((-1j) ** l) * ((-1) ** p)

    return phase * radial * np.exp(1j * l * phi)


def mode_aperture_fraction(tmax: float, p: int, l: int):
    n = abs(l)

    # Enough nodes for the requested radial orders.
    nodes, weights = leggauss(max(120, 8 * (p + n + 1)))
    t = 0.5 * tmax * (nodes + 1.0)
    wt = 0.5 * tmax * weights

    pref = math.exp(gammaln(p + 1.0) - gammaln(p + n + 1.0))
    L = eval_genlaguerre(p, n, t)

    density = pref * np.power(t, n) * (L**2) * np.exp(-t)

    return float(np.sum(density * wt))


def projected_global_circular(khat: np.ndarray, sigma: int):
    if sigma not in (-1, 1):
        raise ValueError("sigma must equal +/-1.")

    e0 = np.array([1.0, 1j * sigma, 0.0], dtype=np.complex128)
    e0 /= math.sqrt(2.0)

    longitudinal = khat @ e0
    e = e0[None, :] - khat * longitudinal[:, None]

    norms = np.sqrt(np.sum(np.abs(e) ** 2, axis=1))

    if np.any(norms <= 0):
        raise RuntimeError("Failed transverse polarization construction.")

    return e / norms[:, None]


def make_grid(qmax: float, k: float, nq: int, nphi: int):
    if not (0.0 < qmax < k):
        raise ValueError("Require 0 < qmax < k.")
    if nq < 6:
        raise ValueError("nq must be at least 6.")
    if nphi < 16:
        raise ValueError("nphi must be at least 16.")

    gx, gw = leggauss(nq)

    q_nodes = 0.5 * qmax * (gx + 1.0)
    q_weights = 0.5 * qmax * gw

    phi_nodes = np.linspace(
        0.0,
        2.0 * math.pi,
        nphi,
        endpoint=False,
    )
    dphi = 2.0 * math.pi / nphi

    q2d, phi2d = np.meshgrid(
        q_nodes,
        phi_nodes,
        indexing="ij",
    )
    wq2d, _ = np.meshgrid(
        q_weights,
        phi_nodes,
        indexing="ij",
    )

    q = q2d.ravel()
    phi = phi2d.ravel()
    weights = (q2d * wq2d * dphi).ravel()

    qx = q * np.cos(phi)
    qy = q * np.sin(phi)

    kz = np.sqrt(k**2 - q**2)

    khat = np.column_stack(
        (qx / k, qy / k, kz / k)
    )

    return AngularGrid(
        q=q,
        phi=phi,
        weights=weights,
        khat=khat,
        qmax=qmax,
    )


def choose_input_qmax(
    *,
    k: float,
    w0: float,
    l_input: int,
    tail_tolerance: float,
    qcap_fraction: float,
):
    if not (0.0 < tail_tolerance < 1.0):
        raise ValueError("input tail tolerance must lie in (0,1).")

    t_needed = gammaincinv(
        abs(l_input) + 1,
        1.0 - tail_tolerance,
    )

    q_needed = math.sqrt(2.0 * t_needed) / w0
    qmax = min(q_needed, qcap_fraction * k)

    t_actual = 0.5 * (w0 * qmax) ** 2
    coverage = float(gammainc(abs(l_input) + 1, t_actual))

    return qmax, coverage



def build_projection(
    output_grid: AngularGrid,
    w0: float,
    pmax: int,
    l_values: np.ndarray,
):
    rows = []
    labels = []

    for p in range(pmax + 1):
        for l in l_values:
            u = lg_angular_spectrum(
                output_grid.q,
                output_grid.phi,
                w0,
                p,
                int(l),
            )

            rows.append(u.conj() * output_grid.weights)
            labels.append((p, int(l)))

    return np.asarray(rows), labels


def output_mode_diagnostics(
    *,
    output_grid: AngularGrid,
    w0: float,
    pmax: int,
    l_values: np.ndarray,
):
    tmax = 0.5 * (w0 * output_grid.qmax) ** 2

    fractions = []
    errors = []

    for p in range(pmax + 1):
        for l in l_values:
            u = lg_angular_spectrum(
                output_grid.q,
                output_grid.phi,
                w0,
                p,
                int(l),
            )

            numeric = float(
                np.sum(np.abs(u) ** 2 * output_grid.weights).real
            )

            analytic = mode_aperture_fraction(
                tmax,
                p,
                int(l),
            )

            fractions.append(analytic)
            errors.append(abs(numeric - analytic))

    return (
        float(min(fractions)),
        float(max(fractions)),
        float(max(errors)),
    )


def scatter_project_all_b(
    *,
    input_grid: AngularGrid,
    output_grid: AngularGrid,
    input_sigma: int,
    mie_a: np.ndarray,
    mie_b: np.ndarray,
    projection: np.ndarray,
    input_spectrum: np.ndarray,
    b_values: np.ndarray,
    w0_um: float,
    k: float,
    block_size: int,
):
    khat_in = input_grid.khat
    khat_out = output_grid.khat

    e_in = projected_global_circular(khat_in, input_sigma)

    e_out = {
        +1: projected_global_circular(khat_out, +1),
        -1: projected_global_circular(khat_out, -1),
    }

    qx_in = k * khat_in[:, 0]
    b_um = np.asarray(b_values, float) * w0_um

    phase_matrix = np.exp(
        1j * qx_in[:, None] * b_um[None, :]
    )

    incident_matrix = (
        input_spectrum[:, None]
        * phase_matrix
        * input_grid.weights[:, None]
    )

    nb = len(b_values)
    nmodes = projection.shape[0]

    sampled_power = np.zeros(nb)

    coefficients = {
        +1: np.zeros((nmodes, nb), dtype=np.complex128),
        -1: np.zeros((nmodes, nb), dtype=np.complex128),
    }

    n_out = len(khat_out)

    for j0 in range(0, n_out, block_size):
        j1 = min(j0 + block_size, n_out)
        ks = khat_out[j0:j1]

        cross = np.cross(
            khat_in[None, :, :],
            ks[:, None, :],
        )

        cross_norm = np.linalg.norm(cross, axis=2)

        mu = np.clip(
            ks @ khat_in.T,
            -1.0,
            1.0,
        )

        s1, s2 = mie_amplitudes(mu, mie_a, mie_b)

        n_perp = np.zeros_like(cross)

        regular = cross_norm > 1e-12

        n_perp[regular] = (
            cross[regular] / cross_norm[regular, None]
        )

        epar_in = np.cross(
            n_perp,
            khat_in[None, :, :],
        )

        epar_out = np.cross(
            n_perp,
            ks[:, None, :],
        )

        inc_perp = np.einsum(
            "bik,ik->bi",
            n_perp,
            e_in,
        )

        inc_par = np.einsum(
            "bik,ik->bi",
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
            jj, ii = np.nonzero(degenerate)
            e_scat[jj, ii, :] = (
                s1[jj, ii, None] * e_in[ii, :]
            )

        for sigma_out in (+1, -1):
            eo = e_out[sigma_out][j0:j1]

            kernel_block = np.einsum(
                "bk,bik->bi",
                eo.conj(),
                e_scat,
            )

            field_block_centered = kernel_block @ incident_matrix

            qx_out_block = k * ks[:, 0]
            output_translation = np.exp(
                -1j * qx_out_block[:, None] * b_um[None, :]
            )

            field_block = (
                field_block_centered * output_translation
            )

            sampled_power += np.sum(
                np.abs(field_block) ** 2
                * output_grid.weights[j0:j1, None],
                axis=0,
            ).real

            coefficients[sigma_out] += (
                projection[:, j0:j1] @ field_block
            )

    return sampled_power, coefficients



def trapezoid(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def phi_average(values, b_over_w0):
    """
    Frozen reference-profile weighting from the current manuscript:
        p(u) = 4u exp(-2u^2), u=b/w0.
    """
    order = np.argsort(b_over_w0)
    u = np.asarray(b_over_w0, float)[order]
    v = np.asarray(values)[order]

    density = 4.0 * u * np.exp(-2.0 * u**2)

    numerical_mass = trapezoid(density, u)

    if numerical_mass <= 0:
        raise RuntimeError("Impact-parameter quadrature has zero mass.")

    avg = trapezoid(v * density, u) / numerical_mass

    exact_mass = (
        math.exp(-2.0 * float(u[0]) ** 2)
        - math.exp(-2.0 * float(u[-1]) ** 2)
    )

    return float(avg), float(exact_mass)


def run_calculation(
    *,
    wavelength_um: float,
    radius_um: float,
    n_medium: float,
    n_sphere: complex,
    w0_um: float,
    eta_col: float,
    input_l: int,
    input_sigma: int,
    lmin: int,
    lmax: int,
    pmax: int,
    bmax: float,
    nb: int,
    nq_in: int,
    nq_out: int,
    nphi: int,
    input_tail_tolerance: float,
    input_qcap_fraction: float,
    block_size: int,
    verbose: bool = True,
):
    if w0_um <= 0:
        raise ValueError("w0 must be positive.")
    if radius_um <= 0 or wavelength_um <= 0:
        raise ValueError("radius and wavelength must be positive.")
    if n_medium <= 0:
        raise ValueError("n_medium must be positive.")
    if not (0.0 < eta_col < 1.0):
        raise ValueError("eta_col must lie between 0 and 1.")
    if pmax < 0:
        raise ValueError("pmax must be nonnegative.")
    if lmin > lmax:
        raise ValueError("lmin cannot exceed lmax.")
    if input_sigma not in (-1, 1):
        raise ValueError("input_sigma must be +/-1.")

    k = 2.0 * math.pi * n_medium / wavelength_um

    m_rel = n_sphere / n_medium
    x = k * radius_um

    mie_a, mie_b, qsca, mie_nmax = mie_coefficients(
        m_rel,
        x,
    )

    q_in_max, input_coverage = choose_input_qmax(
        k=k,
        w0=w0_um,
        l_input=input_l,
        tail_tolerance=input_tail_tolerance,
        qcap_fraction=input_qcap_fraction,
    )

    q_out_max = eta_col * k

    input_grid = make_grid(
        q_in_max,
        k,
        nq_in,
        nphi,
    )

    output_grid = make_grid(
        q_out_max,
        k,
        nq_out,
        nphi,
    )

    l_values = np.arange(lmin, lmax + 1, dtype=int)

    (
        min_mode_aperture,
        max_mode_aperture,
        max_mode_norm_error,
    ) = output_mode_diagnostics(
        output_grid=output_grid,
        w0=w0_um,
        pmax=pmax,
        l_values=l_values,
    )

    projection, pl_labels = build_projection(
        output_grid,
        w0_um,
        pmax,
        l_values,
    )

    input_spectrum = lg_angular_spectrum(
        input_grid.q,
        input_grid.phi,
        w0_um,
        0,
        input_l,
    )

    b_values = np.linspace(
        0.0,
        bmax,
        nb,
    )

    if verbose:
        theta_deg = math.degrees(math.asin(eta_col))
        paraxial_error = eta_col**2 / 4.0

        print()
        print("FIXED PHYSICAL COLLECTION CHANNEL")
        print("=" * 72)
        print(f"eta_col = q_col/k = {eta_col:.8f}")
        print(f"NA_col = {n_medium * eta_col:.8f}")
        print(f"theta_col = {theta_deg:.6f} deg")
        print(
            "Paraxial longitudinal-phase correction estimate "
            f"eta_col^2/4 = {paraxial_error:.6e}"
        )

        print()
        print("Incident angular spectrum")
        print("-" * 72)
        print(f"eta_in,max = {q_in_max / k:.8f}")
        print(f"input coverage = {input_coverage:.12f}")
        print(f"input angular directions = {len(input_grid.q)}")

        print()
        print("Collected output")
        print("-" * 72)
        print(f"output angular directions = {len(output_grid.q)}")
        print(
            "minimum requested mode norm inside aperture = "
            f"{min_mode_aperture:.8e}"
        )
        print(
            "maximum requested mode norm inside aperture = "
            f"{max_mode_aperture:.8e}"
        )
        print(
            "maximum diagonal mode-norm quadrature error = "
            f"{max_mode_norm_error:.6e}"
        )

        print()
        print("Mie parameters")
        print("-" * 72)
        print(f"size parameter x = {x:.8f}")
        print(f"Mie multipole cutoff = {mie_nmax}")
        print(f"plane-wave Qsca = {qsca:.8f}")

        print()
        print("Streaming Mie projection for all impact parameters.")

    sampled_power, coeff_by_spin = scatter_project_all_b(
        input_grid=input_grid,
        output_grid=output_grid,
        input_sigma=input_sigma,
        mie_a=mie_a,
        mie_b=mie_b,
        projection=projection,
        input_spectrum=input_spectrum,
        b_values=b_values,
        w0_um=w0_um,
        k=k,
        block_size=block_size,
    )

    # Full coefficient vector ordered by sigma, then p,l.
    coeffs = np.vstack(
        (coeff_by_spin[+1], coeff_by_spin[-1])
    )

    labels = []

    for sigma_out in (+1, -1):
        for p, l in pl_labels:
            labels.append((int(p), int(l), int(sigma_out)))

    labels = np.asarray(labels, dtype=int)

    p_label = labels[:, 0]
    l_label = labels[:, 1]
    s_label = labels[:, 2]

    power = np.abs(coeffs) ** 2

    pmax_values = np.arange(0, pmax + 1, dtype=int)

    shape = (pmax + 1, nb)

    retained_power = np.zeros(shape)
    capture_fraction = np.zeros(shape)

    conversion_fraction = np.zeros(shape)
    radial_fraction = np.zeros(shape)
    spinflip_fraction = np.zeros(shape)
    compensated_fraction = np.zeros(shape)
    jbreak_fraction = np.zeros(shape)

    raw_conversion = np.zeros(shape)
    raw_radial = np.zeros(shape)
    raw_spinflip = np.zeros(shape)
    raw_compensated = np.zeros(shape)
    raw_jbreak = np.zeros(shape)

    j_input = input_l + input_sigma

    mask_same = (
        (p_label == 0)
        & (l_label == input_l)
        & (s_label == input_sigma)
    )

    mask_radial = (
        (p_label > 0)
        & (l_label == input_l)
        & (s_label == input_sigma)
    )

    mask_spinflip = s_label != input_sigma

    mask_compensated = (
        mask_spinflip
        & ((l_label + s_label) == j_input)
    )

    mask_jbreak = (
        (l_label + s_label) != j_input
    )

    for P in pmax_values:
        include = p_label <= P

        retained = np.sum(
            power[include, :],
            axis=0,
        )

        if np.any(retained <= 0):
            raise RuntimeError("Retained modal power vanished.")

        retained_power[P, :] = retained

        capture_fraction[P, :] = (
            retained / sampled_power
        )

        raw_conversion[P, :] = np.sum(
            power[include & ~mask_same, :],
            axis=0,
        )

        raw_radial[P, :] = np.sum(
            power[include & mask_radial, :],
            axis=0,
        )

        raw_spinflip[P, :] = np.sum(
            power[include & mask_spinflip, :],
            axis=0,
        )

        raw_compensated[P, :] = np.sum(
            power[include & mask_compensated, :],
            axis=0,
        )

        raw_jbreak[P, :] = np.sum(
            power[include & mask_jbreak, :],
            axis=0,
        )

        conversion_fraction[P, :] = (
            raw_conversion[P, :] / retained
        )

        radial_fraction[P, :] = (
            raw_radial[P, :] / retained
        )

        spinflip_fraction[P, :] = (
            raw_spinflip[P, :] / retained
        )

        compensated_fraction[P, :] = (
            raw_compensated[P, :] / retained
        )

        jbreak_fraction[P, :] = (
            raw_jbreak[P, :] / retained
        )

    avg_sampled, phi_mass = phi_average(
        sampled_power,
        b_values,
    )

    phi_capture = np.zeros(pmax + 1)
    phi_conversion = np.zeros(pmax + 1)
    phi_radial = np.zeros(pmax + 1)
    phi_spinflip = np.zeros(pmax + 1)
    phi_comp = np.zeros(pmax + 1)
    phi_jbreak = np.zeros(pmax + 1)

    for P in pmax_values:
        avg_retained, _ = phi_average(
            retained_power[P, :],
            b_values,
        )

        avg_conversion, _ = phi_average(
            raw_conversion[P, :],
            b_values,
        )

        avg_radial, _ = phi_average(
            raw_radial[P, :],
            b_values,
        )

        avg_spinflip, _ = phi_average(
            raw_spinflip[P, :],
            b_values,
        )

        avg_comp, _ = phi_average(
            raw_compensated[P, :],
            b_values,
        )

        avg_jbreak, _ = phi_average(
            raw_jbreak[P, :],
            b_values,
        )

        phi_capture[P] = avg_retained / avg_sampled
        phi_conversion[P] = avg_conversion / avg_retained
        phi_radial[P] = avg_radial / avg_retained
        phi_spinflip[P] = avg_spinflip / avg_retained
        phi_comp[P] = avg_comp / avg_retained
        phi_jbreak[P] = avg_jbreak / avg_retained

    theta_col_deg = math.degrees(math.asin(eta_col))

    return Results(
        pmax_values=pmax_values,
        b_over_w0=b_values,
        sampled_power=sampled_power,
        retained_power=retained_power,
        capture_fraction=capture_fraction,
        conversion_fraction=conversion_fraction,
        radial_fraction=radial_fraction,
        spinflip_fraction=spinflip_fraction,
        compensated_so_fraction=compensated_fraction,
        jbreak_fraction=jbreak_fraction,
        phi_mass_covered=phi_mass,
        phi_capture_fraction=phi_capture,
        phi_conversion_fraction=phi_conversion,
        phi_radial_fraction=phi_radial,
        phi_spinflip_fraction=phi_spinflip,
        phi_compensated_so_fraction=phi_comp,
        phi_jbreak_fraction=phi_jbreak,
        onaxis_jbreak_fraction=jbreak_fraction[:, 0],
        qsca=qsca,
        mie_nmax=mie_nmax,
        eta_col=eta_col,
        na_col=n_medium * eta_col,
        theta_col_deg=theta_col_deg,
        paraxial_phase_error_fraction=eta_col**2 / 4.0,
        eta_in_max=q_in_max / k,
        input_coverage=input_coverage,
        min_output_mode_aperture_fraction=min_mode_aperture,
        max_output_mode_aperture_fraction=max_mode_aperture,
        max_output_mode_norm_quadrature_error=max_mode_norm_error,
    )



def print_report(result: Results, args):
    print()
    print("=" * 76)
    print("FIXED-COLLECTION RESULTS")
    print("=" * 76)

    print(
        f"eta_col={result.eta_col:.6f}, "
        f"NA_col={result.na_col:.6f}, "
        f"theta_col={result.theta_col_deg:.4f} deg"
    )

    print(
        "paraxial phase correction eta^2/4 = "
        f"{result.paraxial_phase_error_fraction:.6e}"
    )

    print(
        f"w0={args.w0_um:.6g} um, "
        f"a={args.radius_um:.6g} um, "
        f"w0/a={args.w0_um / args.radius_um:.6g}"
    )

    print(
        f"input eta_max={result.eta_in_max:.6f}, "
        f"input coverage={result.input_coverage:.12f}"
    )

    print(
        "output mode-norm quadrature error = "
        f"{result.max_output_mode_norm_quadrature_error:.6e}"
    )

    print(
        "Phi reference-weight mass covered = "
        f"{result.phi_mass_covered:.10f}"
    )

    print()
    print(
        "pmax   capture      radial       j-break      "
        "spin-flip    compensated-SO"
    )
    print("-" * 76)

    for P in result.pmax_values:
        print(
            f"{P:4d}  "
            f"{result.phi_capture_fraction[P]:11.6f}  "
            f"{result.phi_radial_fraction[P]:11.6f}  "
            f"{result.phi_jbreak_fraction[P]:11.6f}  "
            f"{result.phi_spinflip_fraction[P]:11.6f}  "
            f"{result.phi_compensated_so_fraction[P]:14.6f}"
        )

    P = int(result.pmax_values[-1])

    print()
    print("Final-basis diagnostics")
    print("-" * 76)

    print(
        "on-axis j_z-changing numerical floor = "
        f"{result.onaxis_jbreak_fraction[P]:.8e}"
    )

    if P > 0:
        for name, arr in [
            ("capture", result.phi_capture_fraction),
            ("j-break", result.phi_jbreak_fraction),
            ("spin-flip", result.phi_spinflip_fraction),
            ("compensated SO", result.phi_compensated_so_fraction),
        ]:
            delta = arr[P] - arr[P - 1]
            rel = abs(delta) / max(abs(arr[P]), 1e-300)

            print(
                f"{name:18s}: "
                f"last delta={delta:+.6e}, "
                f"relative={rel:.6e}"
            )

    print()
    print("Interpretation")
    print("-" * 76)
    print(
        "The physical collection aperture is fixed and does not depend "
        "on pmax or lmax."
    )
    print(
        "The incident angular spectrum is independently converged."
    )
    print(
        "The Gaussian Phi average remains only a frozen-reference-profile "
        "diagnostic."
    )


def write_p_csv(result: Results, path: Path):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "pmax",
                "phi_capture_fraction",
                "phi_conversion_fraction",
                "phi_radial_fraction",
                "phi_spinflip_fraction",
                "phi_compensated_so_fraction",
                "phi_jbreak_fraction",
                "onaxis_jbreak_fraction",
            ]
        )

        for P in result.pmax_values:
            writer.writerow(
                [
                    int(P),
                    f"{result.phi_capture_fraction[P]:.12e}",
                    f"{result.phi_conversion_fraction[P]:.12e}",
                    f"{result.phi_radial_fraction[P]:.12e}",
                    f"{result.phi_spinflip_fraction[P]:.12e}",
                    f"{result.phi_compensated_so_fraction[P]:.12e}",
                    f"{result.phi_jbreak_fraction[P]:.12e}",
                    f"{result.onaxis_jbreak_fraction[P]:.12e}",
                ]
            )


def write_b_csv(result: Results, path: Path):
    P = int(result.pmax_values[-1])

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "b_over_w0",
                "sampled_collected_scattered_power_arbitrary",
                "retained_modal_power_arbitrary",
                "capture_fraction",
                "conversion_fraction_retained",
                "radial_fraction_retained",
                "spinflip_fraction_retained",
                "compensated_so_fraction_retained",
                "jbreak_fraction_retained",
            ]
        )

        for i, u in enumerate(result.b_over_w0):
            writer.writerow(
                [
                    f"{u:.12e}",
                    f"{result.sampled_power[i]:.12e}",
                    f"{result.retained_power[P, i]:.12e}",
                    f"{result.capture_fraction[P, i]:.12e}",
                    f"{result.conversion_fraction[P, i]:.12e}",
                    f"{result.radial_fraction[P, i]:.12e}",
                    f"{result.spinflip_fraction[P, i]:.12e}",
                    f"{result.compensated_so_fraction[P, i]:.12e}",
                    f"{result.jbreak_fraction[P, i]:.12e}",
                ]
            )


def metadata_dict(result: Results, args):
    wavelength_medium_um = args.wavelength_um / args.n_medium

    zR_um = (
        math.pi * args.n_medium * args.w0_um**2
        / args.wavelength_um
    )

    gaussian_divergence_rad = (
        args.wavelength_um
        / (math.pi * args.n_medium * args.w0_um)
    )

    return {
        "calculation_scope": (
            "scattered-field fixed-collection Stage-1 diagnostic"
        ),
        "wavelength_vacuum_um": args.wavelength_um,
        "wavelength_medium_um": wavelength_medium_um,
        "radius_um": args.radius_um,
        "n_medium": args.n_medium,
        "n_sphere_real": args.n_sphere_real,
        "n_sphere_imag": args.n_sphere_imag,
        "w0_um": args.w0_um,
        "w0_over_a": args.w0_um / args.radius_um,
        "rayleigh_range_um": zR_um,
        "gaussian_divergence_rad": gaussian_divergence_rad,
        "gaussian_divergence_deg": math.degrees(
            gaussian_divergence_rad
        ),
        "input_p": 0,
        "input_l": args.input_l,
        "input_sigma": args.input_sigma,
        "lmin": args.lmin,
        "lmax": args.lmax,
        "pmax": args.pmax,
        "eta_col": result.eta_col,
        "NA_col": result.na_col,
        "theta_col_deg": result.theta_col_deg,
        "paraxial_phase_error_fraction_eta2_over4": (
            result.paraxial_phase_error_fraction
        ),
        "eta_in_max": result.eta_in_max,
        "input_coverage": result.input_coverage,
        "input_tail_tolerance": args.input_tail_tolerance,
        "bmax_over_w0": args.bmax,
        "nb": args.nb,
        "nq_in": args.nq_in,
        "nq_out": args.nq_out,
        "nphi": args.nphi,
        "phi_reference_mass_covered": result.phi_mass_covered,
        "mie_qsca": result.qsca,
        "mie_nmax": result.mie_nmax,
        "min_output_mode_aperture_fraction": (
            result.min_output_mode_aperture_fraction
        ),
        "max_output_mode_aperture_fraction": (
            result.max_output_mode_aperture_fraction
        ),
        "max_output_mode_norm_quadrature_error": (
            result.max_output_mode_norm_quadrature_error
        ),
        "final_phi_capture_fraction": float(
            result.phi_capture_fraction[-1]
        ),
        "final_phi_jbreak_fraction": float(
            result.phi_jbreak_fraction[-1]
        ),
        "final_phi_spinflip_fraction": float(
            result.phi_spinflip_fraction[-1]
        ),
        "final_phi_compensated_so_fraction": float(
            result.phi_compensated_so_fraction[-1]
        ),
        "final_phi_radial_fraction": float(
            result.phi_radial_fraction[-1]
        ),
        "final_onaxis_jbreak_fraction": float(
            result.onaxis_jbreak_fraction[-1]
        ),
    }


def save_metadata(result: Results, args, path: Path):
    path.write_text(
        json.dumps(
            metadata_dict(result, args),
            indent=2,
        ),
        encoding="utf-8",
    )


def make_plot(x, series, xlabel, ylabel, title, path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping plots.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for y, label in series:
        ax.plot(x, y, marker="o", label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    if len(series) > 1:
        ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_plots(result: Results, prefix: str):
    p = result.pmax_values

    make_plot(
        p,
        [(result.phi_capture_fraction, "capture")],
        r"$p_{\max}$",
        "Collected-field modal capture",
        "Basis capture at fixed physical collection aperture",
        f"{prefix}_capture_vs_pmax.png",
    )

    make_plot(
        p,
        [
            (result.phi_jbreak_fraction, r"$j_z$-changing"),
            (result.phi_spinflip_fraction, "spin flip"),
            (
                result.phi_compensated_so_fraction,
                "compensated SO",
            ),
        ],
        r"$p_{\max}$",
        "Retained-sector fraction",
        "Physics convergence at fixed collection aperture",
        f"{prefix}_physics_vs_pmax.png",
    )

    P = int(p[-1])

    make_plot(
        result.b_over_w0,
        [(result.capture_fraction[P], "capture")],
        r"$b/w_0$",
        "Collected-field modal capture",
        "Capture versus impact parameter",
        f"{prefix}_capture_vs_b.png",
    )



class TeeStream:
    """Write terminal output simultaneously to screen and a log file."""

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

    def isatty(self):
        return any(
            getattr(stream, "isatty", lambda: False)()
            for stream in self.streams
        )


def result_summary(result: Results):
    """Compact final diagnostics for one calculation."""
    return {
        "eta_col": float(result.eta_col),
        "NA_col": float(result.na_col),
        "theta_col_deg": float(result.theta_col_deg),
        "paraxial_phase_error_fraction": float(
            result.paraxial_phase_error_fraction
        ),
        "eta_in_max": float(result.eta_in_max),
        "input_coverage": float(result.input_coverage),
        "phi_mass_covered": float(result.phi_mass_covered),
        "mie_qsca": float(result.qsca),
        "mie_nmax": int(result.mie_nmax),
        "min_output_mode_aperture_fraction": float(
            result.min_output_mode_aperture_fraction
        ),
        "max_output_mode_aperture_fraction": float(
            result.max_output_mode_aperture_fraction
        ),
        "max_output_mode_norm_quadrature_error": float(
            result.max_output_mode_norm_quadrature_error
        ),
        "final_capture_fraction": float(
            result.phi_capture_fraction[-1]
        ),
        "final_conversion_fraction": float(
            result.phi_conversion_fraction[-1]
        ),
        "final_radial_fraction": float(
            result.phi_radial_fraction[-1]
        ),
        "final_spinflip_fraction": float(
            result.phi_spinflip_fraction[-1]
        ),
        "final_compensated_so_fraction": float(
            result.phi_compensated_so_fraction[-1]
        ),
        "final_jbreak_fraction": float(
            result.phi_jbreak_fraction[-1]
        ),
        "final_onaxis_jbreak_fraction": float(
            result.onaxis_jbreak_fraction[-1]
        ),
    }


def full_result_dict(result: Results):
    """
    Serialize every numerically relevant result.

    This makes the JSON report self-contained for later review.
    """
    return {
        "summary": result_summary(result),
        "pmax_values": result.pmax_values.astype(int).tolist(),
        "b_over_w0": result.b_over_w0.astype(float).tolist(),
        "sampled_power": result.sampled_power.astype(float).tolist(),
        "retained_power": result.retained_power.astype(float).tolist(),
        "capture_fraction": result.capture_fraction.astype(float).tolist(),
        "conversion_fraction": (
            result.conversion_fraction.astype(float).tolist()
        ),
        "radial_fraction": result.radial_fraction.astype(float).tolist(),
        "spinflip_fraction": (
            result.spinflip_fraction.astype(float).tolist()
        ),
        "compensated_so_fraction": (
            result.compensated_so_fraction.astype(float).tolist()
        ),
        "jbreak_fraction": result.jbreak_fraction.astype(float).tolist(),
        "phi_capture_fraction": (
            result.phi_capture_fraction.astype(float).tolist()
        ),
        "phi_conversion_fraction": (
            result.phi_conversion_fraction.astype(float).tolist()
        ),
        "phi_radial_fraction": (
            result.phi_radial_fraction.astype(float).tolist()
        ),
        "phi_spinflip_fraction": (
            result.phi_spinflip_fraction.astype(float).tolist()
        ),
        "phi_compensated_so_fraction": (
            result.phi_compensated_so_fraction.astype(float).tolist()
        ),
        "phi_jbreak_fraction": (
            result.phi_jbreak_fraction.astype(float).tolist()
        ),
        "onaxis_jbreak_fraction": (
            result.onaxis_jbreak_fraction.astype(float).tolist()
        ),
    }


def args_dict(args):
    """JSON-safe copy of all command-line parameters."""
    return {
        key: value
        for key, value in vars(args).items()
        if isinstance(
            value,
            (str, int, float, bool, type(None)),
        )
    }


def eta_tag(eta):
    """Filename-safe representation of eta."""
    return f"{eta:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def write_json(data, path):
    path = Path(path)
    path.write_text(
        json.dumps(data, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return path



def convergence_check(args, base: Results):
    print()
    print("=" * 76)
    print("FIXED-APERTURE ANGULAR-QUADRATURE CONVERGENCE")
    print("=" * 76)

    refined_nq_in = args.nq_in + 8
    refined_nq_out = args.nq_out + 8
    refined_nphi = args.nphi + 16

    refined = run_calculation(
        wavelength_um=args.wavelength_um,
        radius_um=args.radius_um,
        n_medium=args.n_medium,
        n_sphere=complex(
            args.n_sphere_real,
            args.n_sphere_imag,
        ),
        w0_um=args.w0_um,
        eta_col=args.eta_col,
        input_l=args.input_l,
        input_sigma=args.input_sigma,
        lmin=args.lmin,
        lmax=args.lmax,
        pmax=args.pmax,
        bmax=args.bmax,
        nb=args.nb,
        nq_in=refined_nq_in,
        nq_out=refined_nq_out,
        nphi=refined_nphi,
        input_tail_tolerance=args.input_tail_tolerance,
        input_qcap_fraction=args.input_qcap_fraction,
        block_size=args.block_size,
        verbose=False,
    )

    metric_specs = [
        (
            "capture",
            base.phi_capture_fraction[-1],
            refined.phi_capture_fraction[-1],
        ),
        (
            "radial",
            base.phi_radial_fraction[-1],
            refined.phi_radial_fraction[-1],
        ),
        (
            "j-break",
            base.phi_jbreak_fraction[-1],
            refined.phi_jbreak_fraction[-1],
        ),
        (
            "spin-flip",
            base.phi_spinflip_fraction[-1],
            refined.phi_spinflip_fraction[-1],
        ),
        (
            "compensated SO",
            base.phi_compensated_so_fraction[-1],
            refined.phi_compensated_so_fraction[-1],
        ),
        (
            "on-axis floor",
            base.onaxis_jbreak_fraction[-1],
            refined.onaxis_jbreak_fraction[-1],
        ),
    ]

    rows = []

    for name, v1, v2 in metric_specs:
        delta = float(v2 - v1)
        rel = abs(delta) / max(
            abs(float(v1)),
            abs(float(v2)),
            1e-300,
        )

        row = {
            "metric": name,
            "base": float(v1),
            "refined": float(v2),
            "absolute_delta": delta,
            "relative_change": float(rel),
        }

        rows.append(row)

        print(
            f"{name:18s}: "
            f"base={v1:.8e}, "
            f"refined={v2:.8e}, "
            f"relative={rel:.6e}"
        )

    csv_path = Path(
        f"{args.prefix}_quadrature_convergence.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    refined_p_path = Path(
        f"{args.prefix}_refined_p_convergence.csv"
    )
    refined_b_path = Path(
        f"{args.prefix}_refined_b_scan.csv"
    )

    write_p_csv(refined, refined_p_path)
    write_b_csv(refined, refined_b_path)

    report = {
        "base_quadrature": {
            "nq_in": int(args.nq_in),
            "nq_out": int(args.nq_out),
            "nphi": int(args.nphi),
        },
        "refined_quadrature": {
            "nq_in": int(refined_nq_in),
            "nq_out": int(refined_nq_out),
            "nphi": int(refined_nphi),
        },
        "metrics": rows,
        "refined_results": full_result_dict(refined),
    }

    json_path = write_json(
        report,
        f"{args.prefix}_quadrature_convergence.json",
    )

    print()
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {refined_p_path}")
    print(f"Saved: {refined_b_path}")

    return report



def parse_float_list(text: str):
    values = []

    for item in text.split(","):
        item = item.strip()

        if not item:
            continue

        value = float(item)

        if not (0.0 < value < 1.0):
            raise ValueError(
                "Every eta sweep value must lie between 0 and 1."
            )

        values.append(value)

    if not values:
        raise ValueError("eta sweep is empty.")

    return values


def eta_sweep(args):
    etas = parse_float_list(args.eta_sweep)

    rows = []
    full_runs = []

    print()
    print("=" * 76)
    print("FIXED-BASIS COLLECTION-APERTURE SENSITIVITY")
    print("=" * 76)

    for eta in etas:
        print()
        print(f"Running eta_col={eta:g}")

        result = run_calculation(
            wavelength_um=args.wavelength_um,
            radius_um=args.radius_um,
            n_medium=args.n_medium,
            n_sphere=complex(
                args.n_sphere_real,
                args.n_sphere_imag,
            ),
            w0_um=args.w0_um,
            eta_col=eta,
            input_l=args.input_l,
            input_sigma=args.input_sigma,
            lmin=args.lmin,
            lmax=args.lmax,
            pmax=args.pmax,
            bmax=args.bmax,
            nb=args.nb,
            nq_in=args.nq_in,
            nq_out=args.nq_out,
            nphi=args.nphi,
            input_tail_tolerance=args.input_tail_tolerance,
            input_qcap_fraction=args.input_qcap_fraction,
            block_size=args.block_size,
            verbose=False,
        )

        row = {
            "eta_col": float(eta),
            "NA_col": float(result.na_col),
            "theta_col_deg": float(result.theta_col_deg),
            "paraxial_phase_error_fraction": float(
                result.paraxial_phase_error_fraction
            ),
            "capture": float(result.phi_capture_fraction[-1]),
            "jbreak": float(result.phi_jbreak_fraction[-1]),
            "spinflip": float(result.phi_spinflip_fraction[-1]),
            "compensated_so": float(
                result.phi_compensated_so_fraction[-1]
            ),
            "radial": float(result.phi_radial_fraction[-1]),
            "onaxis_floor": float(
                result.onaxis_jbreak_fraction[-1]
            ),
            "mode_norm_quadrature_error": float(
                result.max_output_mode_norm_quadrature_error
            ),
        }

        rows.append(row)

        tag = eta_tag(eta)

        eta_p_path = Path(
            f"{args.prefix}_eta_{tag}_p_convergence.csv"
        )
        eta_b_path = Path(
            f"{args.prefix}_eta_{tag}_b_scan.csv"
        )

        write_p_csv(result, eta_p_path)
        write_b_csv(result, eta_b_path)

        full_runs.append(
            {
                "eta_col": float(eta),
                "summary": row,
                "results": full_result_dict(result),
                "saved_p_convergence": str(eta_p_path),
                "saved_b_scan": str(eta_b_path),
            }
        )

        print(
            f"theta={row['theta_col_deg']:.4f} deg, "
            f"capture={row['capture']:.6f}, "
            f"j-break={row['jbreak']:.6e}, "
            f"spin-flip={row['spinflip']:.6e}"
        )

        print(f"Saved: {eta_p_path}")
        print(f"Saved: {eta_b_path}")

    csv_path = Path(f"{args.prefix}_eta_sweep.csv")

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path = write_json(
        {
            "parameters": args_dict(args),
            "summary_rows": rows,
            "full_runs": full_runs,
        },
        f"{args.prefix}_eta_sweep.json",
    )

    if not args.no_plots:
        x = np.asarray([r["eta_col"] for r in rows])

        make_plot(
            x,
            [
                (
                    np.asarray([r["jbreak"] for r in rows]),
                    r"$j_z$-changing",
                ),
                (
                    np.asarray([r["spinflip"] for r in rows]),
                    "spin flip",
                ),
                (
                    np.asarray(
                        [r["compensated_so"] for r in rows]
                    ),
                    "compensated SO",
                ),
            ],
            r"$\eta_{\rm col}=q_{\rm col}/k$",
            "Retained-sector fraction",
            "Collection-aperture sensitivity",
            f"{args.prefix}_eta_sweep_physics.png",
        )

        make_plot(
            x,
            [
                (
                    np.asarray([r["capture"] for r in rows]),
                    "capture",
                )
            ],
            r"$\eta_{\rm col}=q_{\rm col}/k$",
            "Collected-field modal capture",
            "Capture versus collection aperture",
            f"{args.prefix}_eta_sweep_capture.png",
        )

    print()
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")

    return {
        "summary_rows": rows,
        "full_runs": full_runs,
        "csv_path": str(csv_path),
        "json_path": str(json_path),
    }


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Stage-1 spherical-Mie diagnostic with a fixed physical "
            "collection aperture."
        ),
    )

    p.add_argument(
        "--w0-um",
        type=float,
        required=True,
        help="Beam waist in micrometres.",
    )

    p.add_argument(
        "--eta-col",
        type=float,
        default=0.10,
        help="Fixed q_col/k = sin(theta_col).",
    )

    p.add_argument(
        "--eta-sweep",
        type=str,
        default=None,
        help=(
            "Optional comma-separated collection cutoffs, "
            "for example 0.075,0.10,0.15."
        ),
    )

    p.add_argument("--wavelength-um", type=float, default=0.633)
    p.add_argument("--radius-um", type=float, default=2.0)
    p.add_argument("--n-medium", type=float, default=1.0)
    p.add_argument("--n-sphere-real", type=float, default=1.33)
    p.add_argument("--n-sphere-imag", type=float, default=0.0)

    p.add_argument("--input-l", type=int, default=0)
    p.add_argument(
        "--input-sigma",
        type=int,
        choices=(-1, 1),
        default=1,
    )

    p.add_argument("--lmin", type=int, default=-20)
    p.add_argument("--lmax", type=int, default=20)
    p.add_argument("--pmax", type=int, default=20)

    p.add_argument(
        "--bmax",
        type=float,
        default=2.0,
        help="Maximum b/w0.",
    )

    p.add_argument("--nb", type=int, default=21)

    p.add_argument(
        "--nq-in",
        type=int,
        default=48,
        help="Incident angular-spectrum radial quadrature nodes.",
    )

    p.add_argument(
        "--nq-out",
        type=int,
        default=64,
        help="Collected-output radial quadrature nodes.",
    )

    p.add_argument(
        "--nphi",
        type=int,
        default=96,
    )

    p.add_argument(
        "--input-tail-tolerance",
        type=float,
        default=1e-10,
        help="Target omitted incident angular-spectrum power.",
    )

    p.add_argument(
        "--input-qcap-fraction",
        type=float,
        default=0.75,
        help="Safety cap q_in,max/k.",
    )

    p.add_argument(
        "--block-size",
        type=int,
        default=64,
    )

    p.add_argument(
        "--prefix",
        default="sphere_glmt_fixed",
    )

    p.add_argument(
        "--check-convergence",
        action="store_true",
    )

    p.add_argument(
        "--no-plots",
        action="store_true",
    )

    return p.parse_args()


def execute_run(args):
    generated_files = []

    result = run_calculation(
        wavelength_um=args.wavelength_um,
        radius_um=args.radius_um,
        n_medium=args.n_medium,
        n_sphere=complex(
            args.n_sphere_real,
            args.n_sphere_imag,
        ),
        w0_um=args.w0_um,
        eta_col=args.eta_col,
        input_l=args.input_l,
        input_sigma=args.input_sigma,
        lmin=args.lmin,
        lmax=args.lmax,
        pmax=args.pmax,
        bmax=args.bmax,
        nb=args.nb,
        nq_in=args.nq_in,
        nq_out=args.nq_out,
        nphi=args.nphi,
        input_tail_tolerance=args.input_tail_tolerance,
        input_qcap_fraction=args.input_qcap_fraction,
        block_size=args.block_size,
        verbose=True,
    )

    print_report(result, args)

    p_csv = Path(f"{args.prefix}_p_convergence.csv")
    b_csv = Path(f"{args.prefix}_b_scan.csv")
    metadata_path = Path(f"{args.prefix}_metadata.json")

    write_p_csv(result, p_csv)
    write_b_csv(result, b_csv)
    save_metadata(result, args, metadata_path)

    generated_files.extend(
        [
            str(p_csv),
            str(b_csv),
            str(metadata_path),
        ]
    )

    print()
    print(f"Saved: {p_csv}")
    print(f"Saved: {b_csv}")
    print(f"Saved: {metadata_path}")

    if not args.no_plots:
        save_plots(result, args.prefix)

        plot_files = [
            f"{args.prefix}_capture_vs_pmax.png",
            f"{args.prefix}_physics_vs_pmax.png",
            f"{args.prefix}_capture_vs_b.png",
        ]

        generated_files.extend(plot_files)

        for path in plot_files:
            print(f"Saved: {path}")

    convergence_report = None

    if args.check_convergence:
        convergence_report = convergence_check(
            args,
            result,
        )

        generated_files.extend(
            [
                f"{args.prefix}_quadrature_convergence.csv",
                f"{args.prefix}_quadrature_convergence.json",
                f"{args.prefix}_refined_p_convergence.csv",
                f"{args.prefix}_refined_b_scan.csv",
            ]
        )

    eta_report = None

    if args.eta_sweep:
        eta_report = eta_sweep(args)

        generated_files.extend(
            [
                f"{args.prefix}_eta_sweep.csv",
                f"{args.prefix}_eta_sweep.json",
            ]
        )

        if not args.no_plots:
            generated_files.extend(
                [
                    f"{args.prefix}_eta_sweep_physics.png",
                    f"{args.prefix}_eta_sweep_capture.png",
                ]
            )

        for run in eta_report["full_runs"]:
            generated_files.append(
                run["saved_p_convergence"]
            )
            generated_files.append(
                run["saved_b_scan"]
            )

    complete_report = {
        "script_version": 3,
        "calculation_scope": (
            "fixed-collection scattered-field Stage-1 diagnostic"
        ),
        "parameters": args_dict(args),
        "base_results": full_result_dict(result),
        "quadrature_convergence": convergence_report,
        "eta_sweep": eta_report,
        "generated_files": generated_files,
    }

    report_path = write_json(
        complete_report,
        f"{args.prefix}_FULL_RESULTS.json",
    )

    generated_files.append(str(report_path))

    manifest_path = write_json(
        {
            "script_version": 3,
            "parameters": args_dict(args),
            "all_generated_files": generated_files,
            "primary_review_file": str(report_path),
            "console_log": f"{args.prefix}_console.log",
        },
        f"{args.prefix}_manifest.json",
    )

    generated_files.append(str(manifest_path))

    print()
    print("=" * 76)
    print("COMPLETE NUMERICAL ARCHIVE")
    print("=" * 76)
    print(f"Primary review file: {report_path}")
    print(f"Manifest: {manifest_path}")
    print(
        "Every numerical result needed for review is saved."
    )

    return report_path, manifest_path


def main():
    args = parse_args()

    log_path = Path(f"{args.prefix}_console.log")

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
                "Complete console transcript is being saved to "
                f"{log_path}"
            )
            execute_run(args)

        except Exception:
            print()
            print("=" * 76)
            print("RUN FAILED")
            print("=" * 76)
            traceback.print_exc()
            raise

        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    print(f"Console transcript saved: {log_path}")


if __name__ == "__main__":
    main()
