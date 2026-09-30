#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.sparse import csr_matrix


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
    "sphere_glmt_fixed_collection_v3_stage4b",
)

flux = load_module(
    "sphere_glmt_stage3_flux.py",
    "sphere_glmt_stage3_flux_stage4b",
)

stage3b = load_module(
    "sphere_glmt_stage3b_oam.py",
    "sphere_glmt_stage3b_oam_stage4b",
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
            (
                str,
                int,
                float,
                bool,
                type(None),
            ),
        )
    }


def parse_float_list(text):
    values = []

    for item in text.split(","):
        item = item.strip()

        if item:
            values.append(float(item))

    if not values:
        raise ValueError(
            "Output tau list is empty."
        )

    values = sorted(
        set(values)
    )

    if values[0] < 0:
        raise ValueError(
            "Output tau values must be nonnegative."
        )

    return values


def make_cartesian_disk(
    qmax,
    nside,
    k,
):
    if nside < 5:
        raise ValueError(
            "nside must be at least 5."
        )

    axis = np.linspace(
        -qmax,
        qmax,
        nside,
    )

    dq = float(
        axis[1] - axis[0]
    )

    qx2d, qy2d = np.meshgrid(
        axis,
        axis,
        indexing="ij",
    )

    mask = (
        qx2d**2 + qy2d**2
        <= qmax**2 + 1e-15
    )

    ix, iy = np.nonzero(mask)

    qx = qx2d[mask]
    qy = qy2d[mask]

    q = np.sqrt(
        qx**2 + qy**2
    )

    eta = q / k

    cos_theta = np.sqrt(
        np.maximum(
            0.0,
            1.0 - eta**2,
        )
    )

    khat = np.column_stack(
        (
            qx / k,
            qy / k,
            cos_theta,
        )
    )

    index_map = -np.ones(
        (nside, nside),
        dtype=np.int32,
    )

    index_map[ix, iy] = np.arange(
        len(ix),
        dtype=np.int32,
    )

    return {
        "axis": axis,
        "dq": dq,
        "cell_area": dq**2,
        "mask": mask,
        "ix": ix.astype(
            np.int16
        ),
        "iy": iy.astype(
            np.int16
        ),
        "index_map": index_map,
        "qx": qx,
        "qy": qy,
        "q": q,
        "eta": eta,
        "cos_theta": cos_theta,
        "khat": khat,
        "nside": int(nside),
        "npoints": int(len(q)),
        "qmax": float(qmax),
    }


def forward_amplitude_beta(
    mie_a,
    mie_b,
):
    s1, s2 = base.mie_amplitudes(
        np.array([1.0]),
        mie_a,
        mie_b,
    )

    sf = 0.5 * (
        s1[0] + s2[0]
    )

    beta = (
        float(sf.imag)
        / (
            2.0
            * float(sf.real)
        )
    )

    return sf, beta


def precompute_pair_spin_matrices(
    *,
    grid,
    mie_a,
    mie_b,
    k,
    sigma_s,
):
    khat = grid["khat"]
    cos_theta = grid["cos_theta"]
    cell_area = grid["cell_area"]

    n = grid["npoints"]

    e_in = {
        +1: stage3b.local_circular(
            khat,
            +1,
        ),
        -1: stage3b.local_circular(
            khat,
            -1,
        ),
    }

    e_out = e_in

    matrices = np.empty(
        (
            n,
            n,
            2,
            2,
        ),
        dtype=np.complex64,
    )

    sigmas = (+1, -1)

    for ii, sigma_i in enumerate(
        sigmas
    ):
        scat = flux.scattering_vectors(
            khat,
            khat,
            e_in[sigma_i],
            mie_a,
            mie_b,
        )

        for oo, sigma_o in enumerate(
            sigmas
        ):
            projected = np.einsum(
                "ok,oik->oi",
                e_out[sigma_o].conj(),
                scat,
            )

            denom = (
                k**2
                * np.sqrt(
                    cos_theta[:, None]
                    * cos_theta[None, :]
                )
            )

            matrices[
                :,
                :,
                oo,
                ii,
            ] = (
                np.sqrt(
                    cell_area / sigma_s
                )
                * projected
                / denom
            ).astype(
                np.complex64
            )

        del scat

    # Flatten pair metadata in output-major order.
    output_idx = np.repeat(
        np.arange(
            n,
            dtype=np.int32,
        ),
        n,
    )

    input_idx = np.tile(
        np.arange(
            n,
            dtype=np.int32,
        ),
        n,
    )

    ix = grid["ix"].astype(
        np.int32
    )

    iy = grid["iy"].astype(
        np.int32
    )

    dx = (
        ix[output_idx]
        - ix[input_idx]
    )

    dy = (
        iy[output_idx]
        - iy[input_idx]
    )

    shift_side = (
        2 * grid["nside"] - 1
    )

    qid = (
        (dx + grid["nside"] - 1)
        * shift_side
        + (
            dy
            + grid["nside"]
            - 1
        )
    ).astype(
        np.int32
    )

    flat_matrix = matrices.reshape(
        n * n,
        2,
        2,
    )

    order = np.argsort(
        qid,
        kind="stable",
    )

    qid_sorted = qid[order]

    output_sorted = output_idx[
        order
    ]

    input_sorted = input_idx[
        order
    ]

    matrix_sorted = flat_matrix[
        order
    ]

    nqids = shift_side**2

    counts = np.bincount(
        qid_sorted,
        minlength=nqids,
    )

    starts = np.zeros(
        nqids + 1,
        dtype=np.int64,
    )

    starts[1:] = np.cumsum(
        counts,
        dtype=np.int64,
    )

    # Precompute A^dagger A for fast Q-probability evaluation.
    bmat = np.einsum(
        "eai,eaj->eij",
        matrix_sorted.conj(),
        matrix_sorted,
        optimize=True,
    )

    B00 = csr_matrix(
        (
            bmat[:, 0, 0].real,
            (
                qid_sorted,
                input_sorted,
            ),
        ),
        shape=(nqids, n),
    )

    B11 = csr_matrix(
        (
            bmat[:, 1, 1].real,
            (
                qid_sorted,
                input_sorted,
            ),
        ),
        shape=(nqids, n),
    )

    B01 = csr_matrix(
        (
            bmat[:, 0, 1],
            (
                qid_sorted,
                input_sorted,
            ),
        ),
        shape=(nqids, n),
    )

    return {
        "qid": qid_sorted,
        "output_idx": output_sorted,
        "input_idx": input_sorted,
        "matrix": matrix_sorted,
        "starts": starts,
        "counts": counts,
        "nqids": int(nqids),
        "shift_side": int(
            shift_side
        ),
        "B00": B00,
        "B11": B11,
        "B01": B01,
    }


def initial_lg_state(
    *,
    grid,
    w0_um,
    p,
    l,
    sigma,
):
    q = grid["q"]
    phi = np.arctan2(
        grid["qy"],
        grid["qx"],
    )

    U = base.lg_angular_spectrum(
        q,
        phi,
        w0_um,
        p,
        l,
    )

    psi_flux = (
        np.sqrt(
            grid["cos_theta"]
        )
        * U
    )

    state = np.zeros(
        (
            grid["npoints"],
            2,
        ),
        dtype=np.complex128,
    )

    spin_index = (
        0 if sigma == +1 else 1
    )

    state[
        :,
        spin_index,
    ] = (
        math.sqrt(
            grid["cell_area"]
        )
        * psi_flux
    )

    norm = float(
        np.sum(
            np.abs(state) ** 2
        )
    )

    state /= math.sqrt(norm)

    return state, norm


def retained_q_probabilities(
    state,
    jumps,
):
    rho00 = (
        np.abs(state[:, 0]) ** 2
    ).real

    rho11 = (
        np.abs(state[:, 1]) ** 2
    ).real

    coherence = (
        state[:, 0].conj()
        * state[:, 1]
    )

    qprob = (
        jumps["B00"].dot(rho00)
        + jumps["B11"].dot(rho11)
        + 2.0
        * np.real(
            jumps["B01"].dot(
                coherence
            )
        )
    )

    # Roundoff can create tiny negative entries.
    qprob = np.maximum(
        np.asarray(
            qprob
        ).ravel(),
        0.0,
    )

    return qprob


def apply_retained_jump(
    state,
    jumps,
    qid,
    *,
    interaction_picture=False,
    phase_diag=None,
    absolute_tau=None,
):
    start = int(
        jumps["starts"][qid]
    )

    stop = int(
        jumps["starts"][
            qid + 1
        ]
    )

    if stop <= start:
        raise RuntimeError(
            "Selected empty Q channel."
        )

    inp = jumps[
        "input_idx"
    ][start:stop]

    out = jumps[
        "output_idx"
    ][start:stop]

    mats = jumps[
        "matrix"
    ][start:stop]

    if interaction_picture:
        if phase_diag is None:
            raise ValueError(
                "phase_diag is required "
                "in interaction picture."
            )

        if absolute_tau is None:
            raise ValueError(
                "absolute_tau is required "
                "in interaction picture."
            )

        pair_phase = np.exp(
            1j
            * (
                phase_diag[out]
                - phase_diag[inp]
            )
            * absolute_tau
        )

        mats_eff = (
            mats
            * pair_phase[
                :,
                None,
                None,
            ]
        )
    else:
        mats_eff = mats

    y = np.einsum(
        "eab,eb->ea",
        mats_eff,
        state[inp],
        optimize=True,
    )

    result = np.zeros_like(
        state
    )

    # Fixed Q is a one-to-one grid shift.
    result[out] = y

    norm = float(
        np.sum(
            np.abs(result) ** 2
        )
    )

    if norm <= 0:
        raise RuntimeError(
            "Retained jump has zero norm."
        )

    result /= math.sqrt(
        norm
    )

    return result, norm


def solve_jump_time(
    state,
    rate_diag,
    rng,
):
    populations = np.sum(
        np.abs(state) ** 2,
        axis=1,
    ).real

    u = float(
        rng.random()
    )

    u = max(
        u,
        1e-300,
    )

    rmin = float(
        np.min(rate_diag)
    )

    rmax = float(
        np.max(rate_diag)
    )

    lo = (
        -math.log(u)
        / rmax
    )

    hi = (
        -math.log(u)
        / rmin
    )

    def survival(t):
        return float(
            np.sum(
                populations
                * np.exp(
                    -rate_diag * t
                )
            )
        )

    # Bisection is robust because the rate interval is narrow.
    for _ in range(45):
        mid = 0.5 * (
            lo + hi
        )

        if survival(mid) > u:
            lo = mid
        else:
            hi = mid

    return 0.5 * (
        lo + hi
    )


def evolve_no_jump(
    state,
    dtau,
    rate_diag,
    phase_diag,
    normalize,
    interaction_picture=False,
):
    factor = np.exp(
        -0.5
        * rate_diag
        * dtau
    )

    if not interaction_picture:
        factor = (
            factor
            * np.exp(
                -1j
                * phase_diag
                * dtau
            )
        )

    result = (
        state
        * factor[:, None]
    )

    norm = float(
        np.sum(
            np.abs(result) ** 2
        )
    )

    if normalize:
        if norm <= 0:
            raise RuntimeError(
                "No-jump state vanished."
            )

        result /= math.sqrt(
            norm
        )

    return result, norm


def make_measurement_grid(
    qmax,
    nqr,
    nphi,
):
    from numpy.polynomial.legendre import leggauss

    x, wx = leggauss(
        nqr
    )

    qr = (
        0.5
        * qmax
        * (
            x + 1.0
        )
    )

    dq_weights = (
        0.5
        * qmax
        * wx
    )

    radial_weights = (
        qr
        * dq_weights
    )

    phi = np.linspace(
        0.0,
        2.0 * math.pi,
        nphi,
        endpoint=False,
    )

    return {
        "q": qr,
        "radial_weights": (
            radial_weights
        ),
        "phi": phi,
        "nqr": int(nqr),
        "nphi": int(nphi),
    }


def state_to_oam(
    *,
    state,
    grid,
    measurement,
    lmin,
    lmax,
    interpolation_order,
):
    nside = grid["nside"]

    square = np.zeros(
        (
            2,
            nside,
            nside,
        ),
        dtype=np.complex128,
    )

    psi_values = (
        state
        / math.sqrt(
            grid["cell_area"]
        )
    )

    square[
        :,
        grid["ix"],
        grid["iy"],
    ] = psi_values.T

    qr = measurement["q"]
    phi = measurement["phi"]

    qx = (
        qr[:, None]
        * np.cos(
            phi[None, :]
        )
    )

    qy = (
        qr[:, None]
        * np.sin(
            phi[None, :]
        )
    )

    axis0 = grid["axis"][0]
    dq = grid["dq"]

    coord_x = (
        qx - axis0
    ) / dq

    coord_y = (
        qy - axis0
    ) / dq

    coords = np.vstack(
        (
            coord_x.ravel(),
            coord_y.ravel(),
        )
    )

    fft_l = np.fft.fftfreq(
        measurement["nphi"],
        d=1.0
        / measurement["nphi"],
    ).astype(int)

    selected = (
        (fft_l >= lmin)
        & (fft_l <= lmax)
    )

    selected_l_unsorted = fft_l[
        selected
    ]

    sort_order = np.argsort(
        selected_l_unsorted
    )

    selected_l = (
        selected_l_unsorted[
            sort_order
        ]
    )

    power_spin = np.zeros(
        (
            2,
            len(selected_l),
        ),
        dtype=float,
    )

    full_power = 0.0

    for spin in range(2):
        real = map_coordinates(
            square[spin].real,
            coords,
            order=interpolation_order,
            mode="constant",
            cval=0.0,
        ).reshape(
            measurement["nqr"],
            measurement["nphi"],
        )

        imag = map_coordinates(
            square[spin].imag,
            coords,
            order=interpolation_order,
            mode="constant",
            cval=0.0,
        ).reshape(
            measurement["nqr"],
            measurement["nphi"],
        )

        field = (
            real + 1j * imag
        )

        coeff = (
            math.sqrt(
                2.0 * math.pi
            )
            / measurement["nphi"]
            * np.fft.fft(
                field,
                axis=1,
            )
        )

        spectral = (
            np.abs(coeff) ** 2
        )

        radial = (
            measurement[
                "radial_weights"
            ][:, None]
        )

        all_l_power = np.sum(
            radial
            * spectral,
            axis=0,
        )

        full_power += float(
            np.sum(
                all_l_power
            )
        )

        power_spin[spin] = (
            all_l_power[
                selected
            ][
                sort_order
            ]
        )

    return {
        "l_values": (
            selected_l
        ),
        "power_spin": (
            power_spin
        ),
        "measured_total": float(
            full_power
        ),
    }


def weighted_choice(
    weights,
    rng,
):
    total = float(
        np.sum(weights)
    )

    if total <= 0:
        raise RuntimeError(
            "Cannot sample zero weights."
        )

    target = (
        float(
            rng.random()
        )
        * total
    )

    cumulative = np.cumsum(
        weights
    )

    return int(
        np.searchsorted(
            cumulative,
            target,
            side="right",
        )
    )


def propagate_segment(
    *,
    state,
    delta_tau,
    tau_start,
    jumps,
    rate_diag,
    phase_diag,
    rng,
    interaction_picture,
):
    current = state.copy()
    weight = 1.0
    elapsed = 0.0
    njumps = 0

    while elapsed < delta_tau:
        next_jump = solve_jump_time(
            current,
            rate_diag,
            rng,
        )

        remaining = (
            delta_tau - elapsed
        )

        if next_jump >= remaining:
            end_state, _ = (
                evolve_no_jump(
                    current,
                    remaining,
                    rate_diag,
                    phase_diag,
                    normalize=True,
                    interaction_picture=(
                        interaction_picture
                    ),
                )
            )

            return (
                end_state,
                weight,
                njumps,
            )

        pre, _ = evolve_no_jump(
            current,
            next_jump,
            rate_diag,
            phase_diag,
            normalize=True,
            interaction_picture=(
                interaction_picture
            ),
        )

        total_rate = float(
            np.sum(
                np.sum(
                    np.abs(pre) ** 2,
                    axis=1,
                )
                * rate_diag
            )
        )

        qprob = (
            retained_q_probabilities(
                pre,
                jumps,
            )
        )

        retained_rate = float(
            np.sum(qprob)
        )

        if retained_rate <= 0:
            return (
                pre,
                0.0,
                njumps,
            )

        p_ret = (
            retained_rate
            / total_rate
        )

        weight *= p_ret

        qid = weighted_choice(
            qprob,
            rng,
        )

        jump_absolute_tau = (
            tau_start
            + elapsed
            + next_jump
        )

        current, _ = (
            apply_retained_jump(
                pre,
                jumps,
                qid,
                interaction_picture=(
                    interaction_picture
                ),
                phase_diag=phase_diag,
                absolute_tau=(
                    jump_absolute_tau
                ),
            )
        )

        elapsed += next_jump
        njumps += 1

    return (
        current,
        weight,
        njumps,
    )


def measure_weighted_ensemble(
    *,
    states,
    weights,
    grid,
    measurement,
    lmin,
    lmax,
    interpolation_order,
):
    l_values = np.arange(
        lmin,
        lmax + 1,
        dtype=int,
    )

    power = np.zeros(
        (
            2,
            len(l_values),
        ),
        dtype=float,
    )

    capture = 0.0

    for state, weight in zip(
        states,
        weights,
    ):
        measured = state_to_oam(
            state=state,
            grid=grid,
            measurement=measurement,
            lmin=lmin,
            lmax=lmax,
            interpolation_order=(
                interpolation_order
            ),
        )

        if not np.array_equal(
            measured["l_values"],
            l_values,
        ):
            raise RuntimeError(
                "Measurement OAM ordering mismatch."
            )

        power += (
            weight
            * measured[
                "power_spin"
            ]
        )

        capture += (
            weight
            * measured[
                "measured_total"
            ]
        )

    norm_weight = float(
        np.sum(weights)
    )

    if norm_weight > 0:
        power /= norm_weight
        capture /= norm_weight

    return {
        "l_values": l_values,
        "conditional_power": power,
        "conditional_measurement_capture": float(
            capture
        ),
    }


def systematic_resample(
    weights,
    rng,
):
    weights = np.asarray(
        weights,
        dtype=float,
    )

    total = float(
        np.sum(weights)
    )

    if total <= 0:
        raise RuntimeError(
            "All SMC weights vanished."
        )

    normalized = (
        weights / total
    )

    n = len(weights)

    positions = (
        float(rng.random())
        + np.arange(n)
    ) / n

    cumulative = np.cumsum(
        normalized
    )

    return np.searchsorted(
        cumulative,
        positions,
        side="right",
    )


def simulate(
    *,
    args,
    grid,
    jumps,
    initial_state,
    rate_diag,
    phase_diag,
    measurement,
    output_tau,
):
    rng = np.random.default_rng(
        args.seed
    )

    states = np.repeat(
        initial_state[
            None,
            :,
            :,
        ],
        args.ntraj,
        axis=0,
    )

    output_tau = np.asarray(
        output_tau,
        dtype=float,
    )

    l_values = np.arange(
        args.lmin_measure,
        args.lmax_measure + 1,
        dtype=int,
    )

    nt = len(output_tau)
    nl = len(l_values)

    conditional_power = np.zeros(
        (
            nt,
            2,
            nl,
        ),
        dtype=float,
    )

    unconditional_power = np.zeros_like(
        conditional_power
    )

    transmittance = np.zeros(
        nt,
        dtype=float,
    )

    measurement_capture = np.zeros(
        nt,
        dtype=float,
    )

    effective_sample_size = np.zeros(
        nt,
        dtype=float,
    )

    mean_segment_jumps = np.zeros(
        nt,
        dtype=float,
    )

    max_segment_jumps = np.zeros(
        nt,
        dtype=int,
    )

    max_tau = float(
        output_tau[-1]
    )

    if args.resample_step <= 0:
        raise ValueError(
            "resample-step must be positive."
        )

    regular = np.arange(
        args.resample_step,
        max_tau
        + 0.5
        * args.resample_step,
        args.resample_step,
    )

    checkpoints = np.unique(
        np.concatenate(
            (
                output_tau,
                regular[
                    regular <= max_tau + 1e-12
                ],
            )
        )
    )

    checkpoints.sort()

    cumulative_T = 1.0
    previous_tau = 0.0

    output_lookup = {
        round(
            float(tau),
            12,
        ): index
        for index, tau in enumerate(
            output_tau
        )
    }

    # Initial output.
    if (
        abs(output_tau[0]) < 1e-15
    ):
        measured0 = (
            measure_weighted_ensemble(
                states=states,
                weights=np.ones(
                    args.ntraj
                ),
                grid=grid,
                measurement=measurement,
                lmin=args.lmin_measure,
                lmax=args.lmax_measure,
                interpolation_order=(
                    args.measurement_order
                ),
            )
        )

        conditional_power[0] = (
            measured0[
                "conditional_power"
            ]
        )

        unconditional_power[0] = (
            conditional_power[0]
        )

        transmittance[0] = 1.0

        measurement_capture[0] = (
            measured0[
                "conditional_measurement_capture"
            ]
        )

        effective_sample_size[0] = float(
            args.ntraj
        )

    for checkpoint in checkpoints:
        checkpoint = float(
            checkpoint
        )

        if checkpoint <= 1e-15:
            continue

        delta_tau = (
            checkpoint
            - previous_tau
        )

        if delta_tau < -1e-12:
            raise RuntimeError(
                "Checkpoint ordering failure."
            )

        end_states = np.empty_like(
            states
        )

        segment_weights = np.zeros(
            args.ntraj,
            dtype=float,
        )

        jump_counts = np.zeros(
            args.ntraj,
            dtype=int,
        )

        for ntraj_index in range(
            args.ntraj
        ):
            (
                end_state,
                seg_weight,
                njumps,
            ) = propagate_segment(
                state=states[
                    ntraj_index
                ],
                delta_tau=delta_tau,
                tau_start=previous_tau,
                jumps=jumps,
                rate_diag=rate_diag,
                phase_diag=phase_diag,
                rng=rng,
                interaction_picture=(
                    not args.schrodinger_picture
                ),
            )

            end_states[
                ntraj_index
            ] = end_state

            segment_weights[
                ntraj_index
            ] = seg_weight

            jump_counts[
                ntraj_index
            ] = njumps

        mean_weight = float(
            np.mean(
                segment_weights
            )
        )

        cumulative_T *= (
            mean_weight
        )

        s1 = float(
            np.sum(
                segment_weights
            )
        )

        s2 = float(
            np.sum(
                segment_weights**2
            )
        )

        local_ess = (
            s1**2 / s2
            if s2 > 0
            else 0.0
        )

        key = round(
            checkpoint,
            12,
        )

        if key in output_lookup:
            it = output_lookup[
                key
            ]

            measured = (
                measure_weighted_ensemble(
                    states=end_states,
                    weights=segment_weights,
                    grid=grid,
                    measurement=measurement,
                    lmin=args.lmin_measure,
                    lmax=args.lmax_measure,
                    interpolation_order=(
                        args.measurement_order
                    ),
                )
            )

            conditional_power[it] = (
                measured[
                    "conditional_power"
                ]
            )

            unconditional_power[it] = (
                cumulative_T
                * conditional_power[it]
            )

            transmittance[it] = (
                cumulative_T
            )

            measurement_capture[it] = (
                measured[
                    "conditional_measurement_capture"
                ]
            )

            effective_sample_size[it] = float(
                local_ess
            )

            mean_segment_jumps[it] = float(
                np.mean(
                    jump_counts
                )
            )

            max_segment_jumps[it] = int(
                np.max(
                    jump_counts
                )
            )

            print(
                f"tau={checkpoint:g}: "
                f"T={cumulative_T:.6e}, "
                f"local ESS={local_ess:.1f}"
            )

        # Resample at every checkpoint except final propagation endpoint.
        if checkpoint < max_tau - 1e-12:
            indices = systematic_resample(
                segment_weights,
                rng,
            )

            states = (
                end_states[
                    indices
                ].copy()
            )

        else:
            states = end_states

        previous_tau = (
            checkpoint
        )

    return {
        "l_values": l_values,
        "weighted_power": (
            unconditional_power
        ),
        "conditional_power": (
            conditional_power
        ),
        "transmittance": (
            transmittance
        ),
        "measurement_capture": (
            measurement_capture
        ),
        "effective_sample_size": (
            effective_sample_size
        ),
        "mean_segment_jumps": (
            mean_segment_jumps
        ),
        "max_segment_jumps": (
            max_segment_jumps
        ),
    }



def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
        description=(
            "Stage 4B radial-resolved "
            "multiple-scattering trajectories."
        ),
    )

    p.add_argument(
        "--w0-um",
        type=float,
        required=True,
    )

    p.add_argument(
        "--phi",
        type=float,
        required=True,
        help=(
            "Monodisperse particle volume fraction."
        ),
    )

    p.add_argument(
        "--eta-col",
        type=float,
        default=0.10,
    )

    p.add_argument(
        "--input-p",
        type=int,
        default=0,
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
        "--output-tau",
        type=str,
        default="0,1,5,15",
    )

    p.add_argument(
        "--nside",
        type=int,
        default=25,
    )

    p.add_argument(
        "--ntraj",
        type=int,
        default=2000,
    )

    p.add_argument(
        "--resample-step",
        type=float,
        default=1.0,
        help=(
            "Maximum tau interval between SMC resampling checkpoints."
        ),
    )

    p.add_argument(
        "--seed",
        type=int,
        default=12345,
    )

    p.add_argument(
        "--lmin-measure",
        type=int,
        default=-12,
    )

    p.add_argument(
        "--lmax-measure",
        type=int,
        default=12,
    )

    p.add_argument(
        "--nqr-measure",
        type=int,
        default=36,
    )

    p.add_argument(
        "--nphi-measure",
        type=int,
        default=96,
    )

    p.add_argument(
        "--measurement-order",
        type=int,
        choices=(1, 3),
        default=3,
        help="Cartesian-to-polar interpolation order.",
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
        "--schrodinger-picture",
        action="store_true",
        help=(
            "Store rapidly accumulating free-propagation "
            "phases explicitly. Default uses the equivalent "
            "time-dependent interaction-picture jumps."
        ),
    )

    p.add_argument(
        "--disable-forward-mean-field",
        action="store_true",
    )

    p.add_argument(
        "--prefix",
        default="sphere_glmt_stage4b_trajectories",
    )

    return p.parse_args()


def execute(args):
    if args.phi <= 0:
        raise ValueError(
            "phi must be positive."
        )

    output_tau = np.asarray(
        parse_float_list(
            args.output_tau
        ),
        dtype=float,
    )

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

    n_v = (
        3.0
        * args.phi
        / (
            4.0
            * math.pi
            * args.radius_um**3
        )
    )

    ell_s = (
        1.0
        / (
            n_v * sigma_s
        )
    )

    zR = (
        math.pi
        * args.n_medium
        * args.w0_um**2
        / args.wavelength_um
    )

    qmax = (
        args.eta_col * k
    )

    grid = make_cartesian_disk(
        qmax,
        args.nside,
        k,
    )

    print()
    print("=" * 92)
    print("PRECOMPUTING CARTESIAN MOMENTUM-TRANSFER JUMPS")
    print("=" * 92)

    print(
        f"retained q points = {grid['npoints']}"
    )

    jumps = (
        precompute_pair_spin_matrices(
            grid=grid,
            mie_a=mie_a,
            mie_b=mie_b,
            k=k,
            sigma_s=sigma_s,
        )
    )

    initial, raw_norm = (
        initial_lg_state(
            grid=grid,
            w0_um=args.w0_um,
            p=args.input_p,
            l=args.input_l,
            sigma=args.input_sigma,
        )
    )

    rate_diag = (
        1.0
        / grid["cos_theta"]
    )

    H_diff = (
        k
        * (
            1.0
            - grid[
                "cos_theta"
            ]
        )
    )

    sf, beta = (
        forward_amplitude_beta(
            mie_a,
            mie_b,
        )
    )

    mean_field = np.zeros_like(
        H_diff
    )

    if not (
        args.disable_forward_mean_field
    ):
        mean_field = (
            beta
            * (
                1.0
                / grid[
                    "cos_theta"
                ]
                - 1.0
            )
        )

    phase_diag = (
        ell_s * H_diff
        + mean_field
    )

    qprob_initial = (
        retained_q_probabilities(
            initial,
            jumps,
        )
    )

    retained_rate_initial = float(
        np.sum(
            qprob_initial
        )
    )

    total_rate_initial = float(
        np.sum(
            np.sum(
                np.abs(initial) ** 2,
                axis=1,
            )
            * rate_diag
        )
    )

    initial_collected_cross_section = (
        retained_rate_initial
        * sigma_s
    )

    initial_full_cross_section = (
        total_rate_initial
        * sigma_s
    )

    measurement = (
        make_measurement_grid(
            qmax,
            args.nqr_measure,
            args.nphi_measure,
        )
    )

    initial_measure = state_to_oam(
        state=initial,
        grid=grid,
        measurement=measurement,
        lmin=args.lmin_measure,
        lmax=args.lmax_measure,
        interpolation_order=args.measurement_order,
    )

    print(
        "initial grid norm before renormalization = "
        f"{raw_norm:.12f}"
    )

    print(
        "initial full cross section [um^2] = "
        f"{initial_full_cross_section:.9f}"
    )

    print(
        "initial collected cross section [um^2] = "
        f"{initial_collected_cross_section:.9f}"
    )

    print(
        "initial retained fraction = "
        f"{retained_rate_initial / total_rate_initial:.9f}"
    )

    print(
        "initial OAM measurement capture = "
        f"{initial_measure['measured_total']:.9f}"
    )

    results = simulate(
        args=args,
        grid=grid,
        jumps=jumps,
        initial_state=initial,
        rate_diag=rate_diag,
        phase_diag=phase_diag,
        measurement=measurement,
        output_tau=output_tau,
    )

    rows = []

    for it, tau in enumerate(
        output_tau
    ):
        spin_plus = float(
            np.sum(
                results[
                    "weighted_power"
                ][it, 0]
            )
        )

        spin_minus = float(
            np.sum(
                results[
                    "weighted_power"
                ][it, 1]
            )
        )

        rows.append(
            {
                "tau": float(tau),
                "z_um": float(
                    tau * ell_s
                ),
                "transmittance": float(
                    results[
                        "transmittance"
                    ][it]
                ),
                "measurement_captured_weight": float(
                    results[
                        "measurement_capture"
                    ][it]
                ),
                "effective_sample_size": float(
                    results[
                        "effective_sample_size"
                    ][it]
                ),
                "spin_plus_unconditional": (
                    spin_plus
                ),
                "spin_minus_unconditional": (
                    spin_minus
                ),
                "spin_flip_conditional": float(
                    spin_minus
                    / max(
                        spin_plus
                        + spin_minus,
                        1e-300,
                    )
                ),
                "oam_unconditional": {
                    str(
                        int(l)
                    ): float(
                        np.sum(
                            results[
                                "weighted_power"
                            ][
                                it,
                                :,
                                il,
                            ]
                        )
                    )
                    for il, l in enumerate(
                        results[
                            "l_values"
                        ]
                    )
                },
                "oam_conditional": {
                    str(
                        int(l)
                    ): float(
                        np.sum(
                            results[
                                "conditional_power"
                            ][
                                it,
                                :,
                                il,
                            ]
                        )
                    )
                    for il, l in enumerate(
                        results[
                            "l_values"
                        ]
                    )
                },
            }
        )

    full = {
        "script_version": 1,
        "scope": (
            "Stage-4B radial-resolved "
            "multiple-scattering propagation "
            "by weighted quantum trajectories"
        ),
        "parameters": args_dict(
            args
        ),
        "physics": {
            "size_parameter": float(
                x
            ),
            "mie_nmax": int(
                mie_nmax
            ),
            "Qsca": float(
                qsca
            ),
            "sigma_s_um2": float(
                sigma_s
            ),
            "n_v_um3_inv": float(
                n_v
            ),
            "ell_s_um": float(
                ell_s
            ),
            "zR_um": float(
                zR
            ),
            "ell_s_over_zR": float(
                ell_s / zR
            ),
            "forward_amplitude": {
                "real": float(
                    sf.real
                ),
                "imag": float(
                    sf.imag
                ),
            },
            "forward_mean_field_beta_per_ell_s": float(
                beta
            ),
            "mean_field_enabled": bool(
                not args.disable_forward_mean_field
            ),
        },
        "grid_validation": {
            "npoints": int(
                grid["npoints"]
            ),
            "nside": int(
                grid["nside"]
            ),
            "dq_um_inv": float(
                grid["dq"]
            ),
            "initial_raw_flux_norm": float(
                raw_norm
            ),
            "initial_full_cross_section_um2": float(
                initial_full_cross_section
            ),
            "initial_collected_cross_section_um2": float(
                initial_collected_cross_section
            ),
            "initial_retained_fraction": float(
                retained_rate_initial
                / total_rate_initial
            ),
            "initial_oam_measurement_capture": float(
                initial_measure[
                    "measured_total"
                ]
            ),
        },
        "outputs": rows,
        "trajectory_diagnostics": {
            "mean_segment_jumps": (
                results[
                    "mean_segment_jumps"
                ].tolist()
            ),
            "max_segment_jumps": (
                results[
                    "max_segment_jumps"
                ].tolist()
            ),
            "resampling": (
                "systematic resampling at requested output depths"
            ),
        },
        "notes": {
            "weighted_retained_unravelling": (
                "retained jumps are sampled conditionally; "
                "trajectory weights accumulate retained probabilities"
            ),
            "radial_reset": False,
            "free_diffraction": True,
            "collision_picture": (
                "Schrodinger-like"
                if args.schrodinger_picture
                else "interaction picture with "
                "time-dependent jump phases"
            ),
            "interaction_picture_reason": (
                "Radial free phases commute with OAM measurement. "
                "Removing them from the stored state reduces "
                "finite-grid interpolation error."
            ),
        },
    }

    full_path = write_json(
        full,
        f"{args.prefix}_FULL_RESULTS.json",
    )

    print()
    print("=" * 92)
    print("STAGE 4B PROPAGATION SUMMARY")
    print("=" * 92)

    print(
        f"ell_s [um] = {ell_s:.9f}"
    )

    print(
        f"ell_s/zR = {ell_s / zR:.9f}"
    )

    for row in rows:
        print(
            f"tau={row['tau']:6.2f}: "
            f"T={row['transmittance']:.6e}, "
            f"ESS={row['effective_sample_size']:.1f}, "
            f"spinflip|col={row['spin_flip_conditional']:.6e}"
        )

    print()
    print(
        f"Primary review file: {full_path}"
    )


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
