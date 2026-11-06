#!/usr/bin/env python
"""Simulate the phase of an arc (double difference between two points) in the SARvey model.

phase = wrap( factor * [ pb/(r*sin(inc)) * demerr + tb * vel + te * tcoef ] + noise )

with factor = 4*pi/wavelength.

Inputs
------
- ifg_network.h5 : temporal, perpendicular and temperature baselines (sarvey.ifg_network.IfgNetwork)
- points file    : wavelength, slant range and local incidence angle (sarvey.objects.Points).
                   Slant range and incidence angle are taken from one point of the file, which is
                   chosen randomly unless --point-index is given.

Modes
-----
network (default) : one simulated value per real interferogram (baselines (tb, pb, te) belong together).
                    This is the physically consistent simulation.
grid              : dense grid where tb, pb and te are three independent linspaces between the min and max
                    of the network (as in the plotting code of sarvey_plot_arc). The baselines of a sample
                    do NOT belong to the same interferogram. Only meant for visualising the model.

Units (same as SARvey):
    demerr : DEM error [m]
    vel    : velocity [m/year]
    tcoef  : thermal coefficient [m/degC]
    noise  : phase noise standard deviation [rad]

Examples
--------
    python sarvey_simulate_arc.py ifg_network.h5 p1_ifg_wr.h5 inputs/ --demerr 5 --vel 0.005 --tcoef 0.0002 \
        --noise-std 0.6 --seed 1 --plot --estimate \
        --dembound 20 --velbound 0.05 --tcoefbound 0.001 -n 100 --save sim.npz

    python sarvey_simulate_arc.py ifg_network.h5 p1_ifg_wr.h5 inputs/ --demerr 5 --vel 0.005 --tcoef 0.0002 \
        --mode grid --num-grid 1000 --plot
"""
import argparse
from logging import Logger

import numpy as np

from sarvey.ifg_network import IfgNetwork
from sarvey.objects import Points


def load_baselines(ifg_network_file):
    """Read temporal, perpendicular and temperature baselines of all interferograms."""
    ifgnet = IfgNetwork()
    ifgnet.open(path=ifg_network_file)
    tb_ifg = np.asarray(ifgnet.tbase_ifg)
    pb_ifg = np.asarray(ifgnet.pbase_ifg)
    te_ifg = np.asarray(ifgnet.temperatures_ifg)
    return tb_ifg, pb_ifg, te_ifg


def load_geometry(points_file, input_path, point_index=None, seed=None):
    """Read wavelength, slant range and local incidence angle from a SARvey points file.

    If point_index is None, a random point is chosen (reproducible with seed).
    """
    point_obj = Points(file_path=points_file, logger=Logger("log"))
    point_obj.open(input_path=input_path)

    if point_index is None:
        rng = np.random.default_rng(seed)
        point_index = int(rng.integers(0, point_obj.num_points))
    elif not (0 <= point_index < point_obj.num_points):
        raise SystemExit(f"--point-index must be in [0, {point_obj.num_points - 1}]")

    slant_range = float(point_obj.slant_range[point_index])
    loc_inc = float(point_obj.loc_inc[point_index])
    wavelength = float(point_obj.wavelength)
    return wavelength, slant_range, loc_inc, point_index


def build_design_matrix(tb, pb, te, wavelength, slant_range, loc_inc):
    """Design matrix: one row per sample, columns = [demerr, velocity, thermal coefficient]."""
    factor = 4 * np.pi / wavelength
    design_mat = np.zeros((tb.size, 3), dtype=np.float32)
    design_mat[:, 0] = factor * pb / (slant_range * np.sin(loc_inc))
    design_mat[:, 1] = factor * tb
    design_mat[:, 2] = factor * te
    return design_mat


def make_grid(tb_ifg, pb_ifg, te_ifg, num_grid):
    """Independent linspaces between min and max of each baseline (not physically consistent)."""
    tb = np.linspace(tb_ifg.min(), tb_ifg.max(), num_grid)
    pb = np.linspace(pb_ifg.min(), pb_ifg.max(), num_grid)
    te = np.linspace(te_ifg.min(), te_ifg.max(), num_grid)
    return tb, pb, te


def wrap(phase):
    return np.angle(np.exp(1j * phase))


def simulate_phase(design_mat, demerr, vel, tcoef, noise_std=0.0, seed=None, wrap_phase=True):
    """Simulate arc phase.

    Returns
    -------
    phase_obs   : simulated (noisy, wrapped) phase
    phase_clean : noise-free phase (wrapped if wrap_phase=True)
    noise       : the noise realisation [rad]
    """
    rng = np.random.default_rng(seed)
    params = np.array([demerr, vel, tcoef], dtype=np.float64)
    phase_true = design_mat.astype(np.float64) @ params
    noise = rng.normal(0.0, noise_std, phase_true.size) if noise_std > 0 else np.zeros_like(phase_true)
    phase_obs = phase_true + noise
    if wrap_phase:
        return wrap(phase_obs), wrap(phase_true), noise
    return phase_obs, phase_true, noise


def plot_simulation(tb, pb, te, phase_obs, phase_clean, tb_ifg, pb_ifg, te_ifg,
                    wavelength, slant_range, loc_inc, demerr, vel, tcoef, title=""):
    """Scatter the simulated phase against each baseline and overlay the model curve of the matching component."""
    import matplotlib.pyplot as plt

    # dense, independent grids for the per-component model curves
    tb_s, pb_s, te_s = make_grid(tb_ifg, pb_ifg, te_ifg, 1000)
    dm = build_design_matrix(tb_s, pb_s, te_s, wavelength, slant_range, loc_inc)
    curves = [wrap(dm[:, 1] * vel), wrap(dm[:, 0] * demerr), wrap(dm[:, 2] * tcoef)]
    curve_x = [tb_s, pb_s, te_s]
    curve_label = ["velocity", "DEM error", "thermal coef."]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, x, label, cx, cy, cl in zip(axes, [tb, pb, te],
                                        ["Temporal baseline", "Perpendicular baseline", "Temperature baseline"],
                                        curve_x, curves, curve_label):
        ax.scatter(x, phase_obs, c='b', s=15, alpha=0.7, label="simulated (noisy)")
        ax.scatter(x, phase_clean, c='r', s=4, alpha=0.7, label="noise-free")
        ax.scatter(cx, cy, c='k', marker='.', s=0.5, alpha=0.7, label=f"{cl} only")
        ax.set_xlabel(label)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
        ax.legend(loc="lower left", fontsize=7)
    axes[0].set_ylabel("Phase [rad]")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def create_parser():
    p = argparse.ArgumentParser(description="Simulate arc phase (DEM error, velocity, thermal coefficient).",
                                formatter_class=argparse.RawTextHelpFormatter, epilog=__doc__)
    p.add_argument('ifg_network', type=str, help='path to ifg_network.h5')
    p.add_argument('points_file', type=str, help='SARvey points file (wavelength, slant range, incidence angle)')
    p.add_argument('inputpath', type=str, help='input path of the SARvey points file')

    # model parameters
    p.add_argument('--demerr', type=float, default=0.0, help='DEM error [m]')
    p.add_argument('--vel', type=float, default=0.0, help='velocity [m/year]')
    p.add_argument('--tcoef', type=float, default=0.0, help='thermal coefficient [m/degC]')

    # sampling
    p.add_argument('--mode', choices=['network', 'grid'], default='network',
                   help='network: real ifg baselines (default); grid: dense independent linspaces')
    p.add_argument('--num-grid', dest='num_grid', type=int, default=1000, help='number of samples for --mode grid')

    # noise
    p.add_argument('--noise-std', dest='noise_std', type=float, default=0.0,
                   help='std of Gaussian phase noise [rad]; 0 = no noise (default)')
    p.add_argument('--seed', type=int, default=None, help='random seed (noise and random point choice)')
    p.add_argument('--no-wrap', dest='wrap', action='store_false', help='do not wrap the phase')

    # geometry point
    p.add_argument('--point-index', dest='point_index', type=int, default=None,
                   help='index of the point used for slant range / incidence angle (default: random)')

    # output
    p.add_argument('--plot', action='store_true', help='plot simulated phase')
    p.add_argument('--save', type=str, default=None, help='save simulation to .npz')

    # optional re-estimation
    p.add_argument('--estimate', action='store_true', help='run the SARvey 3-parameter search on the simulated phase')
    p.add_argument('--dembound', type=float, help='DEM error bound (needed for --estimate)')
    p.add_argument('--velbound', type=float, help='velocity bound (needed for --estimate)')
    p.add_argument('--tcoefbound', type=float, help='thermal coefficient bound (needed for --estimate)')
    p.add_argument('-n', dest='numsamples', type=int, default=100, help='number of samples per parameter for --estimate')
    return p.parse_args()


def main():
    args = create_parser()

    tb_ifg, pb_ifg, te_ifg = load_baselines(args.ifg_network)
    wavelength, slant_range, loc_inc, point_index = load_geometry(
        args.points_file, args.inputpath, point_index=args.point_index, seed=args.seed)

    if args.mode == 'grid':
        tb, pb, te = make_grid(tb_ifg, pb_ifg, te_ifg, args.num_grid)
    else:
        tb, pb, te = tb_ifg, pb_ifg, te_ifg

    design_mat = build_design_matrix(tb, pb, te, wavelength, slant_range, loc_inc)
    phase_obs, phase_clean, noise = simulate_phase(
        design_mat, args.demerr, args.vel, args.tcoef, noise_std=args.noise_std, seed=args.seed,
        wrap_phase=args.wrap)

    print(f"Mode: {args.mode}, simulated {phase_obs.size} samples")
    print(f"Geometry from point {point_index}: slant range {slant_range:.1f} m, "
          f"incidence angle {np.rad2deg(loc_inc):.2f} deg, wavelength {wavelength:.4f} m")
    print(f"True parameters: DEM error {args.demerr}, velocity {args.vel}, thermal coefficient {args.tcoef}")
    print(f"Noise std: {args.noise_std} rad (realised: {noise.std():.3f} rad)")

    if args.estimate:
        if None in (args.dembound, args.velbound, args.tcoefbound):
            raise SystemExit("--estimate requires --dembound, --velbound and --tcoefbound")
        from sarvey.unwrapping_temperature import oneDimSearchTemporalCoherence_3variables
        demerr, vel, tcoef, gamma = oneDimSearchTemporalCoherence_3variables(
            demerr_range=np.linspace(-args.dembound, args.dembound, args.numsamples),
            vel_range=np.linspace(-args.velbound, args.velbound, args.numsamples),
            tcoef_range=np.linspace(-args.tcoefbound, args.tcoefbound, args.numsamples),
            obs_phase=phase_obs,
            design_mat=design_mat)
        print(f"Estimated parameters: DEM error {demerr}, velocity {vel}, thermal coefficient {tcoef}, gamma {gamma}")

    if args.save is not None:
        np.savez(args.save, phase=phase_obs, phase_clean=phase_clean, noise=noise,
                 tb=tb, pb=pb, te=te, design_mat=design_mat, mode=args.mode,
                 demerr=args.demerr, vel=args.vel, tcoef=args.tcoef, noise_std=args.noise_std,
                 wavelength=wavelength, slant_range=slant_range, loc_inc=loc_inc, point_index=point_index)
        print(f"Saved to {args.save}")

    if args.plot:
        plot_simulation(tb, pb, te, phase_obs, phase_clean, tb_ifg, pb_ifg, te_ifg,
                        wavelength, slant_range, loc_inc, args.demerr, args.vel, args.tcoef,
                        title=f"[{args.mode}] DEM err={args.demerr} m, vel={args.vel} m/yr, "
                              f"tcoef={args.tcoef} m/degC, noise={args.noise_std} rad")


if __name__ == '__main__':
    main()