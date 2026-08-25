#!/usr/bin/env python

# SARvey - A multitemporal InSAR time series tool for the derivation of displacements.
#
# Copyright (C) 2021-2026 Andreas Piter (IPI Hannover, piter@ipi.uni-hannover.de)
#
# This software was developed together with FERN.Lab (fernlab@gfz-potsdam.de) in the context
# of the SAR4Infra project with funds of the German Federal Ministry for Digital and
# Transport and contributions from Landesamt fuer Vermessung und Geoinformation
# Schleswig-Holstein and Landesbetrieb Strassenbau und Verkehr Schleswig-Holstein.
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# Important: This package uses PyMaxFlow. The core of PyMaxflows library is the C++
# implementation by Vladimir Kolmogorov. It is also licensed under the GPL, but it REQUIRES that you
# cite [BOYKOV04] (see LICENSE) in any resulting publication if you use this code for research purposes.
# This requirement extends to SARvey.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with this program. If not, see <https://www.gnu.org/licenses/>.

"""Plot module for SARvey."""
import argparse
import time
import os
from os.path import join, basename, dirname
import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
import numpy as np
import logging
from logging import Logger
import sys
import cmcrameri as cmc

from miaplpy.objects.slcStack import slcStack
from scipy.spatial import KDTree


from sarvey import version
from sarvey.objects import PointsPiecewise, AmplitudeImage
from sarvey import console
from sarvey import viewer
from sarvey.config import loadConfiguration
import sarvey.utils as ut
from sarvey.viewer import LineSelector, ImageViewer
from sarvey.ifg_network import IfgNetwork
from sarvey.unwrapping import oneDimSearchTemporalCoherence

try:
    matplotlib.use('QtAgg')
except ImportError as e:
    print(e)


def plot_arc_fit(phase, tb_ifg, pb_ifg, demerr, vel, factor, slant_range, loc_inc, suptitle):
    """Create a 3x2 diagnostic plot (temporal baseline / perpendicular baseline) for one estimate."""
    pred_phase_demerr = factor * pb_ifg / (slant_range * np.sin(loc_inc)) * demerr
    pred_phase_vel = factor * tb_ifg * vel

    fig, axes = plt.subplots(3, 2, figsize=(8, 10), sharey='col')
    fig.suptitle(suptitle)

    # Set y-limits from -pi to pi for all subplots
    for ax_row in axes:
        for ax in ax_row:
            ax.set_ylim(-np.pi, np.pi)
            ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
            ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])

    #### predicted phase for plotting ####
    nsamples = 1000
    tb_space = np.linspace(tb_ifg.min(), tb_ifg.max(), nsamples)
    pb_space = np.linspace(pb_ifg.min(), pb_ifg.max(), nsamples)
    design_mat_pred = np.zeros((tb_space.size, 2), dtype=np.float32)
    design_mat_pred[:, 0] = factor * pb_space / (slant_range * np.sin(loc_inc))
    design_mat_pred[:, 1] = factor * tb_space

    # Column 1: Temporal baseline
    # Row 1: Raw phase
    axes[0, 0].scatter(tb_ifg, phase, c='b', s=15, alpha=0.7)
    predphase = design_mat_pred[:, 1] * vel
    axes[0, 0].scatter(tb_space, np.angle(np.exp(1j * predphase)), marker='.', s=0.5, alpha=0.7)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].text(0.02, 0.95, r'$\phi$', transform=axes[0, 0].transAxes,
                    fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Row 2: phase - pred_demerror
    resphase = np.angle(np.exp(1j * phase) * np.conjugate(np.exp(1j * pred_phase_demerr)))
    axes[1, 0].scatter(tb_ifg, resphase, c='g', s=15, alpha=0.7)
    predphase = design_mat_pred[:, 1] * vel
    axes[1, 0].scatter(tb_space, np.angle(np.exp(1j * predphase)), marker='.', s=0.5, alpha=0.7)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].text(0.02, 0.95, r'$\phi - \phi_{DEM}$', transform=axes[1, 0].transAxes,
                    fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Row 3: phase - (pred_demerror + pred_vel)
    resphase = np.angle(np.exp(1j * phase) * np.conjugate(np.exp(1j * (pred_phase_demerr + pred_phase_vel))))
    axes[2, 0].scatter(tb_ifg, resphase, c='m', s=15, alpha=0.7)
    axes[2, 0].set_xlabel('Temporal Baseline')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].text(0.02, 0.95, r'$\phi - (\phi_{DEM} + \phi_{Vel})$', transform=axes[2, 0].transAxes,
                    fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Column 2: Perpendicular baseline
    # Row 1: Raw phase
    axes[0, 1].scatter(pb_ifg, phase, c='b', s=15, alpha=0.7)
    predphase = demerr * design_mat_pred[:, 0]
    axes[0, 1].scatter(pb_space, np.angle(np.exp(1j * predphase)), marker='.', s=0.5, alpha=0.7)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].text(0.02, 0.95, r'$\phi$', transform=axes[0, 1].transAxes,
                    fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Row 2: phase - pred_vel
    resphase = np.angle(np.exp(1j * phase) * np.conjugate(np.exp(1j * pred_phase_vel)))
    axes[1, 1].scatter(pb_ifg, resphase, c='g', s=15, alpha=0.7)
    predphase = demerr * design_mat_pred[:, 0]
    axes[1, 1].scatter(pb_space, np.angle(np.exp(1j * predphase)), marker='.', s=0.5, alpha=0.7)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].text(0.02, 0.95, r'$\phi - \phi_{Vel}$', transform=axes[1, 1].transAxes,
                    fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Row 3: phase - (pred_demerror + pred_vel)
    resphase = np.angle(np.exp(1j * phase) * np.conjugate(np.exp(1j * (pred_phase_demerr + pred_phase_vel))))
    axes[2, 1].scatter(pb_ifg, resphase, c='m', s=15, alpha=0.7)
    axes[2, 1].set_xlabel('Perpendicular Baseline')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].text(0.02, 0.95, r'$\phi - (\phi_{DEM} + \phi_{Vel})$', transform=axes[2, 1].transAxes,
                    fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout()
    return fig


def main():
    args = createParser()
    inputpath = args.inputpath
    pointspath = args.disp_file
    p1x, p1y = args.xyp1
    p2x, p2y = args.xyp2
    dembound = args.dembound
    velbound = args.velbound
    velboundexca = args.velboundexca
    num_samples = args.numsamples

    logger = Logger("log")
    point_obj = PointsPiecewise(file_path=pointspath, logger=logger)
    point_obj.open(input_path=inputpath)

    xy_tree = KDTree(point_obj.coord_xy)
    ixp1 = xy_tree.query([p1y, p1x])[-1]
    ixp2 = xy_tree.query([p2y, p2x])[-1]
    p1y, p1x = np.unravel_index(point_obj.point_id[ixp1], (point_obj.length, point_obj.width))
    p2y, p2x = np.unravel_index(point_obj.point_id[ixp2], (point_obj.length, point_obj.width))

    # double-differenced phase for the full period, the pre-excavation period, and the excavation period
    phase = np.angle(np.exp(1j * point_obj.phase[ixp1, :]) * np.conjugate(np.exp(1j * point_obj.phase[ixp2, :])))
    phase_pre = np.angle(np.exp(1j * point_obj.phase_pre[ixp1, :]) * np.conjugate(np.exp(1j * point_obj.phase_pre[ixp2, :])))
    phase_exca = np.angle(np.exp(1j * point_obj.phase_exca[ixp1, :]) * np.conjugate(np.exp(1j * point_obj.phase_exca[ixp2, :])))

    loc_inc = np.mean([point_obj.loc_inc[ixp1], point_obj.loc_inc[ixp2]])
    slant_range = np.mean([point_obj.slant_range[ixp1], point_obj.slant_range[ixp2]])
    factor = 4 * np.pi / point_obj.wavelength

    tb_ifg = point_obj.ifg_net_obj.tbase_ifg
    pb_ifg = point_obj.ifg_net_obj.pbase_ifg

    tb_ifg_pre = point_obj.ifg_net_obj.tbase_ifg_pre
    pb_ifg_pre = point_obj.ifg_net_obj.pbase_ifg_pre

    tb_ifg_exca = point_obj.ifg_net_obj.tbase_ifg_exca
    pb_ifg_exca = point_obj.ifg_net_obj.pbase_ifg_exca

    design_mat = np.zeros((point_obj.ifg_net_obj.num_ifgs, 2), dtype=np.float32)
    design_mat[:, 0] = factor * pb_ifg / (slant_range * np.sin(loc_inc))
    design_mat[:, 1] = factor * tb_ifg

    design_mat_pre = np.zeros((point_obj.ifg_net_obj.num_ifgs_pre, 2), dtype=np.float32)
    design_mat_pre[:, 0] = factor * pb_ifg_pre / (slant_range * np.sin(loc_inc))
    design_mat_pre[:, 1] = factor * tb_ifg_pre

    design_mat_exca = np.zeros((point_obj.ifg_net_obj.num_ifgs_exca, 2), dtype=np.float32)
    design_mat_exca[:, 0] = factor * pb_ifg_exca / (slant_range * np.sin(loc_inc))
    design_mat_exca[:, 1] = factor * tb_ifg_exca

    demerr_range = np.linspace(-dembound, dembound, num_samples)
    vel_range = np.linspace(-velbound, velbound, num_samples)
    vel_excavation_range = np.linspace(-velboundexca, velboundexca, num_samples)

    demerr, vel, gamma = oneDimSearchTemporalCoherence(
        demerr_range=demerr_range,
        vel_range=vel_range,
        obs_phase=phase,
        design_mat=design_mat)

    demerr_pre, vel_pre, gamma_pre = oneDimSearchTemporalCoherence(
        demerr_range=demerr_range,
        vel_range=vel_range,
        obs_phase=phase_pre,
        design_mat=design_mat_pre)

    demerr_exca, vel_exca, gamma_exca = oneDimSearchTemporalCoherence(
        demerr_range=demerr_range,
        vel_range=vel_excavation_range,
        obs_phase=phase_exca,
        design_mat=design_mat_exca)

    print(f"Full period      -> DEM error: {demerr}, velocity: {vel}, gamma: {gamma}")
    print(f"Pre-excavation    -> DEM error: {demerr_pre}, velocity: {vel_pre}, gamma: {gamma_pre}")
    print(f"Excavation        -> DEM error: {demerr_exca}, velocity: {vel_exca}, gamma: {gamma_exca}")

    plot_arc_fit(phase, tb_ifg, pb_ifg, demerr, vel, factor, slant_range, loc_inc,
                 suptitle="Full period")
    plot_arc_fit(phase_pre, tb_ifg_pre, pb_ifg_pre, demerr_pre, vel_pre, factor, slant_range, loc_inc,
                 suptitle="Pre-excavation")
    plot_arc_fit(phase_exca, tb_ifg_exca, pb_ifg_exca, demerr_exca, vel_exca, factor, slant_range, loc_inc,
                 suptitle="Excavation")

    plt.show()

    # Option to save the figures
    # fig.savefig('phase_corrections_plot.png', dpi=300, bbox_inches='tight')


def createParser():
    """plotting arc for testing"""
    EXAMPLE = """
        sarvey_plot_arc.py p1_ifg_wr.h5 -p1 84 391 -p2 48 392
    """
    parser = argparse.ArgumentParser(
        description='Arc modelling\n\n',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=EXAMPLE)

    parser.add_argument('disp_file', type=str, help='displacement file to wr in h5 format')
    parser.add_argument('inputpath', type=str, help='inputpath')
    parser.add_argument('-p1', dest="xyp1", nargs=2, type=int, help="x and y coords of p1: 391 50")
    parser.add_argument('-p2', dest="xyp2", nargs=2, type=int, help="x and y coords of p2: 388 50")
    parser.add_argument('--dembound', dest='dembound', type=float, required=True, help='DEM error bound')
    parser.add_argument('--velbound', dest='velbound', type=float, required=True, help='linear velocity bound')
    parser.add_argument('--velboundexca', dest='velboundexca', type=float, required=True,
                         help='linear velocity bound for the excavation period')
    parser.add_argument('-n', dest='numsamples', type=int, required=True, help='Number of samples for optimization')

    return parser.parse_args()


if __name__ == '__main__':
    main()
