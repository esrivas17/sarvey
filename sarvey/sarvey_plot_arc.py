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
from sarvey.objects import Points, AmplitudeImage
from sarvey import console
from sarvey import viewer
from sarvey.config import loadConfiguration
import sarvey.utils as ut
from sarvey.viewer import LineSelector, ImageViewer
from sarvey.ifg_network import IfgNetwork
from sarvey.unwrapping_temperature import oneDimSearchTemporalCoherence_3variables_hytest, oneDimSearchTemporalCoherence_3variables

try:
    matplotlib.use('QtAgg')
except ImportError as e:
    print(e)


def main():
    args = createParser()
    inputpath = args.inputpath
    pointspath = args.disp_file
    p1x, p1y = args.xyp1
    p2x, p2y = args.xyp2
    dembound = args.dembound
    velbound = args.velbound
    tcoefbound = args.tcoefbound
    num_samples = args.numsamples


    logger = Logger("log")
    point_obj = Points(file_path=pointspath, logger=logger)
    point_obj.open(input_path=inputpath)

    xy_tree = KDTree(point_obj.coord_xy)
    ixp1 = xy_tree.query([p1y, p1x])[-1]
    ixp2 = xy_tree.query([p2y, p2x])[-1]
    p1y, p1x = np.unravel_index(point_obj.point_id[ixp1], (point_obj.length, point_obj.width))
    p2y, p2x = np.unravel_index(point_obj.point_id[ixp2], (point_obj.length, point_obj.width))

    phase = np.angle(np.exp(1j * point_obj.phase[ixp1, :]) * np.conjugate(np.exp(1j * point_obj.phase[ixp2, :])))
    loc_inc = np.mean([point_obj.loc_inc[ixp1], point_obj.loc_inc[ixp2]])
    slant_range = np.mean([point_obj.slant_range[ixp1], point_obj.slant_range[ixp2]])

    tb_ifg = point_obj.ifg_net_obj.tbase_ifg
    pb_ifg = point_obj.ifg_net_obj.pbase_ifg
    te_ifg = point_obj.ifg_net_obj.temperatures_ifg

    design_mat = np.zeros((point_obj.ifg_net_obj.num_ifgs, 3), dtype=np.float32)


    factor = 4 * np.pi / point_obj.wavelength

    design_mat[:, 0] = factor * pb_ifg / (slant_range * np.sin(loc_inc))
    design_mat[:, 1] = factor * tb_ifg
    design_mat[:, 2] = factor * te_ifg

    demerr_range = np.linspace(-dembound, dembound, num_samples)
    vel_range = np.linspace(-velbound, velbound, num_samples)
    tcoef_range = np.linspace(-tcoefbound, tcoefbound, num_samples)

    demerr, vel, tcoef, gamma = oneDimSearchTemporalCoherence_3variables_hytest(
                demerr_range=demerr_range,
                vel_range=vel_range,
                tcoef_range=tcoef_range,
                obs_phase=phase,
                design_mat=design_mat)

    demerr, vel, tcoef, gamma = oneDimSearchTemporalCoherence_3variables(
                    demerr_range=demerr_range,
                    vel_range=vel_range,
                    tcoef_range=tcoef_range,
                    obs_phase=phase,
                    design_mat=design_mat)


    pred_phase_demerr = factor * pb_ifg / (slant_range * np.sin(loc_inc)) * demerr
    pred_phase_vel = factor * tb_ifg * vel
    pred_phase_tcoef = factor * te_ifg * tcoef

    print(f"Estimated parameter: DEM error {demerr}, velocity: {vel}, thermal coefficient: {tcoef}, gamma: {gamma}")
    #print(f"Estimated parameter: DEM error {demerr2}, velocity: {vel2}, thermal coefficient: {tcoef2}, gamma: {gamma2}")

    ######### Create 4x3 subplot grid ###############
    fig, axes = plt.subplots(4, 3, figsize=(15, 16), sharey='col')

    # Set y-limits from -pi to pi for all subplots
    for ax_row in axes:
        for ax in ax_row:
            ax.set_ylim(-np.pi, np.pi)
            ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
            ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])

    # Column 1: Temporal baseline
    # Row 1: Raw phase
    axes[0, 0].scatter(tb_ifg, phase, c='b', s=17, alpha=0.7)
    #axes[0, 0].set_title('Phase (Temporal)', fontsize=12)
    #axes[0, 0].set_xlabel('Temporal Baseline')
    axes[0, 0].grid(True, alpha=0.3)
    # Add Greek symbol text
    axes[0, 0].text(0.05, 0.95, r'$\phi$', transform=axes[0, 0].transAxes, 
                    fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Row 2: phase - pred_demerror
    resphase = np.angle(np.exp(1j * phase) * np.conjugate(np.exp(1j * pred_phase_demerr)))
    axes[1, 0].scatter(tb_ifg, resphase, c='g', s=17, alpha=0.7)
    resphase = demerr * tb_ifg
    axes[1, 0].plot(tb_ifg, np.angle(resphase))
    #axes[1, 0].set_title('Phase - Pred_DemError (Temporal)', fontsize=12)
    #axes[1, 0].set_xlabel('Temporal Baseline')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].text(0.05, 0.95, r'$\phi - \phi_{DEM}$', transform=axes[1, 0].transAxes, 
                    fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Row 3: phase - (pred_demerror + pred_tcoef)
    resphase = np.angle(np.exp(1j * phase) * np.conjugate(np.exp(1j * (pred_phase_demerr + pred_phase_vel))))
    axes[2, 0].scatter(tb_ifg, resphase, c='r', s=17, alpha=0.7)
    #axes[2, 0].set_title('Phase - (DemError + TCoef) (Temporal)', fontsize=12)
    #axes[2, 0].set_xlabel('Temporal Baseline')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].text(0.05, 0.95, r'$\phi - (\phi_{DEM} + \phi_{Vel})$', transform=axes[2, 0].transAxes, 
                    fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Row 4: phase - (pred_demerror + pred_tcoef + pred_vel)
    resphase = np.angle(np.exp(1j * phase) * np.conjugate(np.exp(1j * (pred_phase_demerr + pred_phase_vel + pred_phase_tcoef))))
    axes[3, 0].scatter(tb_ifg, resphase, c='m', s=17, alpha=0.7)
    axes[3, 0].set_xlabel('Temporal Baseline')
    #axes[3, 0].set_ylabel('Phase')  # Only bottom plot shows y-label
    axes[3, 0].grid(True, alpha=0.3)
    axes[3, 0].text(0.05, 0.95, r'$\phi - (\phi_{DEM} + \phi_{Vel} + \phi_{TCoef})$', transform=axes[3, 0].transAxes, 
                    fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))


    # Column 2: Perpendicular baseline
    # Row 1: Raw phase
    axes[0, 1].scatter(pb_ifg, phase, c='b', s=17, alpha=0.7)
    #axes[0, 1].set_title('Phase (Perp Baseline)', fontsize=12)
    #axes[0, 1].set_xlabel('Perpendicular Baseline')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].text(0.05, 0.95, r'$\phi$', transform=axes[0, 1].transAxes, 
                    fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Row 2: phase - pred_demerror
    resphase = np.angle(np.exp(1j * phase) * np.conjugate(np.exp(1j * pred_phase_tcoef)))
    axes[1, 1].scatter(pb_ifg, resphase, c='g', s=17, alpha=0.7)
    #axes[1, 1].set_title('Phase - Pred_DemError (Perp Baseline)', fontsize=12)
    #axes[1, 1].set_xlabel('Perpendicular Baseline')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].text(0.05, 0.95, r'$\phi - \phi_{TCoef}$', transform=axes[1, 1].transAxes, 
                    fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Row 3: phase - (pred_demerror + pred_tcoef)
    resphase = np.angle(np.exp(1j * phase) * np.conjugate(np.exp(1j * (pred_phase_demerr + pred_phase_tcoef))))
    axes[2, 1].scatter(pb_ifg, resphase, c='r', s=17, alpha=0.7)
    #axes[2, 1].set_title('Phase - (DemError + TCoef) (Perp Baseline)', fontsize=12)
    #axes[2, 1].set_xlabel('Perpendicular Baseline')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].text(0.05, 0.95, r'$\phi - (\phi_{DEM} + \phi_{TCoef})$', transform=axes[2, 1].transAxes, 
                    fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Row 4: phase - (pred_demerror + pred_tcoef + pred_vel)
    resphase = np.angle(np.exp(1j * phase) * np.conjugate(np.exp(1j * (pred_phase_demerr + pred_phase_tcoef + pred_phase_vel))))
    axes[3, 1].scatter(pb_ifg, resphase, c='m', s=17, alpha=0.7)
    #axes[3, 1].set_title('Phase - (DemError + TCoef + Vel) (Perp Baseline)', fontsize=12)
    axes[3, 1].set_xlabel('Perpendicular Baseline')
    #axes[3, 1].set_ylabel('Phase')  # Only bottom plot shows y-label
    axes[3, 1].grid(True, alpha=0.3)
    axes[3, 1].text(0.05, 0.95, r'$\phi - (\phi_{DEM} + \phi_{TCoef} + \phi_{Vel})$', transform=axes[3, 1].transAxes, 
                    fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))


    # Column 3: Temperature baseline
    # Row 1: Raw phase
    axes[0, 2].scatter(te_ifg, phase, c='b', s=17, alpha=0.7)
   # axes[0, 2].set_title('Phase (Temperature)', fontsize=12)
    #axes[0, 2].set_xlabel('Temperature Baseline')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].text(0.05, 0.95, r'$\phi$', transform=axes[0, 2].transAxes, 
                    fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Row 2: phase - pred_demerror
    resphase = np.angle(np.exp(1j * phase) * np.conjugate(np.exp(1j * pred_phase_demerr)))
    axes[1, 2].scatter(te_ifg, resphase, c='g', s=17, alpha=0.7)
    #axes[1, 2].set_title('Phase - Pred_DemError (Temperature)', fontsize=12)
    #axes[1, 2].set_xlabel('Temperature Baseline')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].text(0.05, 0.95, r'$\phi - \phi_{DEM}$', transform=axes[1, 2].transAxes, 
                    fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Row 3: phase - (pred_demerror + pred_tcoef)
    resphase = np.angle(np.exp(1j * phase) * np.conjugate(np.exp(1j * (pred_phase_demerr + pred_phase_tcoef))))
    axes[2, 2].scatter(te_ifg, resphase, c='r', s=17, alpha=0.7)
    #axes[2, 2].set_title('Phase - (DemError + TCoef) (Temperature)', fontsize=12)
    #axes[2, 2].set_xlabel('Temperature Baseline')
    axes[2, 2].grid(True, alpha=0.3)
    axes[2, 2].text(0.05, 0.95, r'$\phi - (\phi_{DEM} + \phi_{TCoef})$', transform=axes[2, 2].transAxes, 
                    fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Row 4: phase - (pred_demerror + pred_tcoef + pred_vel)
    resphase = np.angle(np.exp(1j * phase) * np.conjugate(np.exp(1j * (pred_phase_demerr + pred_phase_tcoef + pred_phase_vel))))
    axes[3, 2].scatter(te_ifg, resphase, c='m', s=17, alpha=0.7)
    #axes[3, 2].set_title('Phase - (DemError + TCoef + Vel) (Temperature)', fontsize=12)
    axes[3, 2].set_xlabel('Temperature Baseline')
    #axes[3, 2].set_ylabel('Phase')  # Only bottom plot shows y-label
    axes[3, 2].grid(True, alpha=0.3)
    axes[3, 2].text(0.05, 0.95, r'$\phi - (\phi_{DEM} + \phi_{TCoef} + \phi_{Vel})$', transform=axes[3, 2].transAxes, 
                    fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Adjust layout to prevent overlapping
    plt.tight_layout()
    plt.show()

    # Option to save the figure
    # plt.savefig('phase_corrections_plot.png', dpi=300, bbox_inches='tight')


def createParser():
    """plotting arc with temperature for testing"""
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
    parser.add_argument('--tcoefbound', dest='tcoefbound', type=float, required=True, help='Temperature coefficient bound')
    parser.add_argument('-n', dest='numsamples', type=int, required=True, help='Number of samples for optimization')

    return parser.parse_args()


if __name__ == '__main__':
    main()
