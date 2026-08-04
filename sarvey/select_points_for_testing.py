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
import os
from os.path import join, basename, dirname
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from logging import Logger
import sys
import cmcrameri as cmc
import shutil

from mintpy.utils import readfile
import sarvey.utils as ut
from sarvey.objects import AmplitudeImage
from sarvey.preparation import selectPixels, createConstraintArcsBetweenPoints, selectPixelsWithADIandTCOH
import pdb

try:
    matplotlib.use('QtAgg')
except ImportError as e:
    print(e)


def main():
    args = createParser()
    qpath = args.path
    qselection = args.selection.lower()
    tcohthresh = args.tcohthresh
    adithresh = args.adithresh
    maskfile = args.maskpath
    gridsize = args.gridsize
    plotwidth = args.plotwidth
    plotheight = args.plotheight
    adipath = args.adipath
    plotflag = args.plotflag

    logger = Logger("log")

    if gridsize <= 0:
        gridsize = None

    if maskfile is not None:
        bmap_obj = AmplitudeImage(file_path=join(qpath, "background_map.h5"))
        mask_valid_area = ut.detectValidAreas(bmap_obj=bmap_obj, logger=logger)
        path_mask_aoi = join(maskfile)
        print("load mask for area of interest from: {}.".format(path_mask_aoi))
        mask_aoi = readfile.read(path_mask_aoi, datasetName='mask')[0].astype(np.bool_)
        mask_valid_area &= mask_aoi

    if qselection == 'tcoh':
        if len(tcohthresh) <= 0:
            raise Exception("No temporal coherence thresholds values given")

        for tcohv in tcohthresh:
            proxy_value = int(tcohv * 100)
            proxy_id = f"coh{proxy_value}"
            cand_mask1 = selectPixels(path=qpath, selection_method="temp_coh", thrsh=tcohv,
                grid_size=gridsize, bool_plot=plotflag, plotwidth=plotwidth, plotheight=plotheight, logger=logger)
            
            print(f"Number of selected pixels: {np.sum(cand_mask1)} with thresh: {tcohv}")
            if maskfile:
                cand_mask1 &= mask_valid_area
                print(f"Number of selected pixels with mask: {np.sum(cand_mask1)} with thresh: {tcohv}")

                fig = plt.figure(figsize=(plotwidth, plotheight))
                ax = fig.add_subplot()
                ax.imshow(mask_valid_area, cmap=cmc.cm.cmaps["grayC"], alpha=0.5, zorder=10, vmin=0, vmax=1, aspect='auto')
                bmap_obj.plot(ax=ax, logger=logger)
                coord_xy = np.array(np.where(cand_mask1)).transpose()
                val = np.ones_like(cand_mask1)
                sc = ax.scatter(coord_xy[:, 1], coord_xy[:, 0], c=val[cand_mask1], s=1, cmap=cmc.cm.cmaps["lajolla_r"],
                                vmin=1, vmax=2)  # set min, max to ensure that points are yellow
                fig.set_constrained_layout(True)
                plt.title(f"Mask for 1OP - {proxy_id}")
                fig.savefig(join(qpath, "pic", f"mask_{proxy_id}.png"), dpi=300)
                plt.close(fig)

    elif qselection == 'adi':
        if os.path.isfile(adipath):
            shutil.copy(adipath, join(qpath, "amplitude_dispersion.h5"))
        else:
            raise Exception(f"{adipath} is not a file")

        if len(adithresh) <= 0:
            raise Exception("No adi thresholds values given")
       
        for adiv in adithresh:
            proxy_value = int(adiv * 100)
            proxy_id = f"adi{proxy_value}"
            cand_mask1 = selectPixels(path=qpath, selection_method="adi", thrsh=adiv,
                grid_size=gridsize, bool_plot=plotflag,plotwidth=plotwidth, plotheight=plotheight, logger=logger)
            
            print(f"Number of selected pixels: {np.sum(cand_mask1)} with thresh: {tcohv}")

            if maskfile:
                cand_mask1 &= mask_valid_area
                print(f"Number of selected pixels with mask: {np.sum(cand_mask1)} with ADI thresh: {tcohv}")

                fig = plt.figure(figsize=(plotwidth, plotheight))
                ax = fig.add_subplot()
                ax.imshow(mask_valid_area, cmap=cmc.cm.cmaps["grayC"], alpha=0.5, zorder=10, vmin=0, vmax=1, aspect='auto')
                bmap_obj.plot(ax=ax, logger=logger)
                coord_xy = np.array(np.where(cand_mask1)).transpose()
                val = np.ones_like(cand_mask1)
                sc = ax.scatter(coord_xy[:, 1], coord_xy[:, 0], c=val[cand_mask1], s=1, cmap=cmc.cm.cmaps["lajolla_r"],
                                vmin=1, vmax=2)  # set min, max to ensure that points are yellow
                fig.set_constrained_layout(True)
                plt.title(f"Mask for 1OP - {proxy_id}")
                fig.savefig(join(qpath, "pic", f"mask_{proxy_id}.png"), dpi=300)
                plt.close(fig)

    elif qselection == 'both':
        if os.path.isfile(adipath):
            shutil.copy(adipath, join(qpath, "amplitude_dispersion.h5"))
        else:
            raise Exception(f"{adipath} is not a file")

        if len(adithresh) <= 0:
            raise Exception("No adi thresholds values given")
        if len(tcohthresh) <= 0:
            raise Exception("No temporal coherence thresholds values given")

        for tcohv in tcohthresh:
            for adiv in adithresh:
                proxy_value_adi = int(adiv * 100)
                proxy_value_tcoh = int(tcohv * 100)
                proxy_id = f"adi{proxy_value_adi}_tcoh{proxy_value_tcoh}"
                cand_mask1 = selectPixelsWithADIandTCOH(path=qpath,
                                                        thrsh_adi=adiv,
                                                        thresh_tcoh=tcohv,
                                                        grid_size=gridsize, bool_plot=plotflag,
                                                        plotwidth=plotwidth, plotheight=plotheight,
                                                        logger=logger)
                print(f"Number of selected pixels: {np.sum(cand_mask1)} with temporal coherence: {tcohv} and ADI: {adiv}")

            if maskfile:
                cand_mask1 &= mask_valid_area
                print(f"Number of selected pixels with mask: {np.sum(cand_mask1)} with temporal coherence: {tcohv} and ADI: {adiv}")

                fig = plt.figure(figsize=(plotwidth, plotheight))
                ax = fig.add_subplot()
                ax.imshow(mask_valid_area, cmap=cmc.cm.cmaps["grayC"], alpha=0.5, zorder=10, vmin=0, vmax=1, aspect='auto')
                bmap_obj.plot(ax=ax, logger=logger)
                coord_xy = np.array(np.where(cand_mask1)).transpose()
                val = np.ones_like(cand_mask1)
                sc = ax.scatter(coord_xy[:, 1], coord_xy[:, 0], c=val[cand_mask1], s=1, cmap=cmc.cm.cmaps["lajolla_r"],
                                vmin=1, vmax=2)  # set min, max to ensure that points are yellow
                fig.set_constrained_layout(True)
                plt.title(f"Mask for 1OP - {proxy_id}")
                fig.savefig(join(qpath, "pic", f"mask_{proxy_id}.png"), dpi=300)
                plt.close(fig)


def createParser():
    """Seletion of points based on TCOH or ADI"""
    EXAMPLE = """
        select_points_for_testing.py -p sbas -s both -tcohv 0.9 0.8 -adiv 0.2 0.25 -plot -adipath sbas/adi.h5
    """
    parser = argparse.ArgumentParser(
        description='Point selections \n\n',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=EXAMPLE)


    parser.add_argument('-p', dest="path", default=None, type=str, required=True, help="Path to project, where temporal_coherence.h5 and/or amplitude_dispersion are located")
    parser.add_argument('-s', dest="selection", choices=["tcoh", "adi", "both"], default="tcoh", help="Quality selection type")
    parser.add_argument('-tcohv', dest="tcohthresh", nargs="*", type=float, help="Thersholds for temporal coherence. -tcohv: 0.9 0.8 0.7")
    parser.add_argument('-adiv', dest="adithresh", nargs="*", type=float, help="Thersholds for ADI. -tcohv: 0.2 0.25 0.30")
    parser.add_argument('-gs', dest='gridsize', type=int, default=0, help='Grid size, default: 0')
    parser.add_argument('-pw', dest='plotwidth', type=int, default=8, help='Plot width, default: 8')
    parser.add_argument('-ph', dest='plotheight', type=int, default=8, help='Plot height, default: 8')
    parser.add_argument('-adipath', dest="adipath", default=None, type=str, help="Pointing to a specific adi file")
    parser.add_argument('-m', dest="maskpath", default=None, type=str, help="Path to mask file")
    parser.add_argument('-plot', dest='plotflag', action="store_true", default=False, help='Plot flag for selection function')
    return parser.parse_args()


if __name__ == '__main__':
    main()
