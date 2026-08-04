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


import sarvey.utils as ut
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
    gridsize = args.gridsize
    plotwidth = args.plotwidth
    plotheight = args.plotheight
    adipath = args.adipath
    plotflag = args.plotflag

    logger = Logger("log")


    if qselection == 'tcoh':
        for tcohv in tcohthresh:
            cand_mask1 = selectPixels(path=qpath, selection_method="temp_coh", thrsh=tcohv,
                grid_size=gridsize, bool_plot=plotflag, plotwidth=plotwidth, plotheight=plotheight, logger=logger)
            
            print(f"Number of selected pixels: {np.sum(cand_mask1)} with thresh: {tcohv}")

    elif qselection == 'adi':
        if os.path.isfile(adipath):
            shutil.copy(adipath, join(qpath, "amplitude_dispersion.h5"))
        else:
            raise Exception(f"{adipath} is not a file")
        for adiv in adithresh:
            cand_mask1 = selectPixels(path=qpath, selection_method="adi", thrsh=adiv,
                grid_size=gridsize, bool_plot=plotflag,plotwidth=plotwidth, plotheight=plotheight, logger=logger)
            
            print(f"Number of selected pixels: {np.sum(cand_mask1)} with thresh: {tcohv}")

    elif qselection == 'both':
        if os.path.isfile(adipath):
            shutil.copy(adipath, join(qpath, "amplitude_dispersion.h5"))
        else:
            raise Exception(f"{adipath} is not a file")

        for tcohv in tcohthresh:
            for adiv in adithresh:
                cand_mask1 = selectPixelsWithADIandTCOH(path=qpath,
                                                        thrsh_adi=adiv,
                                                        thresh_tcoh=tcohv,
                                                        grid_size=gridsize, bool_plot=plotflag,
                                                        plotwidth=plotwidth, plotheight=plotheight,
                                                        logger=logger)
                print(f"Number of selected pixels: {np.sum(cand_mask1)} with temporal coherence: {tcohv} and ADI: {adiv}")

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
    parser.add_argument('-plot', dest='plotflag', type=bool, default=False, help='Plot flag for selection function')
    return parser.parse_args()


if __name__ == '__main__':
    main()
