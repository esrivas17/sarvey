#!/usr/bin/env python

import argparse
import time
import os
from os.path import join, basename, dirname
import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import logging
from logging import Logger
import sys
import cmcrameri as cmc

from miaplpy.objects.slcStack import slcStack

from sarvey.objects import Points, AmplitudeImage
from sarvey import console
from sarvey import viewer
from sarvey.config import loadConfiguration
import sarvey.utils as ut
from sarvey.viewer import selectNeighbourhood

from scipy.spatial import KDTree
import datetime

from GWL_Network import GWL_Network

EXAMPLE = """
    gwl_taipei.py sbas/p2_coh_ts.h5 gwl_stations.h5 --raref 1500 2000
"""
scale_dict = {"mm": 1000, "cm": 100, "dm": 10, "m": 1}

def main(iargs=None):
    parser = createParser()
    args = parser.parse_args(iargs)
    if args.workdir is None:
        args.workdir = os.getcwd()

    args.input_file = join(args.workdir, args.input_file)
    config_file_path = os.path.abspath(join(args.workdir, dirname(args.input_file), "config.json"))
    config = loadConfiguration(path=config_file_path)

    # whatever logger
    logger = logging.getLogger(__name__)

    point_obj = Points(file_path=args.input_file, logger=logger)
    point_obj.open(input_path=config.general.input_path)

    # trees
    tree = KDTree(point_obj.coord_xy)
    tree_utm = KDTree(point_obj.coord_utm)

    if point_obj.ifg_net_obj.dates is not None:
        times = [datetime.date.fromisoformat(date) for date in point_obj.ifg_net_obj.dates]
    else:  # backwards compatible, if ifg_net_obj does not contain dates
        times = point_obj.ifg_net_obj.tbase

    # Reference
    x, y = args.raref
    idx = tree.query([y, x])[-1]
    ts_point_idx = idx
    ts_refpoint_idx = idx
    
    vel, demerr, ref_atmo, coherence, omega, v_hat = ut.estimateParameters(obj=point_obj, ifg_space=False)

    # changing reference
    print("changed reference to ID: {}".format(point_obj.point_id[ts_refpoint_idx]))
    logger.info(msg="changed reference to ID: {}".format(point_obj.point_id[ts_refpoint_idx]))
    point_obj.phase -= point_obj.phase[ts_refpoint_idx, :]
    vel, demerr, ref_atmo, coherence, omega, v_hat = ut.estimateParameters(obj=point_obj, ifg_space=False)

    # GWL
    gwl_data = GWL_Network(args.gwl_network, mode='h5', startdate=times[0], enddate=times[-1])
    gwl_xy = gwl_data.get_radar_coords(os.path.join(config.general.input_path, "geometryRadar.h5"))
    #gwl_xs, gwl_ys = np.array(gwl_xy).T

    for station in gwl_data.stations:
        print(f'Station: {station.name}')
        mean_ts = None
        gwlx, gwly = station.ll2xy(os.path.join(config.general.input_path, "geometryRadar.h5"))

        idx = tree.query([gwly, gwlx])[-1]
        gwl_ix = gwl_data.kdtree.query([x, y], k=4)
        neighb_idxs = tree_utm.query_ball_point(point_obj.coord_utm[idx, :], r=args.radius)

        print(f'Num of neighbours: {len(neighb_idxs)}')

        # average of every single neighbour
        if len(neighb_idxs) == 1:
            mean_ts = get_timeseries(point_obj, neighb_idxs[0], vel, demerr, args.scale, False)
            mean_inc = point_obj.loc_inc[neighb_idxs[0]]
            mean_ts = mean_ts*np.cos(np.deg2rad(mean_inc))
        elif len(neighb_idxs) > 1:
            ts_list = list()

            for neighix in neighb_idxs:
                ts = get_timeseries(point_obj, neighix, vel, demerr, args.scale, False)
                inc = point_obj.loc_inc[neighix]
                ts_pseudo = ts*np.cos(np.deg2rad(inc))
                ts_list.append(ts_pseudo)
            ts_array = np.vstack(ts_list)
            mean_ts = np.mean(ts_array, axis=0)
        
        insarts = np.array([mean_ts, times]).T
        station.insarts(insarts, args.savedir, False)
        

    
def get_timeseries(point_obj, point_idx, vel, demerr, scale, flag_demerr=False):
    """Prepare phase time series for plotting."""
        # transform phase time series into meters
    resulting_ts = point_obj.wavelength / (4 * np.pi) * point_obj.phase[point_idx, :]
    # Displacement
    resulting_ts = resulting_ts - point_obj.ifg_net_obj.tbase * vel[point_idx]
    if flag_demerr:  # DEM error
        phase_topo = (point_obj.ifg_net_obj.pbase / (point_obj.slant_range[point_idx] * np.sin(point_obj.loc_inc[point_idx])) * demerr[point_idx])
        resulting_ts = resulting_ts - phase_topo

    return resulting_ts * scale_dict[scale]

def createParser():
    """Create_parser."""
    parser = argparse.ArgumentParser(
        description='Plot results from MTI\n\n',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=EXAMPLE)

    parser.add_argument('input_file', help='Path to the input file')
    parser.add_argument('gwl_network', type=str, help='Path to GWL data saved as a h5 file')
    parser.add_argument('--raref', dest='raref', nargs=2, default=None, required=True, 
                        help='Range and azimuth reference. X,Y')
    parser.add_argument('--radius', dest='radius', required=True, type=float, help='Radius for each GW well')
    parser.add_argument('--scale', dest='scale', choices=["mm", "dm", "cm", "m"], default="mm", help="displacement scale")
    parser.add_argument('-s', '--savedir', dest='savedir', required=True, type=str, help='Directory to save figures')
    parser.add_argument('-w', '--workdir', default=None, dest="workdir",
                        help='Working directory (default: current directory).')
    
    
    return parser

if __name__ == "__main__":
    main()
