#!/usr/bin/env python

# SARvey - A multitemporal InSAR time series tool for the derivation of displacements.
#
# Copyright (C) 2021-2025 Andreas Piter (IPI Hannover, piter@ipi.uni-hannover.de)
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

"""Unwrapping module for SARvey."""
import multiprocessing
from os.path import join, dirname
import time
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from kamui import unwrap_arbitrary
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import structural_rank
from scipy.sparse.linalg import lsqr
from scipy.optimize import minimize
from logging import Logger
import cmcrameri as cmc

from mintpy.utils import ptime

import sarvey.utils as ut
from sarvey.ifg_network import IfgNetwork
from sarvey.objects import Network, NetworkParameter, AmplitudeImage


def objFuncTemporalCoherence_t(x, *args):
    """Compute temporal coherence from parameters and phase. To be used as objective function for optimization.

    Parameters
    ----------
    x: np.ndarray
        Search space for the DEM error in a 1D grid.
    args: tuple
        Additional arguments: (design_mat, obs_phase, scale_vel, scale_demerr).

    Returns
    -------
    1 - gamma: float
    """
    (design_mat, obs_phase, scale_vel, scale_demerr, scale_tcoef) = args

    # equalize the gradients in both directions
    x[0] *= scale_demerr
    x[1] *= scale_vel
    x[2] *= scale_tcoef

    pred_phase = np.matmul(design_mat, x)
    res = (obs_phase - pred_phase.T).ravel()
    gamma = np.abs(np.mean(np.exp(1j * res)))
    return 1 - gamma


def findOptimum(*, obs_phase: np.ndarray, design_mat: np.ndarray, val_range: np.ndarray):
    """Find optimal value within a one dimensional search space that fits to the observed phase.

    Parameters
    ----------
    obs_phase: np.ndarray
        Observed phase of the arc.
    design_mat: np.ndarray
        Design matrix for estimating parameters from arc phase.
    val_range: np.ndarray
        Range of possible values for the solution. Can be either for DEM error or velocity.

    Returns
    -------
    opt_val: scipy.optimize.minimize return value
    gamma: float
    pred_phase: np.ndarray
    """
    pred_phase = design_mat[:, np.newaxis] * val_range[np.newaxis, :]  # broadcasting
    if len(obs_phase.shape) == 2:
        # step densification
        res = obs_phase[:, np.newaxis, :] - pred_phase.T
        res = np.moveaxis(res, 0, 1)
        res = res.reshape((pred_phase.shape[1], -1))  # combine residuals from all arcs
    else:
        # step consistency check
        res = obs_phase - pred_phase.T

    gamma = np.abs(np.mean(np.exp(1j * res), axis=1))
    max_idx = np.argmax(gamma)
    opt_val = val_range[max_idx]
    return opt_val, gamma[max_idx], pred_phase[:, max_idx]


def oneDimSearchTemporalCoherence_t(*, demerr_range: np.ndarray, vel_range: np.ndarray, tcoef_range: np.ndarray, obs_phase: np.ndarray,
                                  design_mat: np.ndarray):
    """One dimensional search for maximum temporal coherence that fits the observed arc phase.

    Parameters
    ----------
    demerr_range: np.ndarray
        Search space for the DEM error in a 1D grid.
    vel_range: np.ndarray
        Search space for the velocity in a 1D grid.
    design_mat: np.ndarray
        Design matrix for estimating parameters from arc phase.
    obs_phase: np.ndarray
        Observed phase of the arc.

    Returns
    -------
    demerr: float
    vel: float
    gamma: float
    """
    demerr, gamma_demerr, pred_phase_demerr = findOptimum(
        obs_phase=obs_phase,
        design_mat=design_mat[:, 0],
        val_range=demerr_range
    )

    vel, gamma_vel, pred_phase_vel = findOptimum(
        obs_phase=obs_phase,
        design_mat=design_mat[:, 1],
        val_range=vel_range
    )

    tcoef, gamma_tcoef, pred_phase_tcoef = findOptimum(
        obs_phase=obs_phase,
        design_mat=design_mat[:, 2],
        val_range=tcoef_range
    )



    if gamma_vel > gamma_demerr:
        demerr, gamma_demerr, pred_phase_demerr = findOptimum(
            obs_phase=obs_phase - pred_phase_vel,
            design_mat=design_mat[:, 0],
            val_range=demerr_range
        )
        vel, gamma_vel, pred_phase_vel = findOptimum(
            obs_phase=obs_phase - pred_phase_demerr,
            design_mat=design_mat[:, 1],
            val_range=vel_range
        )

        # refine temp coef search
        tcoef, gamma_tcoef, pred_phase_tcoef = findOptimum(
            obs_phase=obs_phase - (pred_phase_demerr),
            design_mat=design_mat[:, 2],
            val_range=tcoef_range)
        
    elif gamma_demerr < gamma_vel:
        vel, gamma_vel, pred_phase_vel = findOptimum(
            obs_phase=obs_phase - pred_phase_demerr,
            design_mat=design_mat[:, 1],
            val_range=vel_range
        )
        demerr, gamma_demerr, pred_phase_demerr = findOptimum(
            obs_phase=obs_phase - pred_phase_vel,
            design_mat=design_mat[:, 0],
            val_range=demerr_range
        )

        # refine temp coef search
        tcoef, gamma_tcoef, pred_phase_tcoef = findOptimum(
            obs_phase=obs_phase - (pred_phase_vel),
            design_mat=design_mat[:, 2],
            val_range=tcoef_range)
    

    # improve initial estimate with gradient descent approach
    scale_demerr = demerr_range.max()
    scale_vel = vel_range.max()
    scale_tcoef = tcoef_range.max()

    demerr, vel, tcoef, gamma = gradientSearchTemporalCoherence_t(
        scale_vel=scale_vel,
        scale_demerr=scale_demerr,
        scale_tcoef=scale_tcoef,
        obs_phase=obs_phase,
        design_mat=design_mat,
        x0=np.array([demerr/scale_demerr, vel/scale_vel, tcoef/scale_tcoef]).T
    )

    pred_phase = np.matmul(design_mat, np.array([demerr, vel, tcoef]))
    res = (obs_phase - pred_phase.T).ravel()
    gamma = np.abs(np.mean(np.exp(1j * res)))
    return demerr, vel, tcoef, gamma


def gradientSearchTemporalCoherence_t(*, scale_vel: float, scale_demerr: float, scale_tcoef: float, obs_phase: np.ndarray,
                                    design_mat: np.ndarray, x0: np.ndarray):
    """GradientSearchTemporalCoherence.

    Parameters
    ----------
    scale_demerr: float
        Scaling factor for DEM error to equalize the axis of the search space.
    scale_vel: float
        Scaling factor for velocity to equalize the axis of the search space.
    design_mat: np.ndarray
        Design matrix for estimating parameters from arc phase.
    obs_phase: np.ndarray
        Observed phase of the arc.
    x0: np.ndarray
        Initial values for optimization.

    Returns
    -------
    demerr: float
    vel: float
    gamma: float
    """
    opt_res = minimize(
        objFuncTemporalCoherence_t,
        x0,
        args=(design_mat, obs_phase, scale_vel, scale_demerr, scale_tcoef),
        bounds=((-1, 1), (-1, 1), (-1, 1)),
        method='L-BFGS-B'
    )
    gamma = 1 - opt_res.fun
    demerr = opt_res.x[0] * scale_demerr
    vel = opt_res.x[1] * scale_vel
    tcoef = opt_res.x[2] * scale_tcoef
    return demerr, vel, tcoef, gamma


def launchAmbiguityFunctionSearch_t(parameters: tuple):
    """Wrap for launching ambiguity function for temporal unwrapping in parallel.

    Parameters
    ----------
    parameters: tuple
        Arguments for temporal unwrapping in parallel.

    Returns
    -------
    arc_idx_range: np.ndarray
    demerr: np.ndarray
    vel: np.ndarray
    gamma: np.ndarray
    """
    (arc_idx_range, num_arcs, phase, slant_range, loc_inc, ifg_net_obj, wavelength, 
     velocity_bound, demerr_bound, tcoef_bound, num_samples) = parameters

    demerr = np.zeros((num_arcs, 1), dtype=np.float32)
    vel = np.zeros((num_arcs, 1), dtype=np.float32)
    tcoef = np.zeros((num_arcs, 1), dtype=np.float32)
    gamma = np.zeros((num_arcs, 1), dtype=np.float32)

    design_mat = np.zeros((ifg_net_obj.num_ifgs, 3), dtype=np.float32)

    demerr_range = np.linspace(-demerr_bound, demerr_bound, num_samples)
    vel_range = np.linspace(-velocity_bound, velocity_bound, num_samples)
    tcoef_range = np.linspace(-tcoef_bound, tcoef_bound, num_samples)

    # prog_bar = ptime.progressBar(maxValue=num_arcs)

    factor = 4 * np.pi / wavelength

    for k in range(num_arcs):
        design_mat[:, 0] = factor * ifg_net_obj.pbase_ifg / (slant_range[k] * np.sin(loc_inc[k]))
        design_mat[:, 1] = factor * ifg_net_obj.tbase_ifg
        design_mat[:, 2] = factor * ifg_net_obj.temperatures_ifg

        demerr[k], vel[k], tcoef[k], gamma[k] = oneDimSearchTemporalCoherence_t(
            demerr_range=demerr_range,
            vel_range=vel_range,
            tcoef_range=tcoef_range,
            obs_phase=phase[k, :],
            design_mat=design_mat
        )

    return arc_idx_range, demerr, vel, tcoef, gamma


def temporalUnwrapping_t(*, ifg_net_obj: IfgNetwork, net_obj: Network,  wavelength: float, velocity_bound: float,
                       demerr_bound: float, coef_bound: float, num_samples: int, num_cores: int = 1, logger: Logger) -> \
        tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve ambiguities for every arc in spatial Network object.

    Parameters
    ----------
    ifg_net_obj: IfgNetwork
        The IfgNetwork object.
    net_obj: Network
        The Network object.
    wavelength: float
        The wavelength.
    velocity_bound: float
        The velocity bound.
    demerr_bound: float
        The DEM error bound.
    num_samples: int
        The number of samples for the search space.
    num_cores: int
        Number of cores to be used. Default is 1.
    logger: Logger
        Logging handler.

    Returns
    -------
    demerr: np.ndarray
    vel: np.ndarray
    coef: np.arry
    gamma: np.ndarray
    """
    msg = "#" * 10
    msg += " TEMPORAL UNWRAPPING: AMBIGUITY FUNCTION "
    msg += "#" * 10
    logger.info(msg=msg)

    start_time = time.time()

    if num_cores == 1:
        args = (
            np.arange(net_obj.num_arcs), net_obj.num_arcs, net_obj.phase,
            net_obj.slant_range, net_obj.loc_inc, ifg_net_obj, wavelength, velocity_bound, demerr_bound, coef_bound, num_samples)
        arc_idx_range, demerr, vel, tcoef, gamma = launchAmbiguityFunctionSearch_t(parameters=args)
    else:
        logger.info(msg="start parallel processing with {} cores.".format(num_cores))

        demerr = np.zeros((net_obj.num_arcs, 1), dtype=np.float32)
        vel = np.zeros((net_obj.num_arcs, 1), dtype=np.float32)
        tcoef = np.zeros((net_obj.num_arcs, 1), dtype=np.float32)
        gamma = np.zeros((net_obj.num_arcs, 1), dtype=np.float32)


        num_cores = net_obj.num_arcs if num_cores > net_obj.num_arcs else num_cores  # avoids having more samples then
        # cores
        idx = ut.splitDatasetForParallelProcessing(num_samples=net_obj.num_arcs, num_cores=num_cores)

        args = [(
            idx_range,
            idx_range.shape[0],
            net_obj.phase[idx_range, :],
            net_obj.slant_range[idx_range],
            net_obj.loc_inc[idx_range],
            ifg_net_obj,
            wavelength,
            velocity_bound,
            demerr_bound,
            coef_bound,
            num_samples) for idx_range in idx]

        with multiprocessing.Pool(processes=num_cores) as pool:
            results = pool.map(func=launchAmbiguityFunctionSearch_t, iterable=args)

        # retrieve results
        for i, demerr_i, vel_i, tcoef_i, gamma_i in results:
            demerr[i] = demerr_i
            vel[i] = vel_i
            tcoef[i] = tcoef_i
            gamma[i] = gamma_i

    m, s = divmod(time.time() - start_time, 60)
    logger.info(msg="Finished temporal unwrapping.")
    logger.debug(msg='time used: {:02.0f} mins {:02.1f} secs.'.format(m, s))
    return demerr, vel, tcoef, gamma


def parameterBasedNoisyPointRemoval_t(*, net_par_obj: NetworkParameter, point_id: np.ndarray, coord_xy: np.ndarray,
                                    design_mat: np.ndarray, rmse_thrsh: float = 0.02, num_points_remove: int = 1,
                                    bmap_obj: AmplitudeImage = None, bool_plot: bool = False,
                                    logger: Logger):
    """Remove Points during spatial integration step if residuals at many connected arcs are high.

    The idea is similar to outlier removal in DePSI, but without hypothesis testing.
    It can be used as a preprocessing step to spatial integration.
    The points are removed based on the RMSE computed from the residuals of the parameters (DEM error, velocity) per
    arc. The point with the highest RMSE is removed in each iteration. The process stops when the maximum RMSE is below
    a threshold.


    Parameters
    ----------
    net_par_obj: NetworkParameter
        The spatial NetworkParameter object containing the parameters estimates at each arc.
    point_id: np.ndarray
        ID of the points in the network.
    coord_xy: np.ndarray
        Radar coordinates of the points in the spatial network.
    design_mat: np.ndarray
        Design matrix describing the relation between arcs and points.
    rmse_thrsh: float
        Threshold for the RMSE of the residuals per point. Default = 0.02.
    num_points_remove: int
        Number of points to remove in each iteration. Default = 1.
    bmap_obj: AmplitudeImage
        Basemap object for plotting. Default = None.
    bool_plot: bool
        Plot the RMSE per point. Default = False.
    logger: Logger
        Logging handler.

    Returns
    -------
    spatial_ref_id: int
        ID of the spatial reference point.
    point_id: np.ndarray
        ID of the points in the network without the removed points.
    net_par_obj: NetworkParameter
        The NetworkParameter object without the removed points.
    """
    msg = "#" * 10
    msg += " NOISY POINT REMOVAL BASED ON ARC PARAMETERS "
    msg += "#" * 10
    logger.info(msg=msg)

    num_points = point_id.shape[0]

    logger.info(msg="Selection of the reference PSC")
    # select one of the two pixels which are connected via the arc with the highest quality
    spatial_ref_idx = np.where(design_mat[np.argmax(net_par_obj.gamma), :] != 0)[0][0]
    coord_xy = np.delete(arr=coord_xy, obj=spatial_ref_idx, axis=0)
    spatial_ref_id = point_id[spatial_ref_idx]
    point_id = np.delete(arr=point_id, obj=spatial_ref_idx, axis=0)
    num_points -= 1

    # remove reference point from design matrix
    design_mat = net_par_obj.gamma * np.delete(arr=design_mat, obj=spatial_ref_idx, axis=1)

    logger.info(msg="Spatial integration to detect noisy point")
    start_time = time.time()

    it_count = 0
    while True:
        logger.info(msg="ITERATION: {}".format(it_count))
        design_mat = csr_matrix(design_mat)

        if structural_rank(design_mat) < design_mat.shape[1]:
            logger.error(msg="Singular normal matrix. Network is no longer connected!")
            # point_id = np.sort(np.hstack([spatial_ref_id, point_id]))
            # return spatial_ref_id, point_id, net_par_obj
            raise ValueError
        # demerr
        obv_vec = net_par_obj.demerr.reshape(-1, )
        demerr_points = lsqr(design_mat.toarray(), obv_vec * net_par_obj.gamma.reshape(-1, ))[0]
        r_demerr = obv_vec - np.matmul(design_mat.toarray(), demerr_points)

        # vel
        obv_vec = net_par_obj.vel.reshape(-1, )
        vel_points = lsqr(design_mat.toarray(), obv_vec * net_par_obj.gamma.reshape(-1, ))[0]
        r_vel = obv_vec - np.matmul(design_mat.toarray(), vel_points)

        # tcoef
        obv_vec = net_par_obj.temperature.reshape(-1, )
        tcoef_points = lsqr(design_mat.toarray(), obv_vec * net_par_obj.gamma.reshape(-1, ))[0]
        r_tcoef = obv_vec - np.matmul(design_mat.toarray(), tcoef_points)

        rmse_demerr = np.zeros((num_points,))
        rmse_vel = np.zeros((num_points,))
        rmse_tcoef = np.zeros((num_points,))

        for p in range(num_points):
            r_mask = design_mat[:, p].toarray() != 0
            rmse_demerr[p] = np.sqrt(np.mean(r_demerr[r_mask.ravel()].ravel() ** 2))
            rmse_vel[p] = np.sqrt(np.mean(r_vel[r_mask.ravel()].ravel() ** 2))
            rmse_tcoef[p] = np.sqrt(np.mean(r_tcoef[r_mask.ravel()].ravel() ** 2))

        rmse = rmse_vel.copy()
        max_rmse = np.max(rmse.ravel())
        logger.info(msg="Maximum RMSE DEM correction: {:.2f} m".format(np.max(rmse_demerr.ravel())))
        logger.info(msg="Maximum RMSE velocity: {:.4f} m / year".format(np.max(rmse_vel.ravel())))
        logger.info(msg="Maximum RMSE Temp Coefficient: {:.4f}".format(np.max(rmse_tcoef.ravel())))

        if bool_plot:
            # vel
            ax = bmap_obj.plot(logger=logger)
            sc = ax.scatter(coord_xy[:, 1], coord_xy[:, 0], c=rmse_vel * 1000, s=3.5,
                            cmap=cmc.cm.cmaps["lajolla"], vmin=0, vmax=rmse_thrsh * 1000)
            plt.colorbar(sc, pad=0.03, shrink=0.5)
            ax.set_title("{}. iteration\nmean velocity - RMSE per point in [mm / year]".format(it_count))
            fig = ax.get_figure()
            plt.tight_layout()
            fig.savefig(join(dirname(net_par_obj.file_path), "pic", f"step_1_rmse_vel_{it_count}th_iter.png"),
                        dpi=300)
            plt.close(fig)

            # demerr
            ax = bmap_obj.plot(logger=logger)
            sc = ax.scatter(coord_xy[:, 1], coord_xy[:, 0], c=rmse_demerr, s=3.5,
                            cmap=cmc.cm.cmaps["lajolla"])
            plt.colorbar(sc, pad=0.03, shrink=0.5)
            ax.set_title("{}. iteration\nDEM correction - RMSE per point in [m]".format(it_count))
            fig = ax.get_figure()
            plt.tight_layout()
            fig.savefig(join(dirname(net_par_obj.file_path), "pic",
                             f"step_1_rmse_dem_correction_{it_count}th_iter.png"),
                        dpi=300)
            plt.close(fig)

            # tcoef
            ax = bmap_obj.plot(logger=logger)
            sc = ax.scatter(coord_xy[:, 1], coord_xy[:, 0], c=rmse_tcoef, s=3.5,
                            cmap=cmc.cm.cmaps["lajolla"])
            plt.colorbar(sc, pad=0.03, shrink=0.5)
            ax.set_title("{}. iteration\nTemp Coef - RMSE per point in [m]".format(it_count))
            fig = ax.get_figure()
            plt.tight_layout()
            fig.savefig(join(dirname(net_par_obj.file_path), "pic",
                             f"step_1_rmse_tcoef_{it_count}th_iter.png"),
                        dpi=300)
            plt.close(fig)

        if max_rmse <= rmse_thrsh:
            logger.info(msg="No noisy pixels detected.")
            break

        # remove point with highest rmse
        p_mask = np.ones((num_points,), dtype=np.bool_)
        p_mask[np.argsort(rmse)[::-1][:num_points_remove]] = False  # see description of function removeArcsByPointMask
        net_par_obj, point_id, coord_xy, design_mat = removeArcsByPointMask(net_obj=net_par_obj, point_id=point_id,
                                                                            coord_xy=coord_xy, p_mask=p_mask,
                                                                            design_mat=design_mat.toarray(),
                                                                            logger=logger)
        num_points -= num_points_remove
        it_count += 1

    m, s = divmod(time.time() - start_time, 60)
    logger.debug(msg='time used: {:02.0f} mins {:02.1f} secs.'.format(m, s))

    # add spatialRefIdx back to point_id
    point_id = np.sort(np.hstack([spatial_ref_id, point_id]))
    return spatial_ref_id, point_id, net_par_obj


def removeArcsByPointMask(*, net_obj: Union[Network, NetworkParameter], point_id: np.ndarray, coord_xy: np.ndarray,
                          p_mask: np.ndarray, design_mat: np.ndarray,
                          logger: Logger) -> tuple[Network, np.ndarray, np.ndarray, np.ndarray]:
    """Remove all entries related to the arc observations connected to the points which have a False value in p_mask.

    Parameters
    ----------
    net_obj: Network
        The Network object.
    point_id: np.ndarray
        ID of the points in the network.
    coord_xy: np.ndarray
        Radar coordinates of the points in the spatial network.
    p_mask: np.ndarray
        Boolean mask with True for points to keep, and False for points to remove.
    design_mat: np.ndarray
        Design matrix describing the relation between arcs and points.
    logger: Logger
        Logging handler.

    Returns
    -------
    net_obj: Network
        Network object without the removed arcs and points.
    point_id: np.ndarray
        ID of the points in the network without the removed points.
    coord_xy: np.ndarray
        Radar coordinates of the points in the spatial network without the removed points.
    design_mat: np.ndarray
        Design matrix describing the relation between arcs and points without the removed points and arcs.
    """
    # find respective arcs
    a_idx = list()
    for p_idx in np.where(~p_mask)[0]:
        a_idx.append(np.where(design_mat[:, p_idx] != 0)[0])

    if len(a_idx) != 0:
        a_idx = np.hstack(a_idx)
        a_mask = np.ones((net_obj.num_arcs,), dtype=np.bool_)
        a_mask[a_idx] = False
        net_obj.removeArcs(mask=a_mask)
        design_mat = design_mat[a_mask, :]
    else:
        a_idx = np.array(a_idx)  # so I can check the size

    # remove psPoints
    point_id = point_id[p_mask]
    design_mat = design_mat[:, p_mask]
    coord_xy = coord_xy[p_mask, :]

    # beside removing the arcs in "arcs", the tuple indices have to be changed to make them fit to new point indices
    for p_idx in np.sort(np.where(~p_mask)[0])[::-1]:
        net_obj.arcs[np.where((net_obj.arcs[:, 0] > p_idx)), 0] -= 1
        net_obj.arcs[np.where((net_obj.arcs[:, 1] > p_idx)), 1] -= 1

    logger.info(msg="Removed {} arc(s) connected to the removed point(s)".format(a_idx.size))
