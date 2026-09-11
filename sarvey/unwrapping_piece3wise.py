#!/usr/bin/env python

# SARvey - A multitemporal InSAR time series tool for the derivation of displacements.
#
# Copyright (C) 2021-2026 Andreas Piter (IPI Hannover, piter@ipi.uni-hannover.de)
#
# This software was developed together with FERN.Lab (fernlab@gfz.de) in the context
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
import time
import networkx as nx
import numpy as np
from kamui import unwrap_arbitrary
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import structural_rank
from scipy.sparse.linalg import lsqr
from scipy.optimize import minimize
from logging import Logger

from mintpy.utils import ptime

import sarvey.utils as ut
from sarvey.ifg_network_piecewise_three import IfgNetwork3Piecewise
from sarvey.ifg_network_piecewise import IfgNetworkPiecewise

from sarvey.objects import NetworkParameter, NetworkPiecewise, NetworkParameterPiecewise, Network3Piecewise, NetworkParameter3Piecewise


def objFuncTemporalCoherence(x, *args):
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
    (design_mat, obs_phase, scale_vel, scale_demerr) = args

    # equalize the gradients in both directions
    x[0] *= scale_demerr
    x[1] *= scale_vel

    pred_phase = np.matmul(design_mat, x)
    res = (obs_phase - pred_phase.T).ravel()
    gamma = np.abs(np.mean(np.exp(1j * res)))
    return 1 - gamma


def gridSearchTemporalCoherence(*, demerr_grid: np.ndarray, vel_grid: np.ndarray, design_mat: np.ndarray,
                                obs_phase: np.ndarray):
    """Grid search which maximizes the temporal coherence as the objective function.

    Parameters
    ----------
    demerr_grid: np.ndarray
        Search space for the DEM error in a 2D grid.
    vel_grid: np.ndarray
        Search space for the velocity in a 2D grid.
    design_mat: np.ndarray
        Design matrix for estimating parameters from arc phase.
    obs_phase: np.ndarray
        Observed phase of the arc.

    Returns
    -------
    demerr: float
        estimated DEM error.
    vel: float
        estimated velocity.
    gamma: float
        estimated temporal coherence.
    """
    demerr_grid_flat = demerr_grid.flatten()
    vel_grid_flat = vel_grid.flatten()
    gamma_flat = np.array(
        [1 - objFuncTemporalCoherence(np.array([demerr_grid_flat[i], vel_grid_flat[i]]),
                                      design_mat, obs_phase, 1, 1)
         for i in range(demerr_grid_flat.shape[0])])
    gamma = gamma_flat.reshape(demerr_grid.shape)
    idx_max_gamma = np.argmax(gamma_flat)

    # return demerr_grid_flat[idx_max_gamma], vel_grid_flat[idx_max_gamma], gamma_flat[idx_max_gamma]
    return demerr_grid_flat[idx_max_gamma], vel_grid_flat[idx_max_gamma], gamma


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


def oneDimSearchTemporalCoherence(*, demerr_range: np.ndarray, vel_range: np.ndarray, obs_phase: np.ndarray,
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
    else:
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

    # improve initial estimate with gradient descent approach
    scale_demerr = demerr_range.max()
    scale_vel = np.max(np.abs(vel_range))

    demerr, vel, gamma = gradientSearchTemporalCoherence(
        scale_vel=scale_vel,
        scale_demerr=scale_demerr,
        obs_phase=obs_phase,
        design_mat=design_mat,
        x0=np.array([demerr / scale_demerr,
                     vel / scale_vel]).T
    )

    pred_phase = np.matmul(design_mat, np.array([demerr, vel]))
    res = (obs_phase - pred_phase.T).ravel()
    gamma = np.abs(np.mean(np.exp(1j * res)))
    return demerr, vel, gamma


def oneDimSearchTemporalCoherencePiecewise(*, demerr_range: np.ndarray, vel_range: np.ndarray, vel_excavation_range, obs_phase: np.ndarray, 
                                           design_mat: np.ndarray, design_mat_pre: np.ndarray, design_mat_exca: np.ndarray, ix_pre: list, ix_exca: list):
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

    vel_pre, gamma_vel_pre, pred_phase_vel_pre = findOptimum(
            obs_phase=obs_phase[ix_pre],
            design_mat=design_mat_pre[:, 1],
            val_range=vel_range)

    vel_exca, gamma_vel_exca, pred_phase_vel_exca = findOptimum(
            obs_phase=obs_phase[ix_exca],
            design_mat=design_mat_exca[:, 1],
            val_range=vel_excavation_range)
    
    pred_phase_vel_combined = np.zeros_like(design_mat[:, 1])
    pred_phase_vel_combined[ix_pre] = design_mat_pre[:,1] * vel_pre
    pred_phase_vel_combined[ix_exca] = design_mat_exca[:,1] * vel_exca
    residual = obs_phase - pred_phase_vel_combined.T
    gamma_vel_combined = np.abs(np.mean(np.exp(1j*residual)))


    if gamma_vel_combined > gamma_demerr:
        demerr, gamma_demerr, pred_phase_demerr = findOptimum(
            obs_phase=obs_phase - pred_phase_vel_combined,
            design_mat=design_mat[:, 0],
            val_range=demerr_range
        )

        obs_phase_reduced = obs_phase - pred_phase_demerr
        vel_pre, gamma_vel_pre, pred_phase_vel_pre = findOptimum(
            obs_phase=obs_phase_reduced[ix_pre],
            design_mat=design_mat_pre[:, 1],
            val_range=vel_range)

        vel_exca, gamma_vel_exca, pred_phase_vel_exca = findOptimum(
            obs_phase=obs_phase_reduced[ix_exca],
            design_mat=design_mat_exca[:, 1],
            val_range=vel_excavation_range)

    else:
        obs_phase_reduced = obs_phase - pred_phase_demerr

        vel_pre, gamma_vel_pre, pred_phase_vel_pre = findOptimum(
                    obs_phase=obs_phase_reduced[ix_pre],
                    design_mat=design_mat_pre[:, 1],
                    val_range=vel_range)
        
        vel_exca, gamma_vel_exca, pred_phase_vel_exca = findOptimum(
                    obs_phase=obs_phase_reduced[ix_exca],
                    design_mat=design_mat_exca[:, 1],
                    val_range=vel_excavation_range)
        
        pred_phase_vel_combined = np.zeros_like(design_mat[:, 1])
        pred_phase_vel_combined[ix_pre] = design_mat_pre[:,1] * vel_pre
        pred_phase_vel_combined[ix_exca] = design_mat_exca[:,1] * vel_exca
        
        demerr, gamma_demerr, pred_phase_demerr = findOptimum(
            obs_phase=obs_phase - pred_phase_vel_combined,
            design_mat=design_mat[:, 0],
            val_range=demerr_range
        )

    # improve initial estimate with gradient descent approach
    scale_demerr = demerr_range.max()
    scale_vel = np.max(np.abs(vel_range))
    scale_vel_exca = np.max(np.abs(vel_excavation_range))

    demerr_pre, vel_pre, gamma_pre = gradientSearchTemporalCoherence(
        scale_vel=scale_vel,
        scale_demerr=scale_demerr,
        obs_phase=obs_phase[ix_pre],
        design_mat=design_mat_pre,
        x0=np.array([demerr / scale_demerr,
                     vel_pre / scale_vel]).T)

    demerr_exca, vel_exca, gamma_exca = gradientSearchTemporalCoherence(
    scale_vel=scale_vel_exca,
    scale_demerr=scale_demerr,
    obs_phase=obs_phase[ix_exca],
    design_mat=design_mat_exca,
    x0=np.array([demerr / scale_demerr,
                    vel_pre / scale_vel]).T)

    if gamma_pre > gamma_exca:
        pred_phase_demerr = design_mat[:,0] * demerr_pre
    else:
        pred_phase_demerr = design_mat[:,0] * demerr_exca

    pred_phase_vel_combined = np.zeros_like(design_mat[:, 1])
    pred_phase_vel_combined[ix_pre] = design_mat_pre[:,1] * vel_pre
    pred_phase_vel_combined[ix_exca] = design_mat_exca[:,1] * vel_exca
    
    pred_phase = pred_phase_demerr + pred_phase_vel_combined
    res = (obs_phase - pred_phase.T).ravel()
    gamma = np.abs(np.mean(np.exp(1j * res)))
    return demerr, vel_pre, vel_exca, gamma


def gradientSearchTemporalCoherence(*, scale_vel: float, scale_demerr: float, obs_phase: np.ndarray,
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
        objFuncTemporalCoherence,
        x0,
        args=(design_mat, obs_phase, scale_vel, scale_demerr),
        bounds=((-1, 1), (-1, 1)),
        method='L-BFGS-B'
    )
    gamma = 1 - opt_res.fun
    demerr = opt_res.x[0] * scale_demerr
    vel = opt_res.x[1] * scale_vel
    return demerr, vel, gamma


def launchAmbiguityFunctionSearch(parameters: tuple):
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
    (arc_idx_range, num_arcs, phase, slant_range, loc_inc, ifg_net_obj, wavelength, velocity_bound, demerr_bound,
     num_samples) = parameters

    demerr = np.zeros((num_arcs, 1), dtype=np.float32)
    vel = np.zeros((num_arcs, 1), dtype=np.float32)
    gamma = np.zeros((num_arcs, 1), dtype=np.float32)

    design_mat = np.zeros((ifg_net_obj.num_ifgs, 2), dtype=np.float32)

    demerr_range = np.linspace(-demerr_bound, demerr_bound, num_samples)
    vel_range = np.linspace(-velocity_bound, velocity_bound, num_samples)
    prog_bar = ptime.progressBar(maxValue=num_arcs)

    factor = 4 * np.pi / wavelength
    every = max(1, num_arcs // 10)

    for k in range(num_arcs):
        design_mat[:, 0] = factor * ifg_net_obj.pbase_ifg / (slant_range[k] * np.sin(loc_inc[k]))
        design_mat[:, 1] = factor * ifg_net_obj.tbase_ifg

        demerr[k], vel[k], gamma[k] = oneDimSearchTemporalCoherence(
            demerr_range=demerr_range,
            vel_range=vel_range,
            obs_phase=phase[k, :],
            design_mat=design_mat
        )
        prog_bar.update(value=k + 1, every=every,
                        suffix='{}/{} arcs processed. '.format(k + 1, num_arcs))

    return arc_idx_range, demerr, vel, gamma


def temporalUnwrapping3Piecewise(*, ifg_net_obj: IfgNetwork3Piecewise, net_obj: Network3Piecewise, wavelength: float, velocity_bound: float, vel_excavation_bound: float, 
                                 demerr_bound: float, num_samples: int, num_cores: int = 1, logger: Logger):

    msg = "#" * 10
    msg += " TEMPORAL UNWRAPPING FOR TUNNELLING: AMBIGUITY FUNCTION "
    msg += "#" * 10
    logger.info(msg=msg)

    start_time = time.time()
    ix_pre = net_obj.ifg_net_obj.ix_ifg_pre
    ix_exca = net_obj.ifg_net_obj.ix_ifg_exca
    ix_conso = net_obj.ifg_net_obj.ix_ifg_conso

    if num_cores == 1:
        args = (np.arange(net_obj.num_arcs), net_obj.num_arcs, net_obj.phase, net_obj.phase[:, ix_pre], net_obj.phase[:,ix_exca], net_obj.phase[:,ix_conso],
            net_obj.slant_range, net_obj.loc_inc, ifg_net_obj, wavelength, velocity_bound, vel_excavation_bound, demerr_bound, num_samples)
        arc_idx_range, demerr, vel, gamma, demerr_pre, vel_pre, gamma_pre, demerr_exca, vel_exca, gamma_exca, demerr_conso, vel_conso, gamma_conso = launchAmbiguityFunctionSearchPiece3wise(parameters=args)
    else:
        logger.info(msg="start parallel processing with {} cores.".format(num_cores))

        demerr = np.zeros((net_obj.num_arcs, 1), dtype=np.float32)
        vel = np.zeros((net_obj.num_arcs, 1), dtype=np.float32)
        gamma = np.zeros((net_obj.num_arcs, 1), dtype=np.float32)

        demerr_pre = np.zeros((net_obj.num_arcs, 1), dtype=np.float32)
        vel_pre = np.zeros((net_obj.num_arcs, 1), dtype=np.float32)
        gamma_pre = np.zeros((net_obj.num_arcs, 1), dtype=np.float32)

        demerr_exca = np.zeros((net_obj.num_arcs, 1), dtype=np.float32)
        vel_exca = np.zeros((net_obj.num_arcs, 1), dtype=np.float32)
        gamma_exca = np.zeros((net_obj.num_arcs, 1), dtype=np.float32)

        demerr_conso = np.zeros((net_obj.num_arcs, 1), dtype=np.float32)
        vel_conso = np.zeros((net_obj.num_arcs, 1), dtype=np.float32)
        gamma_conso = np.zeros((net_obj.num_arcs, 1), dtype=np.float32)

        num_cores = net_obj.num_arcs if num_cores > net_obj.num_arcs else num_cores  # avoids having more samples
        # then cores
        idx = ut.splitDatasetForParallelProcessing(num_samples=net_obj.num_arcs, num_cores=num_cores)

        args = [(
            idx_range,
            idx_range.shape[0],
            net_obj.phase[idx_range, :],
            net_obj.phase[np.ix_(idx_range, ix_pre)], 
            net_obj.phase[np.ix_(idx_range, ix_exca)],
            net_obj.phase[np.ix_(idx_range, ix_conso)],
            net_obj.slant_range[idx_range],
            net_obj.loc_inc[idx_range],
            ifg_net_obj,
            wavelength,
            velocity_bound,
            vel_excavation_bound,
            demerr_bound,
            num_samples) for idx_range in idx]

        with multiprocessing.Pool(processes=num_cores) as pool:
            results = pool.map(func=launchAmbiguityFunctionSearchPiece3wise, iterable=args)

        # retrieve results
        for i, demerr_i, vel_i, gamma_i, demerr_pre_i, vel_pre_i, gamma_pre_i, demerr_exca_i, vel_exca_i, gamma_exca_i, demerr_conso_i, vel_conso_i, gamma_conso_i in results:
            demerr[i] = demerr_i
            vel[i] = vel_i
            gamma[i] = gamma_i
            
            demerr_pre[i] = demerr_pre_i
            vel_pre[i] = vel_pre_i
            gamma_pre[i] = gamma_pre_i
            
            demerr_exca[i] = demerr_exca_i
            vel_exca[i] = vel_exca_i
            gamma_exca[i] = gamma_exca_i

            demerr_conso[i] = demerr_conso_i
            vel_conso[i] = vel_conso_i
            gamma_conso[i] = gamma_conso_i

    m, s = divmod(time.time() - start_time, 60)
    logger.info(msg="Finished temporal unwrapping.")
    logger.debug(msg='time used: {:02.0f} mins {:02.1f} secs.'.format(m, s))

    outarrays = demerr, vel, gamma, demerr_pre, vel_pre, gamma_pre, demerr_exca, vel_exca, gamma_exca, demerr_conso, vel_conso, gamma_conso

    return outarrays



def launchAmbiguityFunctionSearchPiece3wise(parameters: tuple):
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
    (arc_idx_range, num_arcs, phase, phase_pre, phase_exca, phase_conso, slant_range, loc_inc, ifg_net_obj, wavelength,
     velocity_bound, vel_excavation_bound, demerr_bound, num_samples) = parameters

    demerr = np.zeros((num_arcs, 1), dtype=np.float32)
    vel = np.zeros((num_arcs, 1), dtype=np.float32)
    gamma = np.zeros((num_arcs, 1), dtype=np.float32)

    # pre excavation
    demerr_pre = np.zeros((num_arcs, 1), dtype=np.float32)
    vel_pre = np.zeros((num_arcs, 1), dtype=np.float32)
    gamma_pre = np.zeros((num_arcs, 1), dtype=np.float32)

    # excavation
    demerr_exca = np.zeros((num_arcs, 1), dtype=np.float32)
    vel_exca = np.zeros((num_arcs, 1), dtype=np.float32)
    gamma_exca = np.zeros((num_arcs, 1), dtype=np.float32)

    # consolidation
    demerr_conso = np.zeros((num_arcs, 1), dtype=np.float32)
    vel_conso = np.zeros((num_arcs, 1), dtype=np.float32)
    gamma_conso = np.zeros((num_arcs, 1), dtype=np.float32)

    design_mat = np.zeros((ifg_net_obj.num_ifgs, 2), dtype=np.float32)
    design_mat_pre = np.zeros((ifg_net_obj.num_ifgs_pre, 2), dtype=np.float32)
    design_mat_exca = np.zeros((ifg_net_obj.num_ifgs_exca, 2), dtype=np.float32)
    design_mat_conso = np.zeros((ifg_net_obj.num_ifgs_conso, 2), dtype=np.float32)

    demerr_range = np.linspace(-demerr_bound, demerr_bound, num_samples)
    vel_range = np.linspace(-velocity_bound, velocity_bound, num_samples)
    vel_excavation_range = np.linspace(-vel_excavation_bound, vel_excavation_bound, num_samples)
    vel_consolidation_range = np.linspace(-velocity_bound, velocity_bound, num_samples)
    prog_bar = ptime.progressBar(maxValue=num_arcs)

    factor = 4 * np.pi / wavelength
    every = max(1, num_arcs // 10)

    for k in range(num_arcs):
        design_mat[:, 0] = factor * ifg_net_obj.pbase_ifg / (slant_range[k] * np.sin(loc_inc[k]))
        design_mat[:, 1] = factor * ifg_net_obj.tbase_ifg

        design_mat_pre[:, 0] = factor * ifg_net_obj.pbase_ifg_pre / (slant_range[k] * np.sin(loc_inc[k]))
        design_mat_pre[:, 1] = factor * ifg_net_obj.tbase_ifg_pre

        design_mat_exca[:, 0] = factor * ifg_net_obj.pbase_ifg_exca / (slant_range[k] * np.sin(loc_inc[k]))
        design_mat_exca[:, 1] = factor * ifg_net_obj.tbase_ifg_exca

        design_mat_conso[:, 0] = factor * ifg_net_obj.pbase_ifg_conso / (slant_range[k] * np.sin(loc_inc[k]))
        design_mat_conso[:, 1] = factor * ifg_net_obj.tbase_ifg_conso

        demerr[k], vel[k], gamma[k] = oneDimSearchTemporalCoherence(
            demerr_range=demerr_range,
            vel_range=vel_range,
            obs_phase=phase[k, :],
            design_mat=design_mat)

        demerr_pre[k], vel_pre[k], gamma_pre[k] = oneDimSearchTemporalCoherence(
            demerr_range=demerr_range,
            vel_range=vel_range,
            obs_phase=phase_pre[k, :],
            design_mat=design_mat_pre)

        demerr_exca[k], vel_exca[k], gamma_exca[k] = oneDimSearchTemporalCoherence(
            demerr_range=demerr_range,
            vel_range=vel_excavation_range,
            obs_phase=phase_exca[k, :],
            design_mat=design_mat_exca)

        demerr_conso[k], vel_conso[k], gamma_conso[k] = oneDimSearchTemporalCoherence(
            demerr_range=demerr_range,
            vel_range=vel_consolidation_range,
            obs_phase=phase_conso[k, :],
            design_mat=design_mat_conso)

        prog_bar.update(value=k + 1, every=every,
                        suffix='{}/{} arcs processed. '.format(k + 1, num_arcs))

    return arc_idx_range, demerr, vel, gamma, demerr_pre, vel_pre, gamma_pre, demerr_exca, vel_exca, gamma_exca, demerr_conso, vel_conso, gamma_conso