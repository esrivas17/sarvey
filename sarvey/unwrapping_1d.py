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


def temporalUnwrapping_demerr(*, ifg_net_obj: IfgNetwork, net_obj: Network,  wavelength: float,
                       demerr_bound: float, num_samples: int, num_cores: int = 1, logger: Logger) -> \
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
    gamma: np.ndarray
    """
    msg = "#" * 10
    msg += " TEMPORAL UNWRAPPING FOR DEM ERROR: AMBIGUITY FUNCTION "
    msg += "#" * 10
    logger.info(msg=msg)

    start_time = time.time()

    if num_cores == 1:
        args = (
            np.arange(net_obj.num_arcs), net_obj.num_arcs, net_obj.phase,
            net_obj.slant_range, net_obj.loc_inc, ifg_net_obj, wavelength, demerr_bound, num_samples)
        arc_idx_range, demerr, gamma = launchAmbiguityFunctionSearch(parameters=args)
    else:
        logger.info(msg="start parallel processing with {} cores.".format(num_cores))

        demerr = np.zeros((net_obj.num_arcs, 1), dtype=np.float32)
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
            demerr_bound,
            num_samples) for idx_range in idx]

        with multiprocessing.Pool(processes=num_cores) as pool:
            results = pool.map(func=launchAmbiguityFunctionSearch, iterable=args)

        # retrieve results
        for i, demerr_i, gamma_i in results:
            demerr[i] = demerr_i
            gamma[i] = gamma_i

    m, s = divmod(time.time() - start_time, 60)
    logger.info(msg="Finished temporal unwrapping.")
    logger.debug(msg='time used: {:02.0f} mins {:02.1f} secs.'.format(m, s))
    return demerr, gamma


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
    (arc_idx_range, num_arcs, phase, slant_range, loc_inc, ifg_net_obj, wavelength, demerr_bound, num_samples) = parameters

    demerr = np.zeros((num_arcs, 1), dtype=np.float32)
    gamma = np.zeros((num_arcs, 1), dtype=np.float32)

    design_mat = np.zeros((ifg_net_obj.num_ifgs, 1), dtype=np.float32)

    demerr_range = np.linspace(-demerr_bound, demerr_bound, num_samples)

    factor = 4 * np.pi / wavelength

    for k in range(num_arcs):
        design_mat[:, 0] = factor * ifg_net_obj.pbase_ifg / (slant_range[k] * np.sin(loc_inc[k]))
        demerr[k], gamma[k] = coarseSearchTempCoh(demerr_range=demerr_range, obs_phase=phase[k, :], design_mat=design_mat)
        
        #    demerr_range=demerr_range, obs_phase=phase[k, :], design_mat=design_mat)
        #demerr[k], gamma[k] = oneDimSearchTemporalCoherence(
        #    demerr_range=demerr_range, obs_phase=phase[k, :], design_mat=design_mat)

    return arc_idx_range, demerr, gamma

def coarseSearchTempCoh(*, demerr_range: np.ndarray, obs_phase: np.ndarray,  design_mat: np.ndarray):
    from sarvey.unwrapping_temperature import findOptimum

    demerr, gamma_demerr, pred_phase_demerr = findOptimum(obs_phase=obs_phase, design_mat=design_mat[:, 0],val_range=demerr_range)
    return demerr, gamma_demerr


def oneDimSearchTemporalCoherence(*, demerr_range: np.ndarray, obs_phase: np.ndarray,  design_mat: np.ndarray):
    from sarvey.unwrapping_temperature import findOptimum

    demerr, gamma_demerr, pred_phase_demerr = findOptimum(obs_phase=obs_phase, design_mat=design_mat[:, 0],val_range=demerr_range)

    scales = np.array([(demerr_range.max() - demerr_range.min()) / 2.0])

    centers = np.array([(demerr_range.max() + demerr_range.min()) / 2.0])

    # Initial physical guess
    p0 = np.array([demerr])
    # Convert to scaled space for L-BFGS-B
    x0 = (p0 - centers) / scales

    demerr, gamma = gradientSearchTcoh(
        scales=scales,
        centers=centers,
        obs_phase=obs_phase,
        design_mat=design_mat,
        x0=x0)

    pred_phase = np.matmul(design_mat, demerr)
    res = (obs_phase - pred_phase.T).ravel()
    gamma = np.abs(np.mean(np.exp(1j * res)))
    return demerr, gamma


def gradientSearchTcoh(*, scales, centers, obs_phase, design_mat, x0):
    """
    Gradient-based parameter refinement in scaled space.

    Parameters
    ----------
    scales : array-like (3,)
        Scaling factors for parameters.
    centers : array-like (3,)
        Centers of parameters.
    x0 : array-like (3,)
        Initial guess in scaled space.

    Returns
    -------
    (demerr, vel, tcoef, gamma)
        Physical parameter estimates and their temporal coherence.
    """

    # Bounds in scaled space: keep search within [-1, 1] or whatever you want
    bounds = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]

    opt_res = minimize(
        objFuncTcoh,
        x0,
        args=(design_mat, obs_phase, scales, centers),
        bounds=bounds,
        method='L-BFGS-B'
    )

    # Convert scaled solution back to physical parameters
    p_est = opt_res.x * scales + centers
    demerr = p_est

    # Compute gamma at optimum
    pred_phase = design_mat @ p_est
    res = (obs_phase - pred_phase).ravel()
    gamma = np.abs(np.mean(np.exp(1j * res)))

    return demerr, gamma



def objFuncTcoh(x, *args):
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
    demerr = x[0] * scale_demerr

    p = demerr

    pred_phase = design_mat @ p
    res = (obs_phase - pred_phase.T).ravel()
    gamma = np.abs(np.mean(np.exp(1j * res)))
    return 1 - gamma