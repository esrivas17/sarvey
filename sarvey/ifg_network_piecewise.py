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

"""IfgNetwork module for SARvey."""
import datetime
import h5py
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Union
import warnings
from logging import Logger
from scipy.spatial import Delaunay


class IfgNetworkPiecewise:
    """Abstract class/interface for different types of interferogram networks."""

    ifg_list: Union[list, np.ndarray] = None

    def __init__(self):
        """Init."""
        self.pbase = None
        self.tbase = None
        self.num_images = None
        self.ifg_list = list()  # is later converted to np.array
        self.pbase_ifg = None
        self.tbase_ifg = None
        self.num_ifgs = None
        self.dates = list()

        # pre-xcavation
        self.pbase_pre = None
        self.tbase_pre = None
        self.num_images_pre = None
        self.pbase_ifg_pre = None
        self.tbase_ifg_pre = None
        self.num_ifgs_pre = None
        self.dates_pre = list()
        self.ifg_list_pre = list()
        
        # excavation
        self.pbase_exca = None
        self.tbase_exca = None
        self.num_images_exca = None
        self.pbase_ifg_exca = None
        self.tbase_ifg_exca = None
        self.num_ifgs_exca = None
        self.dates_exca = list()
        self.ifg_list_exca = list()

    def plot(self):
        """Plot the network of interferograms."""
        fig = plt.figure(figsize=(15, 5))
        axs = fig.subplots(1, 3)
        dt = [datetime.date.fromisoformat(d) for d in self.dates]
        axs[0].plot(dt, self.pbase, 'ko')
        for idx in self.ifg_list:
            xx = np.array([dt[idx[0]], dt[idx[1]]])
            yy = np.array([self.pbase[idx[0]], self.pbase[idx[1]]])
            axs[0].plot(xx, yy, 'k-')
        axs[0].set_ylabel('perpendicular baseline [m]')
        axs[0].set_xlabel('temporal baseline [years]')
        axs[0].set_title('Network of interferograms')
        fig.autofmt_xdate()

        axs[1].hist(self.tbase_ifg * 365.25, bins=100)
        axs[1].set_ylabel('Absolute frequency')
        axs[1].set_xlabel('temporal baseline [days]')

        axs[2].hist(self.pbase_ifg, bins=100)
        axs[2].set_ylabel('Absolute frequency')
        axs[2].set_xlabel('perpendicular baseline [m]')
        return fig
    
    def plot_pre(self):
        """Plot the network of interferograms."""
        fig = plt.figure(figsize=(15, 5))
        axs = fig.subplots(1, 3)
        dt = [datetime.date.fromisoformat(d) for d in self.dates]
        dt_pre = [datetime.date.fromisoformat(d) for d in self.dates_pre]
        axs[0].plot(dt_pre, self.pbase_pre, 'ko')
        for idx in self.ifg_list_pre:
            xx = np.array([dt[idx[0]], dt[idx[1]]])
            yy = np.array([self.pbase[idx[0]], self.pbase[idx[1]]])
            axs[0].plot(xx, yy, 'k-')
        axs[0].set_ylabel('perpendicular baseline [m]')
        axs[0].set_xlabel('temporal baseline [years]')
        axs[0].set_title('Network of interferograms')
        fig.autofmt_xdate()

        axs[1].hist(self.tbase_ifg_pre * 365.25, bins=100)
        axs[1].set_ylabel('Absolute frequency')
        axs[1].set_xlabel('temporal baseline [days]')

        axs[2].hist(self.pbase_ifg_pre, bins=100)
        axs[2].set_ylabel('Absolute frequency')
        axs[2].set_xlabel('perpendicular baseline [m]')
        return fig

    def plot_exca(self):
        """Plot the network of interferograms."""
        fig = plt.figure(figsize=(15, 5))
        axs = fig.subplots(1, 3)
        dt = [datetime.date.fromisoformat(d) for d in self.dates]
        dt_exca = [datetime.date.fromisoformat(d) for d in self.dates_exca]
        axs[0].plot(dt_exca, self.pbase_exca, 'ko')
        for idx in self.ifg_list_exca:
            xx = np.array([dt[idx[0]], dt[idx[1]]])
            yy = np.array([self.pbase[idx[0]], self.pbase[idx[1]]])
            axs[0].plot(xx, yy, 'k-')
        axs[0].set_ylabel('perpendicular baseline [m]')
        axs[0].set_xlabel('temporal baseline [years]')
        axs[0].set_title('Network of interferograms')
        fig.autofmt_xdate()

        axs[1].hist(self.tbase_ifg_exca * 365.25, bins=100)
        axs[1].set_ylabel('Absolute frequency')
        axs[1].set_xlabel('temporal baseline [days]')

        axs[2].hist(self.pbase_ifg_exca, bins=100)
        axs[2].set_ylabel('Absolute frequency')
        axs[2].set_xlabel('perpendicular baseline [m]')
        return fig

    def getDesignMatrix(self):
        """Compute the design matrix for the smallbaseline network."""
        a = np.zeros((self.num_ifgs, self.num_images))
        for i in range(len(self.ifg_list)):
            a[i, self.ifg_list[i][0]] = 1
            a[i, self.ifg_list[i][1]] = -1
        return a
    
    def getDesignMatrixPre(self):
        a = np.zeros((self.num_ifgs_pre, self.num_images_pre))
        for i in range(len(self.ifg_list_pre)):
            a[i, self.ifg_list_pre[i][0]] = 1
            a[i, self.ifg_list_pre[i][1]] = -1
        return a
    
    def getDesignMatrixExca(self):
        a = np.zeros((self.num_ifgs_exca, self.num_images_exca))
        for i in range(len(self.ifg_list_exca)):
            a[i, self.ifg_list_exca[i][0]] = 1
            a[i, self.ifg_list_exca[i][1]] = -1
        return a

    def open(self, *, path: str):
        """Read stored information from already existing.h5 file.

        Parameter
        -----------
        path: str
            path to existing file to read from.
        """
        with h5py.File(path, 'r') as f:
            self.num_images = f.attrs["num_images"]
            self.num_ifgs = f.attrs["num_ifgs"]

            self.tbase_ifg = f['tbase_ifg'][:]
            self.pbase_ifg = f['pbase_ifg'][:]
            self.tbase = f['tbase'][:]
            self.pbase = f['pbase'][:]
            self.ifg_list = f['ifg_list'][:]
            try:
                self.dates = f['dates'][:]
                self.dates = [date.decode("utf-8") for date in self.dates]
            except KeyError as ke:
                self.dates = None
                print(f"IfgNetwork is in old dataformat. Cannot read 'dates'! {ke}")

            # pre
            self.num_images_pre = f.attrs["num_images_pre"]
            self.num_ifgs_pre = f.attrs["num_ifgs_pre"]

            self.tbase_ifg_pre = f['tbase_ifg_pre'][:]
            self.pbase_ifg_pre = f['pbase_ifg_pre'][:]
            self.tbase_pre = f['tbase_pre'][:]
            self.pbase_pre = f['pbase_pre'][:]
            self.ifg_list_pre = f['ifg_list_pre'][:]
            try:
                self.dates_pre = f['dates_pre'][:]
                self.dates_pre = [date.decode("utf-8") for date in self.dates_pre]
            except KeyError as ke:
                self.dates_pre = None
                print(f"IfgNetwork is in old dataformat. Cannot read 'dates'! {ke}")

            # excavation
            self.num_images_pre = f.attrs["num_images_exca"]
            self.num_ifgs_pre = f.attrs["num_ifgs_exca"]

            self.tbase_ifg_exca = f['tbase_ifg_exca'][:]
            self.pbase_ifg_exca = f['pbase_ifg_exca'][:]
            self.tbase_exca = f['tbase_exca'][:]
            self.pbase_exca = f['pbase_exca'][:]
            self.ifg_list_exca = f['ifg_list_exca'][:]
            try:
                self.dates_exca = f['dates_exca'][:]
                self.dates_exca = [date.decode("utf-8") for date in self.dates_exca]
            except KeyError as ke:
                self.dates_exca = None
                print(f"IfgNetwork is in old dataformat. Cannot read 'dates'! {ke}")


            f.close()

    def writeToFile(self, *, path: str, logger: Logger):
        """Write all existing data to .h5 file.

        Parameters
        ----------
        path: str
            path to filename
        logger: Logger
            Logging handler.
        """
        logger.info(msg="write IfgNetwork to {}".format(path))

        if os.path.exists(path):
            os.remove(path)

        dates = np.array(self.dates, dtype=np.bytes_)
        dates_pre = np.array(self.dates_pre, dtype=np.bytes_)
        dates_exca = np.array(self.dates_exca, dtype=np.bytes_)

        with h5py.File(path, 'w') as f:
            f.attrs["num_images"] = self.num_images
            f.attrs["num_ifgs"] = self.num_ifgs

            f.create_dataset('tbase_ifg', data=self.tbase_ifg)
            f.create_dataset('pbase_ifg', data=self.pbase_ifg)
            f.create_dataset('tbase', data=self.tbase)
            f.create_dataset('pbase', data=self.pbase)
            f.create_dataset('ifg_list', data=self.ifg_list)
            f.create_dataset('dates', data=dates)

            # pre excavation
            f.attrs["num_images_pre"] = self.num_images
            f.attrs["num_ifgs_pre"] = self.num_ifgs

            f.create_dataset('tbase_ifg_pre', data=self.tbase_ifg)
            f.create_dataset('pbase_ifg_pre', data=self.pbase_ifg)
            f.create_dataset('tbase_pre', data=self.tbase)
            f.create_dataset('pbase_pre', data=self.pbase)
            f.create_dataset('ifg_list_pre', data=self.ifg_list)
            f.create_dataset('dates_pre', data=dates_pre)

            # excavation
            f.attrs["num_images_exca"] = self.num_images
            f.attrs["num_ifgs_exca"] = self.num_ifgs

            f.create_dataset('tbase_ifg_exca', data=self.tbase_ifg)
            f.create_dataset('pbase_ifg_exca', data=self.pbase_ifg)
            f.create_dataset('tbase_exca', data=self.tbase)
            f.create_dataset('pbase_exca', data=self.pbase)
            f.create_dataset('ifg_list_exca', data=self.ifg_list)
            f.create_dataset('dates_exca', data=dates_exca)


class StarNetwork(IfgNetworkPiecewise):
    """Star network of interferograms (single-reference)."""

    def configure(self, *, pbase: np.ndarray, tbase: np.ndarray, ref_idx: int, dates: list):
        """Create list of interferograms containing the indices of the images and computes baselines.

        Parameter
        ---------
        pbase: np.ndarray
            Perpendicular baselines of the SAR acquisitions.
        tbase: np.ndarray
            Temporal baselines of the SAR acquisitions.
        ref_idx: int
            Index of the reference image.
        dates: list
            Dates of the acquisitions.
        """
        self.pbase = pbase
        self.tbase = tbase / 365.25
        self.num_images = pbase.shape[0]
        self.dates = dates

        for i in range(self.num_images):
            if i == ref_idx:
                continue
            self.ifg_list.append((ref_idx, i))

        self.pbase_ifg = np.delete(self.pbase - self.pbase[ref_idx], ref_idx)
        self.tbase_ifg = np.delete(self.tbase - self.tbase[ref_idx], ref_idx)
        self.num_ifgs = self.num_images - 1

    def configure_breakpoint(self, *, pbase: np.ndarray, tbase: np.ndarray, ref_idx: int, dates: list, ix_break: int):
        self.pbase = pbase
        self.tbase = tbase / 365.25
        self.num_images = pbase.shape[0]
        self.dates = dates

        for i in range(self.num_images):
            if i == ref_idx:
                continue
            self.ifg_list.append((ref_idx, i))

        self.pbase_ifg = np.delete(self.pbase - self.pbase[ref_idx], ref_idx)
        self.tbase_ifg = np.delete(self.tbase - self.tbase[ref_idx], ref_idx)
        self.num_ifgs = self.num_images - 1

        # pre and excavation configuration

        self.tbase_pre = tbase[:ix_break]/365.25
        self.tbase_exca = tbase[ix_break:]/365.25
        self.pbase_pre = pbase[:ix_break]
        self.pbase_exca = pbase[ix_break:]
        self.dates_pre = dates[:ix_break]
        self.dates_exca = dates[ix_break:]
        self.num_images_pre = self.tbase_pre.shape[0]
        self.num_images_exca = self.tbase_exca.shape[0]

        for i in range(0, ix_break):
                if i == ref_idx:
                    continue
                self.ifg_list_pre.append((ref_idx, i))

        for i in range(ix_break, self.num_images):            
            if i == ref_idx:
                continue
            self.ifg_list_exca.append((ref_idx, i))

        if ref_idx < ix_break:
            self.pbase_ifg_pre = np.delete(self.pbase_pre - self.pbase[ref_idx], ref_idx)
            self.pbase_ifg_exca = self.pbase_exca - self.pbase[ref_idx]
            self.tbase_ifg_pre = np.delete(self.tbase_pre - self.tbase[ref_idx], ref_idx)
            self.tbase_ifg_exca = self.tbase_exca - self.tbase[ref_idx]
            self.num_ifgs_pre = self.num_images_pre - 1
            self.num_ifgs_exca = self.num_images_exca
        else:
            self.pbase_ifg_pre = self.pbase_pre - self.pbase[ref_idx]
            self.pbase_ifg_exca = np.delete(self.pbase_exca - self.pbase[ref_idx], ref_idx)
            self.tbase_ifg_pre = self.tbase_pre - self.tbase[ref_idx]
            self.tbase_ifg_exca = np.delete(self.tbase_exca - self.tbase[ref_idx], ref_idx)
            self.num_ifgs_pre = self.num_images_pre
            self.num_ifgs_exca = self.num_images_exca - 1



class SmallTemporalBaselinesNetwork(IfgNetworkPiecewise):
    """Small temporal baselines network of interferograms without restrictions on the perpendicular baselines."""

    def configure(self, *, pbase: np.ndarray, tbase: np.ndarray, num_link: int = None, dates: list):
        """Create list of interferograms containing the indices of the images and computes baselines.

        Parameter
        -----------
        pbase: np.ndarray
            Perpendicular baselines of the SAR acquisitions.
        tbase: np.ndarray
            Temporal baselines of the SAR acquisitions.
        num_link: int
            Number of consecutive links in time connecting acquisitions.
        dates: list
            Dates of the acquisitions.
        """
        self.pbase = pbase
        self.tbase = tbase / 365.25
        self.num_images = pbase.shape[0]
        self.dates = dates

        for i in range(self.num_images):
            for j in range(num_link):
                if i + j + 1 >= self.num_images:
                    continue
                self.ifg_list.append((i, i + j + 1))

        self.ifg_list = [(i, j) for i, j in self.ifg_list if i != j]  # remove connections to itself, e.g. (0, 0)

        self.pbase_ifg = np.array([self.pbase[idx[1]] - self.pbase[idx[0]] for idx in self.ifg_list])
        self.tbase_ifg = np.array([self.tbase[idx[1]] - self.tbase[idx[0]] for idx in self.ifg_list])
        self.num_ifgs = self.pbase_ifg.shape[0]

    def configure_breakpoint(self, *, pbase: np.ndarray, tbase: np.ndarray, num_link: int = None, dates: list, ix_break: int):
            self.pbase = pbase
            self.tbase = tbase / 365.25
            self.num_images = pbase.shape[0]
            self.dates = dates

            for i in range(self.num_images):
                for j in range(num_link):
                    if i + j + 1 >= self.num_images:
                        continue
                    self.ifg_list.append((i, i + j + 1))

            self.ifg_list = [(i, j) for i, j in self.ifg_list if i != j]  # remove connections to itself, e.g. (0, 0)

            self.pbase_ifg = np.array([self.pbase[idx[1]] - self.pbase[idx[0]] for idx in self.ifg_list])
            self.tbase_ifg = np.array([self.tbase[idx[1]] - self.tbase[idx[0]] for idx in self.ifg_list])
            self.num_ifgs = self.pbase_ifg.shape[0]


            # pre and excavation configuration
            self.tbase_pre = tbase[:ix_break]/365.25
            self.tbase_exca = tbase[ix_break:]/365.25
            self.pbase_pre = pbase[:ix_break]
            self.pbase_exca = pbase[ix_break:]
            self.dates_pre = dates[:ix_break]
            self.dates_exca = dates[ix_break:]
            self.num_images_pre = self.tbase_pre.shape[0]
            self.num_images_exca = self.tbase_exca.shape[0]

            for i in range(0, ix_break):
                for j in range(num_link):
                    if i + j + 1 >= self.num_images:
                        continue
                    self.ifg_list_pre.append((i, i + j + 1))

            for i in range(ix_break, self.num_images):            
                for j in range(num_link):
                    if i + j + 1 >= self.num_images:
                        continue
                    self.ifg_list_exca.append((i, i + j + 1))

            self.pbase_ifg_pre = np.array([self.pbase_pre[idx[1]] - self.pbase_pre[idx[0]] for idx in self.ifg_list_pre])
            self.tbase_ifg_pre = np.array([self.tbase_pre[idx[1]] - self.tbase_pre[idx[0]] for idx in self.ifg_list_pre])
            self.num_ifgs_pre = self.pbase_ifg_pre.shape[0]

            self.pbase_ifg_exca = np.array([self.pbase_exca[idx[1]] - self.pbase_exca[idx[0]] for idx in self.ifg_list_exca])
            self.tbase_ifg_exca = np.array([self.tbase_exca[idx[1]] - self.tbase_exca[idx[0]] for idx in self.ifg_list_exca])
            self.num_ifgs_exca = self.pbase_ifg_exca.shape[0]



