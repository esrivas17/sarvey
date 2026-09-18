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
import pdb


class IfgNetwork3Piecewise:
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
        self.ix_ifg = list()

        # pre-excavation
        self.pbase_pre = None
        self.tbase_pre = None
        self.num_images_pre = None
        self.pbase_ifg_pre = None
        self.tbase_ifg_pre = None
        self.num_ifgs_pre = None
        self.dates_pre = list()
        self.ifg_list_pre = list()
        self.ix_ifg_pre = list()

        # excavation
        self.pbase_exca = None
        self.tbase_exca = None
        self.num_images_exca = None
        self.pbase_ifg_exca = None
        self.tbase_ifg_exca = None
        self.num_ifgs_exca = None
        self.dates_exca = list()
        self.ifg_list_exca = list()
        self.ix_ifg_exca = list()

        # consolidation
        self.pbase_conso = None
        self.tbase_conso = None
        self.num_images_conso = None
        self.pbase_ifg_conso = None
        self.tbase_ifg_conso = None
        self.num_ifgs_conso = None
        self.dates_conso = list()
        self.ifg_list_conso = list()
        self.ix_ifg_conso = list()

        self.ix_breakpoint1 = None  # shared node between pre and exca
        self.ix_breakpoint2 = None  # shared node between exca and conso

    def plot(self):
        """Plot the full network of interferograms."""
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

    def _plot_period(self, *, dates_period, pbase_period, ifg_list_period, tbase_ifg_period, pbase_ifg_period, title):
        """Shared plotting logic for a single period (pre / exca / conso)."""
        fig = plt.figure(figsize=(15, 5))
        axs = fig.subplots(1, 3)
        dt = [datetime.date.fromisoformat(d) for d in self.dates]
        dt_period = [datetime.date.fromisoformat(d) for d in dates_period]
        axs[0].plot(dt_period, pbase_period, 'ko')
        for idx in ifg_list_period:
            xx = np.array([dt[idx[0]], dt[idx[1]]])
            yy = np.array([self.pbase[idx[0]], self.pbase[idx[1]]])
            axs[0].plot(xx, yy, 'k-')
        axs[0].set_ylabel('perpendicular baseline [m]')
        axs[0].set_xlabel('temporal baseline [years]')
        axs[0].set_title(title)
        fig.autofmt_xdate()

        axs[1].hist(tbase_ifg_period * 365.25, bins=100)
        axs[1].set_ylabel('Absolute frequency')
        axs[1].set_xlabel('temporal baseline [days]')

        axs[2].hist(pbase_ifg_period, bins=100)
        axs[2].set_ylabel('Absolute frequency')
        axs[2].set_xlabel('perpendicular baseline [m]')
        return fig

    def plot_ifg_network_old(self):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.subplots(1, 1)
        dt = [datetime.date.fromisoformat(d) for d in self.dates]
        ax.plot(dt, self.pbase, 'ko', markersize=3)
        for idx in self.ifg_list:
            xx = np.array([dt[idx[0]], dt[idx[1]]])
            yy = np.array([self.pbase[idx[0]], self.pbase[idx[1]]])
            ax.plot(xx, yy, 'k-', lw=0.5)
        ax.set_ylabel('perpendicular baseline [m]')
        ax.set_xlabel('temporal baseline [years]')
        ax.set_title('Network of interferograms')

        bpoints_pbase = [dt[self.ix_breakpoint1], dt[self.ix_breakpoint2]]
        bpoints_tbase = [self.pbase[self.ix_breakpoint2], self.pbase[self.ix_breakpoint2]]
        ax.scatter(bpoints_pbase, bpoints_tbase, s=9, c="orange", label="breakpoints", zorder=5)
        ax.legend()
        fig.autofmt_xdate()
        return fig

    def plot_ifg_network(self):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.subplots(1, 1)

        # Acquisition dates
        dt = [datetime.date.fromisoformat(d) for d in self.dates]

        # Plot all acquisition points
        ax.plot(dt, self.pbase, 'ko', markersize=3)

        # Plot interferograms according to their temporal period
        for idx in self.ifg_list:
            i, j = idx

            xx = np.array([dt[i], dt[j]])
            yy = np.array([self.pbase[i], self.pbase[j]])

            if i < self.ix_breakpoint1:
                linestyle = '-'
            elif i < self.ix_breakpoint2:
                linestyle = '--'
            else:
                linestyle = ':'

            ax.plot(xx,yy,color='k',lw=0.6,linestyle=linestyle)

        # Breakpoint locations
        bpoints_x = [dt[self.ix_breakpoint1],dt[self.ix_breakpoint2]]

        bpoints_y = [self.pbase[self.ix_breakpoint1],self.pbase[self.ix_breakpoint2]]

        ax.scatter(bpoints_x,bpoints_y,s=25,color='orange',edgecolor='k',linewidth=0.5,zorder=5,label='Breakpoints')

        # Labels
        ax.set_ylabel('Perpendicular baseline [m]')
        ax.set_xlabel('Acquisition date')
        ax.set_title('Network of interferograms')

        # Legend for temporal periods
        from matplotlib.lines import Line2D

        legend_lines = [
            Line2D(
                [0], [0],
                color='k',
                lw=0.8,
                linestyle='-',
                label='Pre-excavation'
            ),
            Line2D(
                [0], [0],
                color='k',
                lw=0.8,
                linestyle='--',
                label='Excavation'
            ),
            Line2D(
                [0], [0],
                color='k',
                lw=0.8,
                linestyle=':',
                label='Consolidation'
            ),
            Line2D(
                [0], [0],
                marker='o',
                color='orange',
                markeredgecolor='k',
                markeredgewidth=0.5,
                linestyle='None',
                markersize=5,
                label='Breakpoints'
            )
        ]

        ax.legend(handles=legend_lines)

        fig.autofmt_xdate()

        return fig


    def plot_pre(self):
        """Plot the network of interferograms for the pre-excavation period."""
        return self._plot_period(
            dates_period=self.dates_pre,
            pbase_period=self.pbase_pre,
            ifg_list_period=self.ifg_list_pre,
            tbase_ifg_period=self.tbase_ifg_pre,
            pbase_ifg_period=self.pbase_ifg_pre,
            title='Network of interferograms (pre-excavation)',
        )

    def plot_exca(self):
        """Plot the network of interferograms for the excavation period."""
        return self._plot_period(
            dates_period=self.dates_exca,
            pbase_period=self.pbase_exca,
            ifg_list_period=self.ifg_list_exca,
            tbase_ifg_period=self.tbase_ifg_exca,
            pbase_ifg_period=self.pbase_ifg_exca,
            title='Network of interferograms (excavation)',
        )

    def plot_consolidation(self):
        """Plot the network of interferograms for the consolidation period."""
        return self._plot_period(
            dates_period=self.dates_conso,
            pbase_period=self.pbase_conso,
            ifg_list_period=self.ifg_list_conso,
            tbase_ifg_period=self.tbase_ifg_conso,
            pbase_ifg_period=self.pbase_ifg_conso,
            title='Network of interferograms (consolidation)',
        )

    def getDesignMatrix(self):
        """Compute the design matrix for the full network."""
        a = np.zeros((self.num_ifgs, self.num_images))
        for i in range(len(self.ifg_list)):
            a[i, self.ifg_list[i][0]] = 1
            a[i, self.ifg_list[i][1]] = -1
        return a

    def _getDesignMatrixPeriod(self, *, ifg_list_period, num_ifgs_period, num_images_period, offset):
        """Shared design-matrix logic for a single period. `offset` converts absolute image
        indices (as stored in ifg_list_period) into indices relative to that period's own
        image array, since e.g. exca/conso image indices don't start at 0 in the full series."""
        a = np.zeros((num_ifgs_period, num_images_period))
        for i, (start, end) in enumerate(ifg_list_period):
            a[i, start - offset] = 1
            a[i, end - offset] = -1
        return a

    def getDesignMatrixPre(self):
        """Compute the design matrix for the pre-excavation period."""
        return self._getDesignMatrixPeriod(
            ifg_list_period=self.ifg_list_pre,
            num_ifgs_period=self.num_ifgs_pre,
            num_images_period=self.num_images_pre,
            offset=0,  # pre always starts at index 0
        )

    def getDesignMatrixExca(self):
        """Compute the design matrix for the excavation period."""
        return self._getDesignMatrixPeriod(
            ifg_list_period=self.ifg_list_exca,
            num_ifgs_period=self.num_ifgs_exca,
            num_images_period=self.num_images_exca,
            offset=self.ix_breakpoint1,
        )

    def getDesignMatrixConso(self):
        """Compute the design matrix for the consolidation period."""
        return self._getDesignMatrixPeriod(
            ifg_list_period=self.ifg_list_conso,
            num_ifgs_period=self.num_ifgs_conso,
            num_images_period=self.num_images_conso,
            offset=self.ix_breakpoint2,
        )

    def open(self, *, path: str):
        """Read stored information from an already existing .h5 file.

        Parameter
        -----------
        path: str
            path to existing file to read from.
        """
        with h5py.File(path, 'r') as f:
            self.num_images = f.attrs["num_images"]
            self.num_ifgs = f.attrs["num_ifgs"]
            self.ix_ifg = f.attrs["ix_ifg"]

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
            self.ix_breakpoint1 = f.attrs["ix_breakpoint1"]
            self.ix_ifg_pre = f.attrs["ix_ifg_pre"]

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
                print(f"IfgNetwork is in old dataformat. Cannot read 'dates_pre'! {ke}")

            # excavation
            self.num_images_exca = f.attrs["num_images_exca"]
            self.num_ifgs_exca = f.attrs["num_ifgs_exca"]
            self.ix_ifg_exca = f.attrs["ix_ifg_exca"]

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
                print(f"IfgNetwork is in old dataformat. Cannot read 'dates_exca'! {ke}")

            # consolidation
            try:
                self.num_images_conso = f.attrs["num_images_conso"]
                self.num_ifgs_conso = f.attrs["num_ifgs_conso"]
                self.ix_breakpoint2 = f.attrs["ix_breakpoint2"]
                self.ix_ifg_conso = f.attrs["ix_ifg_conso"]

                self.tbase_ifg_conso = f['tbase_ifg_conso'][:]
                self.pbase_ifg_conso = f['pbase_ifg_conso'][:]
                self.tbase_conso = f['tbase_conso'][:]
                self.pbase_conso = f['pbase_conso'][:]
                self.ifg_list_conso = f['ifg_list_conso'][:]
                self.dates_conso = f['dates_conso'][:]
                self.dates_conso = [date.decode("utf-8") for date in self.dates_conso]
            except KeyError as ke:
                print(f"IfgNetwork is in old dataformat (two periods only). Cannot read consolidation data! {ke}")

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
        dates_conso = np.array(self.dates_conso, dtype=np.bytes_)

        with h5py.File(path, 'w') as f:
            f.attrs["num_images"] = self.num_images
            f.attrs["num_ifgs"] = self.num_ifgs
            f.attrs["ix_ifg"] = self.ix_ifg

            f.create_dataset('tbase_ifg', data=self.tbase_ifg)
            f.create_dataset('pbase_ifg', data=self.pbase_ifg)
            f.create_dataset('tbase', data=self.tbase)
            f.create_dataset('pbase', data=self.pbase)
            f.create_dataset('ifg_list', data=self.ifg_list)
            f.create_dataset('dates', data=dates)

            # pre
            f.attrs["num_images_pre"] = self.num_images_pre
            f.attrs["num_ifgs_pre"] = self.num_ifgs_pre
            f.attrs["ix_breakpoint1"] = self.ix_breakpoint1
            f.attrs["ix_ifg_pre"] = self.ix_ifg_pre

            f.create_dataset('tbase_ifg_pre', data=self.tbase_ifg_pre)
            f.create_dataset('pbase_ifg_pre', data=self.pbase_ifg_pre)
            f.create_dataset('tbase_pre', data=self.tbase_pre)
            f.create_dataset('pbase_pre', data=self.pbase_pre)
            f.create_dataset('ifg_list_pre', data=self.ifg_list_pre)
            f.create_dataset('dates_pre', data=dates_pre)

            # excavation
            f.attrs["num_images_exca"] = self.num_images_exca
            f.attrs["num_ifgs_exca"] = self.num_ifgs_exca
            f.attrs["ix_ifg_exca"] = self.ix_ifg_exca

            f.create_dataset('tbase_ifg_exca', data=self.tbase_ifg_exca)
            f.create_dataset('pbase_ifg_exca', data=self.pbase_ifg_exca)
            f.create_dataset('tbase_exca', data=self.tbase_exca)
            f.create_dataset('pbase_exca', data=self.pbase_exca)
            f.create_dataset('ifg_list_exca', data=self.ifg_list_exca)
            f.create_dataset('dates_exca', data=dates_exca)

            # consolidation
            f.attrs["num_images_conso"] = self.num_images_conso
            f.attrs["num_ifgs_conso"] = self.num_ifgs_conso
            f.attrs["ix_breakpoint2"] = self.ix_breakpoint2
            f.attrs["ix_ifg_conso"] = self.ix_ifg_conso

            f.create_dataset('tbase_ifg_conso', data=self.tbase_ifg_conso)
            f.create_dataset('pbase_ifg_conso', data=self.pbase_ifg_conso)
            f.create_dataset('tbase_conso', data=self.tbase_conso)
            f.create_dataset('pbase_conso', data=self.pbase_conso)
            f.create_dataset('ifg_list_conso', data=self.ifg_list_conso)
            f.create_dataset('dates_conso', data=dates_conso)


class StarNetwork(IfgNetwork3Piecewise):
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

        ix_ifg = 0
        for i in range(self.num_images):
            if i == ref_idx:
                continue
            self.ifg_list.append((ref_idx, i))
            self.ix_ifg.append(ix_ifg)
            if i >= ix_break:
                self.ifg_list_exca.append((ref_idx, i))
                self.ix_ifg_exca.append(ix_ifg)
            else:
                self.ifg_list_pre.append((ref_idx, i))
                self.ix_ifg_pre.append(ix_ifg)
            ix_ifg += 1


        self.pbase_ifg = np.delete(self.pbase - self.pbase[ref_idx], ref_idx)
        self.tbase_ifg = np.delete(self.tbase - self.tbase[ref_idx], ref_idx)
        self.num_ifgs = self.num_images - 1

        # pre and excavation configuration

        self.tbase_pre = tbase[:ix_break]/365.25
        self.pbase_pre = pbase[:ix_break]
        self.dates_pre = dates[:ix_break]
        self.num_images_pre = self.tbase_pre.shape[0]
        

        self.tbase_exca = tbase[ix_break:]/365.25
        self.pbase_exca = pbase[ix_break:]
        self.dates_exca = dates[ix_break:]
        self.num_images_exca = self.tbase_exca.shape[0]

        if ref_idx < ix_break:
            self.pbase_ifg_pre = np.delete(self.pbase_pre - self.pbase[ref_idx], ref_idx)
            self.tbase_ifg_pre = np.delete(self.tbase_pre - self.tbase[ref_idx], ref_idx)
            self.num_ifgs_pre = self.num_images_pre - 1
            self.pbase_ifg_exca = self.pbase_exca - self.pbase[ref_idx]
            self.tbase_ifg_exca = self.tbase_exca - self.tbase[ref_idx]
            self.num_ifgs_exca = self.num_images_exca
        else:
            self.pbase_ifg_pre = self.pbase_pre - self.pbase[ref_idx]
            self.pbase_ifg_exca = np.delete(self.pbase_exca - self.pbase[ref_idx], ref_idx)
            self.tbase_ifg_pre = self.tbase_pre - self.tbase[ref_idx]
            self.tbase_ifg_exca = np.delete(self.tbase_exca - self.tbase[ref_idx], ref_idx)
            self.num_ifgs_pre = self.num_images_pre
            self.num_ifgs_exca = self.num_images_exca - 1



class SmallTemporalBaselinesNetwork(IfgNetwork3Piecewise):
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
        """Create list of interferograms split into two periods: pre and exca (excavation).

        Interferograms are only formed within a single period. The breakpoint index (ix_break) is
        shared between the two periods, acting as a bridge node so the overall network stays connected.
        Any candidate connection that spans from one period's interior into the other's is dropped.

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
        ix_break: int
            Index shared between the "pre" and "exca" periods.
        """
        self.pbase = pbase
        self.tbase = tbase / 365.25
        self.num_images = pbase.shape[0]
        self.dates = dates

        # period ranges (inclusive), overlapping exactly at the shared breakpoint
        periods = [
            (0, ix_break, 'pre'),
            (ix_break, self.num_images - 1, 'exca'),
        ]

        def period_of_pair(i, j):
            """Return the period name if both i and j lie within the same period range, else None."""
            for lo, hi, name in periods:
                if lo <= i <= hi and lo <= j <= hi:
                    return name
            return None

        ix_ifg = 0
        for i in range(self.num_images):
            for j in range(num_link):
                end = i + j + 1
                if end >= self.num_images:
                    continue

                period = period_of_pair(i, end)
                if period is None:
                    continue  # cross-period connection: drop entirely

                self.ifg_list.append((i, end))
                self.ix_ifg.append(ix_ifg)

                if period == 'pre':
                    self.ifg_list_pre.append((i, end))
                    self.ix_ifg_pre.append(ix_ifg)
                else:  # exca
                    self.ifg_list_exca.append((i, end))
                    self.ix_ifg_exca.append(ix_ifg)

                ix_ifg += 1

        self.ifg_list = [(i, j) for i, j in self.ifg_list if i != j]  # remove connections to itself, e.g. (0, 0)
        self.ifg_list_pre = [(i, j) for i, j in self.ifg_list_pre if i != j]
        self.ifg_list_exca = [(i, j) for i, j in self.ifg_list_exca if i != j]

        self.pbase_ifg = np.array([self.pbase[idx[1]] - self.pbase[idx[0]] for idx in self.ifg_list])
        self.tbase_ifg = np.array([self.tbase[idx[1]] - self.tbase[idx[0]] for idx in self.ifg_list])
        self.num_ifgs = self.pbase_ifg.shape[0]

        # pre and excavation configuration
        self.tbase_pre = tbase[:ix_break + 1] / 365.25
        self.tbase_exca = tbase[ix_break:] / 365.25
        self.pbase_pre = pbase[:ix_break + 1]
        self.pbase_exca = pbase[ix_break:]
        self.dates_pre = dates[:ix_break + 1]
        self.dates_exca = dates[ix_break:]
        self.num_images_pre = self.tbase_pre.shape[0]
        self.num_images_exca = self.tbase_exca.shape[0]

        self.pbase_ifg_pre = np.array([self.pbase[idx[1]] - self.pbase[idx[0]] for idx in self.ifg_list_pre])
        self.tbase_ifg_pre = np.array([self.tbase[idx[1]] - self.tbase[idx[0]] for idx in self.ifg_list_pre])
        self.num_ifgs_pre = self.pbase_ifg_pre.shape[0]

        self.pbase_ifg_exca = np.array([self.pbase[idx[1]] - self.pbase[idx[0]] for idx in self.ifg_list_exca])
        self.tbase_ifg_exca = np.array([self.tbase[idx[1]] - self.tbase[idx[0]] for idx in self.ifg_list_exca])
        self.num_ifgs_exca = self.pbase_ifg_exca.shape[0]

    def configure_three_periods(self, *, pbase: np.ndarray, tbase: np.ndarray, num_link: int = None, dates: list,
                                ix_break1: int, ix_break2: int):
        """Create list of interferograms split into three periods: pre, exca (excavation), and conso (consolidation).

        Interferograms are only formed within a single period. The breakpoint indices (ix_break1, ix_break2)
        are shared between adjacent periods, acting as bridge nodes so the overall network stays connected.
        Any candidate connection that spans from one period's interior into another's is dropped.

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
        ix_break1: int
            Index shared between the "pre" and "exca" periods.
        ix_break2: int
            Index shared between the "exca" and "conso" periods.
        """
        self.pbase = pbase
        self.tbase = tbase / 365.25
        self.num_images = pbase.shape[0]
        self.dates = dates
        self.ix_breakpoint1 = ix_break1
        self.ix_breakpoint2 = ix_break2

        # period ranges (inclusive), overlapping exactly at the shared breakpoint
        periods = [
            (0, ix_break1, 'pre'),
            (ix_break1, ix_break2, 'exca'),
            (ix_break2, self.num_images - 1, 'conso'),
        ]

        def period_of_pair(i, j):
            """Return the period name if both i and j lie within the same period range, else None."""
            for lo, hi, name in periods:
                if lo <= i <= hi and lo <= j <= hi:
                    return name
            return None

        ix_ifg = 0
        for i in range(self.num_images):
            for j in range(num_link):
                end = i + j + 1
                if end >= self.num_images:
                    continue

                period = period_of_pair(i, end)
                if period is None:
                    continue  # cross-period connection: drop entirely

                self.ifg_list.append((i, end))
                self.ix_ifg.append(ix_ifg)

                if period == 'pre':
                    self.ifg_list_pre.append((i, end))
                    self.ix_ifg_pre.append(ix_ifg)
                elif period == 'exca':
                    self.ifg_list_exca.append((i, end))
                    self.ix_ifg_exca.append(ix_ifg)
                else:  # conso
                    self.ifg_list_conso.append((i, end))
                    self.ix_ifg_conso.append(ix_ifg)

                ix_ifg += 1

        self.ifg_list = [(i, j) for i, j in self.ifg_list if i != j]  # remove connections to itself, e.g. (0, 0)
        self.ifg_list_pre = [(i, j) for i, j in self.ifg_list_pre if i != j]
        self.ifg_list_exca = [(i, j) for i, j in self.ifg_list_exca if i != j]
        self.ifg_list_conso = [(i, j) for i, j in self.ifg_list_conso if i != j]

        self.pbase_ifg = np.array([self.pbase[idx[1]] - self.pbase[idx[0]] for idx in self.ifg_list])
        self.tbase_ifg = np.array([self.tbase[idx[1]] - self.tbase[idx[0]] for idx in self.ifg_list])
        self.num_ifgs = self.pbase_ifg.shape[0]

        # pre, excavation and consolidation configuration
        self.tbase_pre = tbase[:ix_break1 + 1] / 365.25
        self.tbase_exca = tbase[ix_break1:ix_break2 + 1] / 365.25
        self.tbase_conso = tbase[ix_break2:] / 365.25
        self.pbase_pre = pbase[:ix_break1 + 1]
        self.pbase_exca = pbase[ix_break1:ix_break2 + 1]
        self.pbase_conso = pbase[ix_break2:]
        self.dates_pre = dates[:ix_break1 + 1]
        self.dates_exca = dates[ix_break1:ix_break2 + 1]
        self.dates_conso = dates[ix_break2:]
        self.num_images_pre = self.tbase_pre.shape[0]
        self.num_images_exca = self.tbase_exca.shape[0]
        self.num_images_conso = self.tbase_conso.shape[0]

        self.pbase_ifg_pre = np.array([self.pbase[idx[1]] - self.pbase[idx[0]] for idx in self.ifg_list_pre])
        self.tbase_ifg_pre = np.array([self.tbase[idx[1]] - self.tbase[idx[0]] for idx in self.ifg_list_pre])
        self.num_ifgs_pre = self.pbase_ifg_pre.shape[0]

        self.pbase_ifg_exca = np.array([self.pbase[idx[1]] - self.pbase[idx[0]] for idx in self.ifg_list_exca])
        self.tbase_ifg_exca = np.array([self.tbase[idx[1]] - self.tbase[idx[0]] for idx in self.ifg_list_exca])
        self.num_ifgs_exca = self.pbase_ifg_exca.shape[0]

        self.pbase_ifg_conso = np.array([self.pbase[idx[1]] - self.pbase[idx[0]] for idx in self.ifg_list_conso])
        self.tbase_ifg_conso = np.array([self.tbase[idx[1]] - self.tbase[idx[0]] for idx in self.ifg_list_conso])
        self.num_ifgs_conso = self.pbase_ifg_conso.shape[0]
