from astropy.table import Table, hstack
from astropy.io import fits
import matplotlib.pyplot as plt 
import numpy as np
from astropy.stats import sigma_clip

import pyqtgraph as pg
from pyqtgraph import PlotWidget, PlotItem, ErrorBarItem, GraphicsLayoutWidget

from bruce.data import bin_data

class NGTSFIT_DATA:
    def __init__(self, filename):
        self.filename=filename
        self.optimal_table_hdu = fits.open(self.filename)[0].header['OPTAB'] 
        self.load_data(self.optimal_table_hdu)

    def load_data(self,optimal_index=2):
        self.data = hstack((Table.read(self.filename, hdu=2), Table.read(self.filename, hdu=optimal_index)))

        self.data['FLUX_ERR_ACTION'] = self.data['FLUX_ERR'].copy()
        for action in np.unique(self.data['ACTIONID']):
            mask = self.data['ACTIONID']==action
            self.data['FLUX_ERR_ACTION'][mask] = self.data['FLUX_ERR'][mask]/np.nanmedian(self.data['FLUX_SYSREM_ACTION'][mask])
            self.data['FLUX_SYSREM_ACTION'][mask] = self.data['FLUX_SYSREM_ACTION'][mask]/np.nanmedian(self.data['FLUX_SYSREM_ACTION'][mask])


    def plot_action_summary_matplotlib(self, action_id):

            # Mask rows belonging to this action
            mask = (self.data['ACTIONID'] == action_id) & self.data['CLEAN']
            data = self.data[mask]

            # Extract required columns FROM data (not raw self.data)
            BJD        = data['BJD']
            FLUX       = data['FLUX']
            FLUX_SYSREM = data['FLUX_SYSREM']
            FLUX_ERR   = data['FLUX_ERR']
            FLUX_BKG   = data['FLUX_BKG']
            PSF_SEP    = data['PSFSEP']
            PSF_RICH   = data['PSFRICH']
            PSF_TOM    = data['PSFTOM']
            X,  Y      = data['X'],  data['Y']
            XC, YC     = data['XC'], data['YC']
            AIRMASS    = data['AIRMASS']

            # Compute constant
            quarter_edges = np.linspace(BJD.min()-1e-3, BJD.max()+1e-3, 5)
            dig = np.digitize(BJD, quarter_edges, right=True)
            medians = np.array([
                np.median(FLUX_SYSREM[dig == i]) for i in range(1, len(quarter_edges))
            ])
            constant = np.max(medians)

            # Offset
            time_offset = int(np.floor(BJD.min()))
            time_offset_btjd = time_offset-2457000

            # Bin the SysRem-corrected flux
            t_bin, f_bin, fe_bin, count = bin_data(
                BJD - time_offset,
                FLUX_SYSREM / constant,
                0.5 / 24 / 6
            )
            good = count > 2
            t_bin, f_bin, fe_bin = t_bin[good], f_bin[good], fe_bin[good]

            # Bin raw flux
            t_bin_raw, f_bin_raw, fe_bin_raw, count = bin_data(
                BJD - time_offset,
                FLUX / constant,
                0.5 / 24 / 6
            )
            good = count > 2
            t_bin_raw, f_bin_raw, fe_bin_raw = (
                t_bin_raw[good], f_bin_raw[good], fe_bin_raw[good]
            )

            # Plot setup
            plt.rcParams.update({'font.size': 8, 'font.family': 'Times New Roman'})
            fig, axs = plt.subplots(5, 2, figsize=(10, 15), constrained_layout=True)

            # ---------- PANEL 0,0 ----------
            axs[0,0].errorbar(BJD - time_offset, FLUX_SYSREM / constant,
                            yerr=FLUX_ERR / constant, markersize=3,
                            fmt='k.', ecolor='lightgrey', alpha=0.2)
            ylim0 = axs[0,0].get_ylim()
            axs[0,0].errorbar(t_bin, f_bin, yerr=fe_bin, fmt='r.', markersize=3)
            axs[0,0].set(ylabel='FLUX SYSREM', xlabel=f'BJD - {time_offset_btjd}', ylim=ylim0)

            # ---------- PANEL 0,1 ----------
            axs[0,1].errorbar(t_bin, f_bin, yerr=fe_bin, fmt='r.', markersize=3)
            axs[0,1].errorbar(BJD - time_offset, FLUX_SYSREM / constant,
                            yerr=FLUX_ERR / constant, fmt='k.', alpha=0.2)
            axs[0,1].set(ylabel='FLUX SYSREM', xlabel=f'BJD - {time_offset_btjd}')

            # ---------- PANEL 1,0 ----------
            axs[1,0].plot(BJD - time_offset, FLUX/1000, 'k', lw=1)
            axs[1,0].set(ylabel='Target counts [1e3]', xlabel=f'BJD - {time_offset_btjd}')

            # ---------- PANEL 1,1 ----------
            axs[1,1].plot(BJD - time_offset, FLUX_BKG / constant, 'k')
            axs[1,1].set(ylabel='Sky counts', xlabel=f'BJD - {time_offset_btjd}')

            # ---------- PANEL 2,0 ----------
            axs[2,0].plot(BJD - time_offset, PSF_SEP, 'r', label='SEP')
            axs[2,0].plot(BJD - time_offset, PSF_TOM, 'g', label='TOM')
            axs[2,0].plot(BJD - time_offset, PSF_RICH, 'b', label='RICH')
            axs[2,0].set(ylabel='FWHM [pix]', xlabel=f'BJD - {time_offset_btjd}')
            axs[2,0].legend()

            # ---------- PANEL 2,1 ----------
            axs[2,1].plot(BJD - time_offset, AIRMASS, 'k')
            axs[2,1].set(ylabel='Airmass', xlabel=f'BJD - {time_offset_btjd}')

            # ---------- PANEL 3,0 ----------
            axs[3,0].plot(BJD - time_offset, X, 'k')
            axs[3,0].set(ylabel='X pixel', xlabel=f'BJD - {time_offset_btjd}')

            # ---------- PANEL 3,1 ----------
            axs[3,1].plot(BJD - time_offset, Y, 'k')
            axs[3,1].set(ylabel='Y pixel', xlabel=f'BJD - {time_offset_btjd}')

            # ---------- PANEL 4,0 ----------
            xc_mask = ~sigma_clip(XC, masked=True, sigma=2).mask
            axs[4,0].plot(BJD[xc_mask] - time_offset, XC[xc_mask], 'k')
            axs[4,0].set(ylabel='X centroid', xlabel=f'BJD - {time_offset_btjd}')

            # ---------- PANEL 4,1 ----------
            yc_mask = ~sigma_clip(YC, masked=True, sigma=2).mask
            axs[4,1].plot(BJD[yc_mask] - time_offset, YC[yc_mask], 'k')
            axs[4,1].set(ylabel='Y centroid', xlabel=f'BJD - {time_offset_btjd}')

            plt.show()











    def plot_action_summary_pyqtgraph(self, action_id):
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        # Mask rows belonging to this action
        mask = (self.data['ACTIONID'] == action_id) & self.data['CLEAN']
        data = self.data[mask]

        # Extract columns from filtered data
        BJD        = data['BJD']
        FLUX       = data['FLUX']
        FLUX_SYSREM = data['FLUX_SYSREM']
        FLUX_ERR   = data['FLUX_ERR']
        FLUX_BKG   = data['FLUX_BKG']
        PSF_SEP    = data['PSFSEP']
        PSF_RICH   = data['PSFRICH']
        PSF_TOM    = data['PSFTOM']
        X,  Y      = data['X'],  data['Y']
        XC, YC     = data['XC'], data['YC']
        AIRMASS    = data['AIRMASS']

        # Compute constant
        quarter_edges = np.linspace(BJD.min()-1e-3, BJD.max()+1e-3, 5)
        dig = np.digitize(BJD, quarter_edges, right=True)
        medians = np.array([np.median(FLUX_SYSREM[dig == i]) 
                            for i in range(1, len(quarter_edges))])
        constant = np.max(medians)

        # Offset
        time_offset = int(np.floor(BJD.min()))
        time_offset_btjd = time_offset - 2457000

        # Bin the SysRem-corrected flux
        t_bin, f_bin, fe_bin, count = bin_data(
            BJD - time_offset,
            FLUX_SYSREM / constant,
            0.5 / 24 / 6
        )
        good = count > 2
        t_bin, f_bin, fe_bin = t_bin[good], f_bin[good], fe_bin[good]

        # Bin raw flux
        t_bin_raw, f_bin_raw, fe_bin_raw, count = bin_data(
            BJD - time_offset,
            FLUX / constant,
            0.5 / 24 / 6
        )
        good = count > 2
        t_bin_raw, f_bin_raw, fe_bin_raw = (
            t_bin_raw[good], f_bin_raw[good], fe_bin_raw[good]
        )

        # ------------------------------
        # Create a PyQtGraph window with 5x2 grid
        # ------------------------------
        win = GraphicsLayoutWidget(show=True, title=f"Action {action_id} Summary")
        win.resize(1200, 1800)

        plots = []
        for r in range(5):
            row = []
            for c in range(2):
                p = win.addPlot(row=r, col=c)
                p.showGrid(x=True, y=True, alpha=0.3)
                row.append(p)
            plots.append(row)

        # Convenience function to add error bars
        def add_errorbars(plot, x, y, yerr, pen='k'):
            err = ErrorBarItem(
                x=x, y=y, top=yerr, bottom=yerr,
                beam=0.0, pen=pen
            )
            plot.addItem(err)

        # ===================================================
        # Panel 0,0  SYSREM flux + binned
        # ===================================================
        p = plots[0][0]
        p.setLabel('left', 'FLUX SYSREM')
        p.setLabel('bottom', f'BJD - {time_offset_btjd}')

        # Raw
        p.plot(BJD - time_offset, FLUX_SYSREM / constant,
            pen=None, symbol='o', symbolPen=None,
            symbolBrush=(50, 50, 50, 80), symbolSize=3)

        # Error bars
        add_errorbars(p, BJD - time_offset, FLUX_SYSREM/constant, FLUX_ERR/constant, pen='lightgray')

        # Binned
        p.plot(t_bin, f_bin, pen=None, symbol='o',
            symbolSize=5, symbolBrush='r')

        # ===================================================
        # Panel 0,1  binned first
        # ===================================================
        p = plots[0][1]
        p.setLabel('left', 'FLUX SYSREM')
        p.setLabel('bottom', f'BJD - {time_offset_btjd}')

        p.plot(t_bin, f_bin, pen=None, symbol='o',
            symbolSize=5, symbolBrush='r')

        p.plot(BJD - time_offset, FLUX_SYSREM / constant,
            pen=None, symbol='o', symbolBrush=(50,50,50,80),
            symbolSize=3)

        # Error bars
        add_errorbars(p, BJD - time_offset, FLUX_SYSREM/constant, FLUX_ERR/constant)

        # ===================================================
        # Panel 1,0 raw flux
        # ===================================================
        p = plots[1][0]
        p.setLabel('left', 'Target counts [1e3]')
        p.setLabel('bottom', f'BJD - {time_offset_btjd}')
        p.plot(BJD - time_offset, FLUX/1000, pen='k')

        # ===================================================
        # Panel 1,1 sky background
        # ===================================================
        p = plots[1][1]
        p.setLabel('left', 'Sky counts')
        p.setLabel('bottom', f'BJD - {time_offset_btjd}')
        p.plot(BJD - time_offset, FLUX_BKG/constant, pen='k')

        # ===================================================
        # Panel 2,0 FWHM
        # ===================================================
        p = plots[2][0]
        p.setLabel('left', 'FWHM [pix]')
        p.setLabel('bottom', f'BJD - {time_offset_btjd}')
        p.plot(BJD - time_offset, PSF_SEP, pen='r', name='SEP')
        p.plot(BJD - time_offset, PSF_TOM, pen='g', name='TOM')
        p.plot(BJD - time_offset, PSF_RICH, pen='b', name='RICH')

        # ===================================================
        # Panel 2,1 Airmass
        # ===================================================
        p = plots[2][1]
        p.setLabel('left', 'Airmass')
        p.setLabel('bottom', f'BJD - {time_offset_btjd}')
        p.plot(BJD - time_offset, AIRMASS, pen='k')

        # ===================================================
        # Panel 3,0 X pixel
        # ===================================================
        p = plots[3][0]
        p.setLabel('left', 'X pixel')
        p.setLabel('bottom', f'BJD - {time_offset_btjd}')
        p.plot(BJD - time_offset, X, pen='k')

        # ===================================================
        # Panel 3,1 Y pixel
        # ===================================================
        p = plots[3][1]
        p.setLabel('left', 'Y pixel')
        p.setLabel('bottom', f'BJD - {time_offset_btjd}')
        p.plot(BJD - time_offset, Y, pen='k')

        # ===================================================
        # Panel 4,0 X centroid
        # ===================================================
        xc_mask = ~sigma_clip(XC, masked=True, sigma=2).mask
        p = plots[4][0]
        p.setLabel('left', 'X centroid')
        p.setLabel('bottom', f'BJD - {time_offset_btjd}')
        p.plot(BJD[xc_mask] - time_offset, XC[xc_mask], pen='k')

        # ===================================================
        # Panel 4,1 Y centroid
        # ===================================================
        yc_mask = ~sigma_clip(YC, masked=True, sigma=2).mask
        p = plots[4][1]
        p.setLabel('left', 'Y centroid')
        p.setLabel('bottom', f'BJD - {time_offset_btjd}')
        p.plot(BJD[yc_mask] - time_offset, YC[yc_mask], pen='k')

        return win   # <-- Use inside your Qt app


filename = '/Volumes/NGTS/TIC-186072225_REF_IMAGE_80820250301081204_PHOT.fits'
d = NGTSFIT_DATA(filename=filename)
actions = np.unique(d.data['ACTIONID'])




# d.plot_action_summary_matplotlib(actions[0])
# plt.show()


from PyQt5 import QtWidgets
import sys

app = QtWidgets.QApplication(sys.argv)
win = d.plot_action_summary_pyqtgraph(405479)
sys.exit(app.exec_())