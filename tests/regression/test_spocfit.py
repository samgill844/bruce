"""Check the local-FITS hand-off without launching Qt or querying MAST."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from astropy.io import fits
from astropy.table import Table
from utils.spocfit import load_spoc_fits


class LocalPhotometryTests(unittest.TestCase):
    def make_file(self, filename, include_error=True, include_header=True):
        # Schema written by tesstpf's Fit the lightcurve action.
        table = Table({'TIME': [2037., 2036.], 'SAP_FLUX': [90., 100.],
                       'PDCSAP_FLUX': [90., 100.], 'SAP_BKG': [5., 6.],
                       'SAP_BKG_ERR': [1., 1.], 'QUALITY_MASK': [True, True]})
        if include_error:
            table['PDCSAP_FLUX_ERR'] = [3., 4.]
        table.write(filename)
        if include_header:
            with fits.open(filename, mode='update') as hdus:
                hdus[0].header['TICID'] = 100504488
                hdus[0].header['SECTOR'] = 27

    def test_tpf_handoff_preserves_identity_times_flux_and_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory)/'TIC-100504488_S27_TPF_lightcurve.fits'
            self.make_file(filename)
            table, tic_id, sector = load_spoc_fits(filename)
        self.assertEqual((tic_id, sector), (100504488, 27))
        np.testing.assert_array_equal(table['TIME'], [2036., 2037.])
        np.testing.assert_array_equal(table['PDCSAP_FLUX'], [100., 90.])
        np.testing.assert_array_equal(table['PDCSAP_FLUX_ERR'], [4., 3.])

    def test_incomplete_files_raise_explicit_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_header = Path(directory)/'no_header.fits'
            self.make_file(missing_header, include_header=False)
            with self.assertRaisesRegex(ValueError, 'TICID and SECTOR'):
                load_spoc_fits(missing_header)
            missing_error = Path(directory)/'no_error.fits'
            self.make_file(missing_error, include_error=False)
            with self.assertRaisesRegex(ValueError, 'PDCSAP_FLUX_ERR'):
                load_spoc_fits(missing_error)
