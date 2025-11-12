import os, json, urllib.request, tempfile, zipfile, glob, shutil
import numpy as np
from astropy.table import Table, Column, vstack
from astroquery.mast import Observations
from astropy.io import fits
from astropy.wcs import WCS
from bruce.ambiguous_period.mono_event import photometry_time_series
from astropy.stats import sigma_clip
from bruce.data import bin_data
from astroquery.mast import Catalogs
import lightkurve as lk
import matplotlib.pyplot as plt 
from scipy.stats import median_abs_deviation

def download_tess_data(tic_id, max_sector=None, use_ffi=True, download_dir=None, bin_length=None):
    """
    Download available TESS data (SPOC, QLP, and optionally FFI) for a target.
    
    Parameters
    ----------
    tic_id : str or int
        TIC identifier for the target.
    ra, dec : float
        Right ascension and declination of the target.
    max_sector : int or None, optional
        If given, only download up to this sector.
    use_ffi : bool, optional
        If True (default), create custom light curves from FFIs when no LC is available.
    download_dir : str or None, optional
        Directory to store downloaded data. If None, a temporary directory is created.

    Returns
    -------
    summary : astropy.table.Table
        Table listing data availability and downloaded files.
    datasets : list
        List of photometry_time_series objects.
    data_path : str
        Path to directory containing downloaded products.
    """

    # Lets query TIC8
    # Now query tic8 
    tic8 = Catalogs.query_object('TIC{:}'.format(tic_id), radius=.02, catalog="TIC")
    if len(tic8)==0 : raise ValueError('No obsject found :( ')
    tic8 = tic8[0]
    ra, dec = tic8['ra'], tic8['dec']

    def query_tesscut(product=None):
        url = f"https://mast.stsci.edu/tesscut/api/v0.1/sector?ra={ra}&dec={dec}"
        if product:
            url += f"&product={product}"
        with urllib.request.urlopen(url) as u:
            res = json.load(u)['results']
        if not res:
            return Table()
        keys = list(res[0].keys())
        data = [[i[j] for j in keys] for i in res]
        return Table(np.array(data), names=keys)

    # Create / manage the download directory
    if download_dir is None:
        tempdir = tempfile.TemporaryDirectory()
        base_dir = tempdir.name
        auto_cleanup = True
    else:
        os.makedirs(download_dir, exist_ok=True)
        base_dir = os.path.abspath(download_dir)
        tempdir = None
        auto_cleanup = False

    print(f"\n📂 Download directory: {base_dir}\n")

    # Query FFI and SPOC availability
    data_ffi = query_tesscut() if use_ffi else Table()
    data_spoc = query_tesscut("SPOC")

    if len(data_ffi) == 0 and len(data_spoc) == 0:
        print("No data available.")
        return Table(), [], base_dir

    # Combine and group by sector
    total = vstack((data_ffi, data_spoc)).group_by('sector')
    sectors = total.groups.keys
    if max_sector:
        sectors = sectors[sectors['sector'] <= max_sector]

    # Query MAST for light curves
    print(f"Querying TIC {tic_id} from MAST...")
    obs = Observations.query_criteria(provenance_name=('TESS-SPOC', 'SPOC', 'QLP'),
                                      target_name=str(tic_id))
    try:
        data = Observations.get_product_list(obs)
        mask = np.zeros(len(data), dtype=bool)
        for ext in ('_lc.fits', '_llc.fits', '_tp.fits', '.pdf'):
            mask |= np.char.endswith(data['productFilename'], ext)
        data = data[mask]
    except Exception:
        data = Table()

    # Initialize summary table
    t = Table(masked=True)
    t.add_column(Column(np.array(sectors).astype(int), name='Sector'))
    for name in ['SPOC FFI', 'QLP LC', 'TESS-SPOC LC', 'SPOC LC']:
        t.add_column(Column(np.zeros(len(sectors), dtype='|S1'), name=name))

    data_files, data_origin, datasets, datasets_labels  = [], [], [], []

    # Helper to move product to download_dir
    def move_to_dir(filepath):
        dst = os.path.join(base_dir, os.path.basename(filepath))
        shutil.move(filepath, dst)
        return dst

    # Download priority: SPOC LC → TESS-SPOC LC → QLP LC → FFI
    for i, sector in enumerate(t['Sector']):
        downloaded = False

        # SPOC short-cadence LC
        for row in data:
            if (row['project'] == 'SPOC' and
                f"s{sector:04}" in row['productFilename'] and
                row['productFilename'].endswith('_lc.fits')):
                out = Observations.download_products(row)
                if out['Status'][0] == 'COMPLETE':
                    path = move_to_dir(out['Local Path'][0])
                    data_files.append(path)
                    data_origin.append('SPOC LC')
                    t['SPOC LC'][i] = 'X'
                    downloaded = True

                    tt = Table.read(path)
                    mask = (~np.isinf(tt['TIME']) & ~np.isnan(tt['TIME']) &
                            ~np.isinf(tt['SAP_FLUX']) & ~np.isnan(tt['SAP_FLUX']) &
                            (tt['SAP_FLUX'] > 0.1) & (tt['PDCSAP_FLUX'] > 0.1))
                    tt = tt[mask]
                    
                    time = np.array(tt['TIME'], dtype=np.float64)+2457000
                    flux =  np.array(tt['PDCSAP_FLUX'], dtype=np.float64)
                    flux_err =  np.array(tt['PDCSAP_FLUX_ERR'], dtype=np.float64)
                    sky_bkg =  np.array(tt['SAP_BKG'], dtype=np.float64)


                    if time.shape[0]<10 : continue
                    if (bin_length is not None) and np.median(np.gradient(time))<(0.5*bin_length): 
                        time, flux, flux_err, c = bin_data(time,flux, bin_length)
                        mask = c>2
                        time, flux, flux_err = time[mask], flux[mask], flux_err[mask]
                        
                    datasets.append(photometry_time_series(time, flux, flux_err, sky_bkg=sky_bkg if bin_length is None else None))
                    datasets_labels.append('Sector {:} [SPOC SHORT CADENCE]'.format(sector))
                break
        if downloaded: continue

        # TESS-SPOC FFI LC
        for row in data:
            if (row['project'] == 'TESS-SPOC' and
                f"s{sector:04}" in row['productFilename'] and
                row['productFilename'].endswith('_lc.fits')):
                out = Observations.download_products(row)
                if out['Status'][0] == 'COMPLETE':
                    path = move_to_dir(out['Local Path'][0])
                    data_files.append(path)
                    data_origin.append('TESS-SPOC LC')
                    t['TESS-SPOC LC'][i] = 'X'
                    downloaded = True

                    tt = Table.read(path)
                    mask = (~np.isinf(tt['TIME']) & ~np.isnan(tt['TIME']) &
                            ~np.isinf(tt['SAP_FLUX']) & ~np.isnan(tt['SAP_FLUX']) &
                            (tt['SAP_FLUX'] > 0.1) & (tt['PDCSAP_FLUX'] > 0.1))
                    tt = tt[mask]
                    
                    time = np.array(tt['TIME'], dtype=np.float64)+2457000
                    flux =  np.array(tt['PDCSAP_FLUX'], dtype=np.float64)
                    flux_err =  np.array(tt['PDCSAP_FLUX_ERR'], dtype=np.float64)
                    sky_bkg =  np.array(tt['SAP_BKG'], dtype=np.float64)

                    if time.shape[0]<10 : continue
                    if (bin_length is not None) and np.median(np.gradient(time))<(0.5*bin_length): 
                        time, flux, flux_err, c = bin_data(time,flux, bin_length)
                        mask = c>2
                        time, flux, flux_err = time[mask], flux[mask], flux_err[mask]
                        
                    datasets.append(photometry_time_series(time, flux, flux_err, sky_bkg=sky_bkg if bin_length is None else None))
                    datasets_labels.append('Sector {:} [SPOC FFI]'.format(sector))
                break
        if downloaded: continue

        # QLP LC
        for row in data:
            if (row['project'] == 'QLP' and
                f"s{sector:04}" in row['productFilename'] and
                row['productFilename'].endswith('_llc.fits')):
                out = Observations.download_products(row)
                if out['Status'][0] == 'COMPLETE':
                    path = move_to_dir(out['Local Path'][0])
                    data_files.append(path)
                    data_origin.append('QLP LC')
                    t['QLP LC'][i] = 'X'
                    downloaded = True

                    tt = Table.read(path)
                    mask = (~np.isinf(tt['TIME']) & ~np.isnan(tt['TIME']) &
                            ~np.isinf(tt['SAP_FLUX']) & ~np.isnan(tt['SAP_FLUX']) &
                            (tt['SAP_FLUX'] > 0.1) & (tt['QUALITY']==0))
                    tt = tt[mask]
                    
                    time = np.array(tt['TIME'], dtype=np.float64)+2457000
                    flux =  np.array(tt['SAP_FLUX'], dtype=np.float64)
                    if 'DET_FLUX_ERR' in tt.colnames:
                        #flux =  np.array(tt['DET_FLUX'], dtype=np.float64)
                        flux_err =  np.array(tt['DET_FLUX_ERR'], dtype=np.float64)
                    else : 
                        #flux =  np.array(tt['KSPSAP_FLUX'], dtype=np.float64)
                        flux_err =  np.array(tt['KSPSAP_FLUX_ERR'], dtype=np.float64) 


            

                    sky_bkg = np.array(tt['SAP_BKG'], dtype=np.float64)   



                    if time.shape[0]<10 : continue
                    if (bin_length is not None) and np.median(np.gradient(time))<(0.5*bin_length): 
                        time, flux, flux_err, c = bin_data(time,flux, bin_length)
                        mask = c>2
                        time, flux, flux_err = time[mask], flux[mask], flux_err[mask]
                        
                    datasets.append(photometry_time_series(time, flux, flux_err, sky_bkg=sky_bkg if bin_length is None else None))
                    datasets_labels.append('Sector {:} [QLP FFI]'.format(sector))

                break
        if downloaded: continue

        # FFI custom extraction if requested
        if use_ffi and len(data_ffi) > 0:
            with tempfile.TemporaryDirectory() as tmp:
                zipfile_path = f"{tmp}/cutout.zip"
                cmd = (
                    f'wget -q -O {zipfile_path} '
                    f'"https://mast.stsci.edu/tesscut/api/v0.1/astrocut?ra={ra}&dec={dec}&y=10&x=10&sector={sector}"'
                )
                os.system(cmd)
                with zipfile.ZipFile(zipfile_path, 'r') as zf:
                    zf.extractall(tmp)
                fits_file = glob.glob(f"{tmp}/*.fits")[0]
                final_path = move_to_dir(fits_file)
                data_files.append(final_path)
                data_origin.append('FFI custom LC')
                t['SPOC FFI'][i] = 'X'
                downloaded = True

                # Now create the lightcurve 
                tpf = lk.targetpixelfile.TessTargetPixelFile(final_path)

                # Inspect the TPF
                tpf.plot(frame=0, title="Single Frame of TPF")

                # Define a target mask (pixels containing the star)
                # You can also interactively create it using `tpf.create_threshold_mask()`
                target_mask = tpf.create_threshold_mask(threshold=3)  # 3 sigma above median

                # Define a background mask (pixels far from the target)
                # One way is to invert the target mask, but exclude edges
                #background_mask = ~target_mask

                # Define a background mask based on pixel flux statistics
                # Here, we use frame 30 as a reference, similar to self.FLUX[30]
                reference_frame = 30
                flux_frame = tpf.flux[reference_frame]  # 2D array of pixels

                # Create background mask: pixels below 30th percentile in this frame
                background_mask = flux_frame < np.percentile(flux_frame, 30)

                # Optional: make sure the background mask does not overlap the target mask
                background_mask = background_mask & ~target_mask

                # Extract target flux
                target_flux = tpf.to_lightcurve(aperture_mask=target_mask)

                # Estimate background flux (median over background pixels for each frame)
                background_flux = tpf.to_lightcurve(aperture_mask=background_mask).flux
                background_flux = background_flux.value.astype(np.float64)  # total background
                #background_flux_per_frame = background_flux.mean(axis=1)

                # Subtract background
                corrected_flux = target_flux.flux.value.astype(np.float64) - target_mask.sum()*background_flux/background_mask.sum()

                # here we should check empty TPFs
                if np.nanmedian(corrected_flux)>100 : 
                    # Create a background-subtracted light curve object
                    lc_bgsub = lk.LightCurve(time=target_flux.time, flux=corrected_flux)

                    time = target_flux.time.value.astype(np.float64) + 2457000.0
                    flux = corrected_flux
                    flux_err = (np.ones(corrected_flux.shape[0])*median_abs_deviation(corrected_flux[~np.isnan(corrected_flux) & ~np.isinf(corrected_flux)])).astype(np.float64)
                    sky_bkg = background_flux


                    mask = ~(np.isnan(flux) | np.isinf(flux) | np.isnan(flux_err) | np.isinf(flux_err)| np.isnan(sky_bkg) | np.isinf(sky_bkg))
                    #mask = mask & ~sigma_clip(flux, masked=True, sigma=5).mask
                    time, flux, flux_err, sky_bkg = time[mask], flux[mask], flux_err[mask], sky_bkg[mask]


                    if time.shape[0]<10 : continue
                    if (bin_length is not None) and np.median(np.gradient(time))<(0.5*bin_length): 
                        time, flux, flux_err, c = bin_data(time,flux, bin_length)
                        mask = c>2
                        time, flux, flux_err = time[mask], flux[mask], flux_err[mask]


                    plt.close()
                        
                    datasets.append(photometry_time_series(time, flux, flux_err, sky_bkg=sky_bkg if bin_length is None else None))
                    datasets_labels.append('Sector {:} [FFI custom]'.format(sector))
                else : 
                    print('MEDIAN CHECK not met : ' + str(np.nanmedian(corrected_flux)))
        if not downloaded:
            data_files.append('')
            data_origin.append('Missing')

    # Apply masks column by column
    for col in ['SPOC FFI', 'QLP LC', 'TESS-SPOC LC', 'SPOC LC']:
        t[col].mask = t[col] != 'X'

    t['Source'] = data_origin
    t['File'] = data_files

    print("\n✅ Download summary:")
    t.pprint(max_lines=1000)

    # If using temp dir, warn user where data lives
    if auto_cleanup:
        print(f"\nTemporary data stored in: {base_dir}")
        print("This directory will be deleted when the program exits.")

    return t, np.array(datasets), np.array(datasets_labels), base_dir