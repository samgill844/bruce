import numpy as np
import urllib.request, json
from astropy.table import Table, vstack
from astroquery.mast import Catalogs
from astroquery.mast import Observations
import re, tempfile
from astropy.stats import sigma_clip

from bruce.ambiguous_period import photometry_time_series
from bruce.data import bin_data

def get_tess_data(tic_id, verbose=False, download_dir=None, max_sector=np.inf, projects=['TESS-SPOC', 'SPOC', 'QLP'], data_type='single_product', bin_length=None, sigma=None):
    obsTable = Observations.query_criteria(provenance_name=('TESS-SPOC','SPOC', 'QLP'), target_name=tic_id)
    data = Observations.get_product_list(obsTable)
    mask = np.array([i[-7:]=='lc.fits' for i in data['productFilename']], dtype = bool)
    data = data[mask]
    data['sector'] = np.zeros(len(data), dtype=int)
    for i in range(len(data)):
        re1 = re.search(r's(\d{4})', data['productFilename'][i])
        re2 = re.search(r'-s(\d{4})', data['productFilename'][i])
        if re1 is not None : data['sector'][i] = int(re1.group(1))
        if re2 is not None : data['sector'][i] = int(re2.group(1))

    if verbose : print(data['sector', 'productFilename'])

    # Now get the data to download
    data_to_download=[]
    for i in np.unique(data['sector']):
            if i > max_sector: continue
            data_ = data[data['sector']==i]
            if ('TESS-SPOC' in data_['project']) and ('TESS-SPOC' in projects) : 
                data_to_download.append(data_[np.argwhere(data_['project']=='TESS-SPOC')[0][0]])
                continue
            elif ('SPOC' in data_['project']) and ('SPOC' in projects) : 
                data_to_download.append(data_[np.argwhere(data_['project']=='SPOC')[0][0]])
                continue
            elif ('QLP' in data_['project']) and ('QLP' in projects) : 
                data_to_download.append(data_[np.argwhere(data_['project']=='QLP')[0][0]])
                continue


    if verbose : print(data_to_download)
    data_to_download = vstack(data_to_download)


    if data_type=='single_product':
        # Now download the data
        time, flux, flux_err = np.array([]),np.array([]),np.array([])
        with tempfile.TemporaryDirectory() as tmpdirname:
            for i in range(len(data_to_download)):
                out = Observations.download_products(data_to_download[i], download_dir=download_dir)
                if out['Status'][0]=='COMPLETE':
                    data = Table.read(out['Local Path'][0])
                    if 'SPOC' in data_to_download['project'][i]:
                        data = data[~np.isinf(data['TIME']) & ~np.isnan(data['TIME']) & ~np.isinf(data['SAP_FLUX']) & ~np.isnan(data['SAP_FLUX']) & (data['SAP_FLUX'] > 1) & (data['PDCSAP_FLUX'] > 1)]
                        time = np.concatenate((time, np.array(data['TIME'])+2457000))
                        flux = np.concatenate((flux, np.array(data['PDCSAP_FLUX'])))
                        flux_err = np.concatenate((flux_err, np.array(data['PDCSAP_FLUX_ERR'])))
                    if 'QLP' in data_to_download['project'][i]:
                        #data = data[~np.isinf(data['TIME']) & ~np.isnan(data['TIME']) & ~np.isinf(data['SAP_FLUX']) & ~np.isnan(data['SAP_FLUX']) & (data['SAP_FLUX'] > 1) & (data['PDCSAP_FLUX'] > 1)]
                        time = np.concatenate((time, np.array(data['TIME'])+2457000))
                        if 'KSPSAP_FLUX' in data.colnames:
                            flux = np.concatenate((flux, np.array(data['KSPSAP_FLUX'])))
                            flux_err = np.concatenate((flux_err, np.array(data['KSPSAP_FLUX_ERR'])))
                        if 'DET_FLUX' in data.colnames:
                            flux = np.concatenate((flux, np.array(data['DET_FLUX'])))
                            flux_err = np.concatenate((flux_err, np.array(data['DET_FLUX_ERR'])))
        mask = np.isnan(time) | np.isnan(flux) | np.isnan(flux_err) | np.isinf(time) | np.isinf(flux) | np.isinf(flux_err)

        # Now sigma clip
        mask = mask & ~sigma_clip(flux, sigma=3,masked=True).mask

        return photometry_time_series(time[~mask], flux[~mask], flux_err[~mask]), 'all_data'

    elif data_type=='per_sector':
        data_return = []
        data_return_labels=[]
        with tempfile.TemporaryDirectory() as tmpdirname:
            for i in range(len(data_to_download)):
                out = Observations.download_products(data_to_download[i], download_dir=download_dir)
                if out['Status'][0]=='COMPLETE':
                    data = Table.read(out['Local Path'][0])
                    if 'SPOC' in data_to_download['project'][i]:
                        data = data[~np.isinf(data['TIME']) & ~np.isnan(data['TIME']) & ~np.isinf(data['SAP_FLUX']) & ~np.isnan(data['SAP_FLUX']) & (data['SAP_FLUX'] > 1) & (data['PDCSAP_FLUX'] > 1)]
                        time = np.array(data['TIME'])+2457000
                        flux = np.array(data['PDCSAP_FLUX'])
                        flux_err = np.array(data['PDCSAP_FLUX_ERR'])
                    if 'QLP' in data_to_download['project'][i]:
                        #data = data[~np.isinf(data['TIME']) & ~np.isnan(data['TIME']) & ~np.isinf(data['SAP_FLUX']) & ~np.isnan(data['SAP_FLUX']) & (data['SAP_FLUX'] > 1) & (data['PDCSAP_FLUX'] > 1)]
                        time = np.array(data['TIME'])+2457000
                        if 'KSPSAP_FLUX' in data.colnames:
                            flux = np.array(data['KSPSAP_FLUX'])
                            flux_err = np.array(data['KSPSAP_FLUX_ERR'])
                        if 'DET_FLUX' in data.colnames:
                            flux =np.array(data['DET_FLUX'])
                            flux_err = np.array(data['DET_FLUX_ERR'])
                            
                    mask = ~(np.isnan(time) | np.isnan(flux) | np.isnan(flux_err) | np.isinf(time) | np.isinf(flux) | np.isinf(flux_err))
                    # Now sigma clip
                    if sigma is not None : mask = mask & ~sigma_clip(flux, masked=True, sigma=sigma).mask
                    time, flux, flux_err = time[mask], flux[mask], flux_err[mask]
                    if (bin_length is not None) and np.median(np.gradient(time))<(0.5*bin_length): 
                        time, flux, flux_err, c = bin_data(time,flux, bin_length)
                        mask = c>2
                        time, flux, flux_err = time[mask], flux[mask], flux_err[mask]
                    data_return.append(photometry_time_series(time, flux, flux_err))
                    data_return_labels.append('Sector {:}'.format(data_to_download['sector'][i]))

        return np.array(data_return), np.array(data_return_labels)
    
    elif data_type=='northern_duos':
        data_year24 = data_to_download[data_to_download['sector']<=55]
        data_after = data_to_download[data_to_download['sector']>55]
        data_return = []

        # Now download the data
        time, flux, flux_err = np.array([]),np.array([]),np.array([])
        with tempfile.TemporaryDirectory() as tmpdirname:
            for i in range(len(data_year24)):
                out = Observations.download_products(data_year24[i], download_dir=download_dir)
                if out['Status'][0]=='COMPLETE':
                    data = Table.read(out['Local Path'][0])
                    if 'SPOC' in data_year24['project'][i]:
                        data = data[~np.isinf(data['TIME']) & ~np.isnan(data['TIME']) & ~np.isinf(data['SAP_FLUX']) & ~np.isnan(data['SAP_FLUX']) & (data['SAP_FLUX'] > 1) & (data['PDCSAP_FLUX'] > 1)]
                        time = np.concatenate((time, np.array(data['TIME'])+2457000))
                        flux = np.concatenate((flux, np.array(data['PDCSAP_FLUX'])))
                        flux_err = np.concatenate((flux_err, np.array(data['PDCSAP_FLUX_ERR'])))
                    if 'QLP' in data_year24['project'][i]:
                        #data = data[~np.isinf(data['TIME']) & ~np.isnan(data['TIME']) & ~np.isinf(data['SAP_FLUX']) & ~np.isnan(data['SAP_FLUX']) & (data['SAP_FLUX'] > 1) & (data['PDCSAP_FLUX'] > 1)]
                        time = np.concatenate((time, np.array(data['TIME'])+2457000))
                        if 'KSPSAP_FLUX' in data.colnames:
                            flux = np.concatenate((flux, np.array(data['KSPSAP_FLUX'])))
                            flux_err = np.concatenate((flux_err, np.array(data['KSPSAP_FLUX_ERR'])))
                        if 'DET_FLUX' in data.colnames:
                            flux = np.concatenate((flux, np.array(data['DET_FLUX'])))
                            flux_err = np.concatenate((flux_err, np.array(data['DET_FLUX_ERR'])))
            mask = np.isnan(time) | np.isnan(flux) | np.isnan(flux_err) | np.isinf(time) | np.isinf(flux) | np.isinf(flux_err)

            # Now sigma clip
            mask = mask & ~sigma_clip(flux, sigma=3,masked=True).mask

            data_return.append(photometry_time_series(time[~mask], flux[~mask], flux_err[~mask]))
            data_return_labels = ['Years 1 and 2']

            # Now get others
            for i in range(len(data_after)):
                out = Observations.download_products(data_after[i], download_dir=download_dir)
                if out['Status'][0]=='COMPLETE':
                    data = Table.read(out['Local Path'][0])
                    if 'SPOC' in data_after['project'][i]:
                        data = data[~np.isinf(data['TIME']) & ~np.isnan(data['TIME']) & ~np.isinf(data['SAP_FLUX']) & ~np.isnan(data['SAP_FLUX']) & (data['SAP_FLUX'] > 1) & (data['PDCSAP_FLUX'] > 1)]
                        time = np.array(data['TIME'])+2457000
                        flux = np.array(data['PDCSAP_FLUX'])
                        flux_err = np.array(data['PDCSAP_FLUX_ERR'])
                    if 'QLP' in data_after['project'][i]:
                        #data = data[~np.isinf(data['TIME']) & ~np.isnan(data['TIME']) & ~np.isinf(data['SAP_FLUX']) & ~np.isnan(data['SAP_FLUX']) & (data['SAP_FLUX'] > 1) & (data['PDCSAP_FLUX'] > 1)]
                        time = np.array(data['TIME'])+2457000
                        if 'KSPSAP_FLUX' in data.colnames:
                            flux = np.array(data['KSPSAP_FLUX'])
                            flux_err = np.array(data['KSPSAP_FLUX_ERR'])
                        if 'DET_FLUX' in data.colnames:
                            flux =np.array(data['DET_FLUX'])
                            flux_err = np.array(data['DET_FLUX_ERR'])
                mask = np.isnan(time) | np.isnan(flux) | np.isnan(flux_err) | np.isinf(time) | np.isinf(flux) | np.isinf(flux_err)

                # Now sigma clip
                mask = mask & ~sigma_clip(flux, sigma=3,masked=True).mask

                data_return.append(photometry_time_series(time[~mask], flux[~mask], flux_err[~mask]))
                data_return_labels.append('Sector {:}'.format(data_after['sector'][i]))
        return data_return, data_return_labels