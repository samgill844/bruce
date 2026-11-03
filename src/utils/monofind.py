def main():
    from astropy.table import Table
    from astropy.io import fits
    from astropy.time import TimeDelta
    import numpy as np
    np.int =int
    import matplotlib, argparse, bruce, matplotlib.pyplot as plt, warnings, traceback
    from matplotlib.ticker import StrMethodFormatter, NullFormatter
    #from pycheops.ld import stagger_power2_interpolator
    matplotlib.rcParams['axes.formatter.useoffset'] = False
    warnings.filterwarnings('ignore')
    
    import lightkurve, os
    import matplotlib.cm as cm
    from matplotlib import patches
    from scipy.stats import median_abs_deviation
    import lmfit, gc
    from lmfit.models import SineModel
    from scipy.signal import find_peaks 
    from scipy.ndimage.filters import minimum_filter

    from astropy.timeseries import BoxLeastSquares, LombScargle
    from astropy import units as u

    np.int = int 




    def sigma_clip_polyfit(time, flux, order=2, sigma=3, max_iter=10):
        mask = np.isfinite(time) & np.isfinite(flux)

        for _ in range(max_iter):
            # Fit polynomial
            coeffs = np.polyfit(time[mask], flux[mask], order)
            model = np.polyval(coeffs, time)

            # Residuals
            resid = flux - model

            # Robust scatter estimate
            mad = np.median(np.abs(resid[mask] - np.median(resid[mask])))
            std = 1.4826 * mad

            new_mask = np.abs(resid) < sigma * std

            if np.array_equal(mask, new_mask):
                break

            mask = new_mask

        return coeffs, model, mask, np.std(flux[mask])


    def remove_skybright_qlp(t):

        mask = (t['QUALITY']==0)

        segments = bruce.data.find_nights_from_data(t['TIME'][mask], dx_lim=0.5)
        global_idx = np.where(mask)[0]
        for seg in segments:
            seg_idx = global_idx[seg]  # original indices

            time = t['TIME'][seg_idx]
            flux = t['SAP_BKG'][seg_idx]

            coeffs, model, new_mask, std = sigma_clip_polyfit(
                time, flux, order=2, sigma=3, max_iter=10)

            # Remove points rejected by sigma clipping
            mask[seg_idx[~new_mask]] = False

        return t[mask]

    def box_model(x,height,center, width,off=0.) : return height*(center-width/2 < x)*(x < center+width/2) + off

    def _get_BLS_sde(results_BLS, sde, max_peaks=5, seperation=100, time_median=0, frequency_factor=1):
        mins = minimum_filter(results_BLS['power'], 20, mode='nearest')
        trend = np.poly1d(np.polyfit(np.array(results_BLS['period'], dtype = np.float32), mins, 3))(np.array(results_BLS['period'], dtype = np.float32))
        detrended_power = results_BLS['power'] - trend

        # Next plot the detrended SNR and the MAD used to calculate the SDE
        MAD = median_abs_deviation(detrended_power)

        # Now calculate the 
        SDE = detrended_power/MAD

        # Now calculate the peaks
        peaks, meta = find_peaks(SDE, height=sde, distance = seperation / frequency_factor)
        height_best_idx = meta['peak_heights'].argsort()[::-1][:max_peaks]
        peaks = peaks[height_best_idx]
        return {'period' : results_BLS['period'],
                'duration' : results_BLS['duration'],
                'depth' : results_BLS['depth'],
                'epoch' : results_BLS['transit_time'] + time_median*u.day,
                'power' : results_BLS['power'],
                'trend' : trend,
                'detrended_power' : detrended_power,
                'sde' : SDE,
                'peaks' : peaks}


    def BLS(time, flux, durations = np.arange(0.0625,0.525, 0.025), sde=5, max_peaks=5, seperation=100,
                oversample=300, objective='snr' , 
                minimum_n_transit=1, 
                minimum_period=0.6, 
                maximum_period=None, 
                frequency_factor=3.0):
        time_median = int(np.median(time))
        if maximum_period is None : maximum_period = np.max(time) - np.min(time)
        results_BLS = BoxLeastSquares((time - time_median) * u.day, flux, dy=0.01).autopower(durations, 
                                                                                            oversample=oversample, objective=objective , 
                                                                                            minimum_n_transit=minimum_n_transit, 
                                                                                            minimum_period=minimum_period, 
                                                                                            maximum_period=maximum_period, 
                                                                                            frequency_factor=frequency_factor)
        return  _get_BLS_sde(results_BLS, sde=sde, max_peaks=max_peaks, seperation=seperation, time_median=time_median, frequency_factor=frequency_factor) 

    def convert_depth_to_k(depth):
        '''
        Assumed depth is in flux and positive from 0
        i.e. observed depth is 1 - depth
        '''
        depth = 1 - depth
        depth = -2.5*np.log10(depth)
        depth =  np.sqrt(depth)
        if np.isnan(depth) or np.isinf(depth) : return 0.1
        else : return depth

    def convert_width_and_period_to_radius_1(width, period):
        '''
        Assumes width and period are in days
        '''
        radius_1 =  np.pi*width / period
        if np.isnan(radius_1) or np.isinf(radius_1) : return 0.2
        else : return radius_1

    def find_nearest_1d(i, count_mask, saturated_mask):
        if not count_mask[i]: return 
        count_mask[i] = False
        saturated_mask[i] = True 
        
        if (i > 0) : find_nearest_1d(i-1, count_mask, saturated_mask)
        if (i < (count_mask.shape[0] - 1)): find_nearest_1d(i+1, count_mask, saturated_mask)

    def fit_sin(x,y, report=False, fix_freq=True):
        mod = SineModel()
        pars = mod.guess(y, x=x)
        if fix_freq:
            pars['frequency'].min = -np.inf
            pars['frequency'].max = np.inf
            pars['frequency'].value = 1.
            pars['frequency'].vary = False

        out = mod.fit(y, pars, x=x)
        if report : print(out.fit_report(min_correl=0.25))
        return mod, out

    def _transit_model(x, t_zero, period, radius_1, k ,b, zp, light_3, cadence, ld_law, h_1, h_2, fs,fc,SBR):
        # Convert inclination
        incl = np.arccos(radius_1*b)

        # Convert eccentricity and w
        e,w = bruce.sampler.fsfc_to_ew(fs,fc)

        # convert_ldc
        ldc_1, ldc_2 = bruce.sampler.h1h2_to_ca(h_1, h_2)
        return zp*bruce.binarystar.lc(x, t_zero=t_zero,
                    period=period,
                    radius_1=radius_1,
                    k=k,
                    e=e,
                    w=w,
                    incl=incl,
                    ld_law=ld_law,
                    c=ldc_1,
                    alpha=ldc_2,
                    light_3=light_3,
                    cadence=cadence,
                    noversample=10)

    def get_transit_model():
        return lmfit.Model(_transit_model)

    def get_transit_model_params(t_zero=0., period=1., radius_1=0.2, k=0.2 ,b=0., zp=1, light_3=0., cadence=0., ld_law=2, h_1=0.76205086, h_2=0.45017074,  fs=0.,fc=0.,SBR=0.):
        p = lmfit.Parameters()
        p.add( 't_zero', value = t_zero, vary = True, min = t_zero - 0.1, max = t_zero + 0.1)
        p.add( 'period', value = max(period, 1e-3), vary = True, min = max(period - 0.01, 1e-3), max = max(period - 0.01, 1e-3) + 2*0.01)
        p.add( 'radius_1', value = np.clip(radius_1, 1e-3, 0.3), vary = True, min = 1e-3, max = 0.3)
        p.add( 'k', value = np.clip(k, 1e-3, 0.4), vary = True, min = 1e-3, max = 0.4)
        p.add( 'b', value = b, vary = True, min = 0, max = 1.5)
        #p.add('delta', value=b-0.5, vary=False , min=-0.5, max=1, expr='b-k') # constrain the limit b <= 1 + k
        p.add('delta', value=b+0.5, vary=False, min=0, max=1, expr='b+k')
        p.add( 'zp', value = zp, vary = True, min = zp*0.99, max = zp*1.01)
        p.add( 'light_3', value = np.clip(light_3, 0,5), vary = False, min = 0, max = 5)
        p.add( 'cadence', value = cadence, vary = False)
        p.add( 'h_1', value = h_1, vary = False)
        p.add( 'h_2', value = h_2, vary = False)
        p.add( 'ld_law', value = ld_law, vary = False)
        p.add( 'fs', value = fs, vary = False)
        p.add( 'fc', value = fc, vary = False)
        p.add( 'SBR', value = SBR, vary = False)

        return p


    def get_transit_model_phased_params(params):
        out_ = params.copy()
        out_['t_zero'].min = -np.inf
        out_['t_zero'].max = np.inf
        out_['t_zero'].value = 0
        out_['period'].min = -np.inf
        out_['period'].max = np.inf
        out_['period'].value = 1.
        out_['cadence'].value = params['cadence'].value / params['period'].value

        return out_

    def fit_transit(x,y, ye, params, model, report=False,  method='leastsq'):
        if model=='emcee':
            emcee_kws = dict(steps=5000, burn=500, thin=20, is_weighted=False,
                            progress=True, ntemps=20, workers=10,
                            reuse_sampler=False, float_behavior='posterior', skip_initial_state_check=True)
            emcee_params = params.copy()
            emcee_params.add('__lnsigma', value=np.log(0.1), min=np.log(0.001), max=np.log(2.0))
            out = model.fit(y, emcee_params, x=x, weights = 1 / ye, method='emcee', nan_policy='omit', fit_kws=emcee_kws)
        else : out = model.fit(y, params, x=x, weights = 1 / ye, method=method)
        if report : print(out.fit_report(min_correl=0.25))
        return out    




    # Flattening parameters
    segment_cut = 0.1
    sigmaclip = 3
    window_length = None # determined from cadence
    iter = 10

    # BLS parameters
    oversample=20
    frequency_factor=1
    durations = np.arange(0.0125,0.525, 0.025)
    objective='snr'
    minimum_n_transit=1
    minimum_period=0.6
    maximum_period=None
    bls_sde=10
    max_bls_peaks=5
    bls_peak_seperation=100
    bls_fit_method='leastsq'
    bls_fit_report = False

    # LS parameters
    ls_sde=10
    max_ls_peaks=5
    ls_peak_seperation=100
    ls_fit_report = False

    # MF parameters
    MF_mad = 3

    # Plot params
    datacolor = 'grey'
    datasize = 1
    model_color = 'r'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    arrowprops = dict(
        arrowstyle = "-",
        connectionstyle = "angle,angleA=90,angleB=0,rad=10")
    thumnnail_figsize = (5,2)
    BTJD = 2457000
    mf_axvspan_alpha = 0.1
    plt.rcParams.update({'font.size': 5})
    thumnnail_dpi = 200
    MF_figsize = (7,2*6)
    masked_alpha = 0.1

    BLS_figsize = (10,2*5)

    def parse_args():
        description = '''A program to searc spoc lightcurves for transit signals'''

        # Argument parser
        parser = argparse.ArgumentParser('spocsearch', description=description)

        parser.add_argument("filename",
                        help='The filename of the SPOC lightcurve')

        parser.add_argument('-a', 
                        '--segment_cut',
                        help='The gradient threshold in which to split the segments. [1.]',
                        default=0.1, type=float)
        parser.add_argument('-b', 
                        '--sigmaclip',
                        help='The sigma clip cut.',
                        default=3, type=int)

        parser.add_argument('-c', 
                        '--window_length',
                        help='A scale parameter for the SG filter (111 for the primary, 333 for the extended)',
                        default=403, type=int)
        parser.add_argument('-d', 
                        '--iter',
                        help='The number of savgol iterations',
                        default=5, type=int)   
        parser.add_argument('-e', 
                        '--savedir',
                        help='The save directory',
                        default='.', type=str)  
        
        parser.add_argument('-f', 
                        '--RUN',
                        help='The processing run',
                        default=0, type=int) 
        parser.add_argument('-g', 
                        '--mf_method',
                        help='The monofind method',
                        default='leastsq', type=str) 
        parser.add_argument('--mf', action="store_true", default=False, help="Plot the FOV")
        parser.add_argument('--bls', action="store_true", default=False, help="Plot the FOV")
        parser.add_argument('--ls', action="store_true", default=False, help="Plot the FOV")
        parser.add_argument('--searched', action="store_true", default=False, help="Plot the FOV")
        parser.add_argument('--delete_files', action="store_true", default=False, help="Plot the FOV")
        parser.add_argument('--onlyone', action="store_true", default=False, help="Keep only 1 MF result, otherwise abort")
        parser.add_argument('--adaptive_filter', action="store_true", default=False, help="Keep dividing the filter size by two until noise matches predicitons from Kunimote et al")

        return parser.parse_args()

    def find_mono_events(time, flux, mad=3.):
        # Estimate the cadence
        cadence = np.median(np.gradient(time))
        
        # Calcualte the height, distances, and find the peaks
        height = mad*median_abs_deviation(flux) / (np.sqrt(cadence / 2.31481e-5) / np.sqrt(0.5/24 / 2.31481e-5))
        distance = 48 * ((0.5 / 24) / cadence)
        peaks, meta = find_peaks(1-flux, height=height, distance = 24*2)
        
        dele = []
        for i in range(len(peaks)):
            if cadence > 0.0138889 : # 20 minutes to catch 30 min data
                if ((1 - flux[np.clip(peaks[i]-1, 0, flux.shape[0]-1)]) < height) or ((1 - flux[np.clip(peaks[i]+1, 0, flux.shape[0]-1)]) < height) : 
                    # Now check the 2 to the side or 2 to the other
                    check_left = ((1 - flux[np.clip(peaks[i]+1, 0, flux.shape[0]-1)]) > height) and ((1 - flux[np.clip(peaks[i]+2, 0, flux.shape[0]-1)]) > height)
                    check_right = ((1 - flux[np.clip(peaks[i]-1, 0, flux.shape[0]-1)]) > height) and ((1 - flux[np.clip(peaks[i]-2, 0, flux.shape[0]-1)]) > height)
                    if (not check_left) or (not check_right) : dele.append(i)

            elif cadence > 0.00347222 : # 5 minutes to catch 10 minute data
                idxs = np.clip(np.arange(peaks[i] - 2, peaks[i]+3, dtype = int), 0, flux.shape[0])
                if np.sum((1 - flux[idxs]) < height) >=2 : dele.append(i)

                
        peaks = np.delete(peaks, dele)
        meta['peak_heights'] = np.delete(meta['peak_heights'], dele)
        meta['threshold'] = 1 - height
        
        over_mask = (1-flux) > height / mad
        widths = []
        depths = []
        startstop = []
        for i in range(len(peaks)) : 
            peak_mask = np.zeros(flux.shape[0], dtype = bool)
            peak_mask[peaks[i]] = True
            find_nearest_1d(peaks[i], over_mask, peak_mask)
            widths.append(np.max(time[peak_mask]) - np.min(time[peak_mask]))
            startstop.append([np.min(time[peak_mask]), np.max(time[peak_mask])])

            depths.append(1 - flux[peaks[i]])
        meta['widths'] = widths
        meta['depths'] = depths
        meta['startstop'] = startstop


        # New addition
        meta['widths'] = np.array(widths)
        meta['depths'] = np.array(depths)
        meta['startstop'] = np.array(startstop)

        # Remove detections with widths greater than 1 day
        keep = meta['widths'] <= 1.0

        peaks = peaks[keep]
        for key in ['peak_heights', 'widths', 'depths', 'startstop']:
            meta[key] = np.asarray(meta[key])[keep]

        return peaks, meta

    def noise_est(T, model='10th'):
        """Estimates noise of TESS lightcurve for a star based on magnitude

        Args:
            T (float): TESS magnitude(s) of star
            model (str, optional): fit of noise model used either 10th or 50th. Defaults to '10th'.

        Returns:
            noise (float): 1hr estimated noise in ppm
        """
        if model == '10th':
            C = [23.1, 140.2, 50.1]
        if model =='50th':
            C = [50.2, 97.4, 92.9]
        noise = C[0] + C[1]*np.power(10,(0.2*(T-10.))) + C[2]*np.power(10,(0.4*(T-10)))
        return noise
        


    def make_mf_tpf_plot_1(cadence_T0, cadence_Tc, cadence_T4, c):
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 6
        fig, ax = plt.subplots(1,3, figsize = (6,2.2))
        c.plot(ax=ax[0], show_colorbar=False, title=None, aperture_mask='pipeline', scale='sqrt', cmap=cm.get_cmap('Greys_r'), cadenceno=cadence_T0, aspect='equal')
        c.plot(ax=ax[1], show_colorbar=False, title=None, aperture_mask='pipeline', scale='sqrt', cmap=cm.get_cmap('Greys_r'), cadenceno=cadence_Tc, aspect='equal')
        c.plot(ax=ax[2], show_colorbar=False, title=None, aperture_mask='pipeline', scale='sqrt', cmap=cm.get_cmap('Greys_r'), cadenceno=cadence_T4, aspect='equal')
        [i.set_alpha(0.3) for i in ax[0].patches]
        [i.set_alpha(0.3) for i in ax[1].patches]
        [i.set_alpha(0.3) for i in ax[2].patches]

        
        ax[0].set(title='INGRESS')
        ax[1].set(title='CENTER', ylabel=None, yticks=[])
        ax[2].set(title='EGRESS', ylabel=None, yticks=[])
        plt.subplots_adjust(left = 0.07, bottom = -0.1, right = 0.99, top=1.2)
        
        return fig, ax

    def make_mf_tpf_plot_2(cadence_Tm1, cadence_T0, cadence_Tc, cadence_T4, cadence_Tp1, c):
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 6
        fig, ax = plt.subplots(1,5, figsize = (6,1.5))
        c.plot(ax=ax[0], show_colorbar=False, title=None, aperture_mask='pipeline', scale='sqrt', cmap=cm.get_cmap('Greys_r'), cadenceno=cadence_Tm1, aspect='equal')
        c.plot(ax=ax[1], show_colorbar=False, title=None, aperture_mask='pipeline', scale='sqrt', cmap=cm.get_cmap('Greys_r'), cadenceno=cadence_T0, aspect='equal')
        c.plot(ax=ax[2], show_colorbar=False, title=None, aperture_mask='pipeline', scale='sqrt', cmap=cm.get_cmap('Greys_r'), cadenceno=cadence_Tc, aspect='equal')
        c.plot(ax=ax[3], show_colorbar=False, title=None, aperture_mask='pipeline', scale='sqrt', cmap=cm.get_cmap('Greys_r'), cadenceno=cadence_T4, aspect='equal')
        c.plot(ax=ax[4], show_colorbar=False, title=None, aperture_mask='pipeline', scale='sqrt', cmap=cm.get_cmap('Greys_r'), cadenceno=cadence_Tp1, aspect='equal')
        
        [i.set_alpha(0.3) for i in ax[0].patches]
        [i.set_alpha(0.3) for i in ax[1].patches]
        [i.set_alpha(0.3) for i in ax[2].patches]
        [i.set_alpha(0.3) for i in ax[3].patches]
        [i.set_alpha(0.3) for i in ax[4].patches]

        
        ax[0].set(title='TC - WIDTH')
        ax[1].set(title='INGRESS', ylabel=None, yticks=[])
        ax[2].set(title='CENTER', ylabel=None, yticks=[])
        ax[3].set(title='EGRESS', ylabel=None, yticks=[])
        ax[4].set(title='TC + WIDTH', ylabel=None, yticks=[])

        plt.subplots_adjust(left = 0.072, bottom = -0.1, right = 0.99, top=1.2)
        
        return fig, ax


    def make_mf_tpf_plot_3(cadence_Tm1, cadence_Tc, cadence_Tp1, c):
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 6
        fig, ax = plt.subplots(1,1, figsize=(4,4))
        c.plot(ax=ax, show_colorbar=False, title=None, aperture_mask='pipeline', scale='sqrt', cmap=cm.get_cmap('Greys_r'), cadenceno=cadence_Tc, aspect='equal')

        [i.set_alpha(0.3) for i in ax.patches]
        [i.set_color('green') for i in ax.patches]

        ax.set_title(None)
        
        for i in range(c.flux.shape[1]):
            for j in range(c.flux.shape[2]):
                mask = (c.cadenceno > cadence_Tm1) & (c.cadenceno < cadence_Tp1)
                # Now need to scale data between j and j+1 in x
                X = c.time.btjd[mask]
                X = i  + (X - np.min(X)) / (np.max(X) - np.min(X)) + ax.get_xbound()[0]
                # Y needs to be scaled between i and i+1 
                Y =  c.flux[mask,i,j] 
                Y = (j  + (Y - np.min(Y)) / (np.max(Y) - np.min(Y))) + ax.get_ybound()[0]

                ax.plot(X,Y, c='r', lw=1, alpha = 0.5)
                        
        plt.subplots_adjust(left=0.1, bottom=0.08, right=0.99, top=0.99)
        return fig, ax


    # Parse the arguments
    args = parse_args() 
    
    # Lets load the TPF -5:-5 = 'tp'
    tpf_filename = args.filename
    tpf_filename = tpf_filename.replace('lc', 'tp')

    # Load the filename 
    data = Table.read(args.filename)

    # Here we need to hack this to work with QLP
    # PDCSAP_FLUX -> KSPSAP_FLUX
    QLP = 0
    if 'PDCSAP_FLUX' not in data.colnames:
        QLP=1
        data = remove_skybright_qlp(data)
        # here it is QLP, so lets hack
        if 'DET_FLUX' not in data.colnames : data['DET_FLUX'] = data['KSPSAP_FLUX']
        data['PDCSAP_FLUX'] = data['SAP_FLUX'].copy()
        data['PDCSAP_FLUX_ERR'] = np.ones(len(data))*median_abs_deviation(data['SAP_FLUX'])
        
        data['PSF_CENTR1'] = data['SAP_X'].copy()
        data['PSF_CENTR1_ERR'] = median_abs_deviation(data['SAP_X'])
        data['MOM_CENTR1'] = data['SAP_X'].copy()
        data['MOM_CENTR1_ERR'] = median_abs_deviation(data['SAP_X'])

        data['PSF_CENTR2'] = data['SAP_Y'].copy()
        data['PSF_CENTR2_ERR'] = median_abs_deviation(data['SAP_Y'])
        data['MOM_CENTR2'] = data['SAP_Y'].copy()
        data['MOM_CENTR2_ERR'] = median_abs_deviation(data['SAP_Y'])



    data = data[~np.isinf(data['TIME']) & ~np.isnan(data['TIME']) & ~np.isinf(data['SAP_FLUX']) & ~np.isnan(data['SAP_FLUX']) ] #& (data['SAP_FLUX'] > 1) & (data['PDCSAP_FLUX'] > 1)
    if (len(data) < 100) : 
        if args.delete_files:
            print('Deleting input files')
            os.system('rm -vf {:}'.format(args.filename))
            os.system('rm -vf {:}'.format(tpf_filename))
        raise ValueError('Masked too many values for {:}'.format(args.filename))
    
    data_masked = data[(data['QUALITY']==0) & ~np.isnan(data['QUALITY']) & ~np.isnan(data['PDCSAP_FLUX']) & ~np.isinf(data['PDCSAP_FLUX'])]
    data_masked['TREND'] = np.zeros(len(data_masked), dtype = np.float32)
    data_masked['DETRENDED'] = np.zeros(len(data_masked), dtype = np.float32)
    sector =  int(fits.open(args.filename)[0].header['SECTOR'])
    camera = int(fits.open(args.filename)[0].header['CAMERA'])
    ccd = int(fits.open(args.filename)[0].header['CCD'])
    TICID =  int(fits.open(args.filename)[0].header['TICID'])



    try : tpf = lightkurve.TessTargetPixelFile(fits.open(tpf_filename))
    except : tpf = None


    # Get png base
    png_base = '{:}/TIC-{:}_S{:02}_{:}-{:}_RUN{:}_'.format(args.savedir, TICID, sector, camera, ccd, args.RUN)



    try : TEFF = np.clip(4000,8000, int(fits.open(args.filename)[0].header['TEFF']))
    except : TEFF = 5777
    try : LOGG = np.clip(3.5,5, fits.open(args.filename)[0].header['LOGG'])
    except : LOGG = 4.7
    #c, alpha, h_1, h_2 = stagger_power2_interpolator('TESS')(TEFF, LOGG, 0.)
    #if np.isnan(h_1) or np.isinf(h_1) or np.isnan(h_2) or np.isinf(h_2):
    #    c, alpha, h_1, h_2 = stagger_power2_interpolator('TESS')(5777, 4.44, 0.)
    
    
    c, alpha, h_1, h_2 = 0.65289344, 0.57252791, 0.7861208 , 0.43901413
    # Now find the segments
    segments = bruce.data.find_nights_from_data(data_masked['TIME'], args.segment_cut)
    print('Segments found {:}'.format(len(segments)))

    # Now flatten PDCSAP flux
    cadence = np.median(np.gradient(data_masked['TIME']))*1440 # Cadence in minutes
    if window_length is None:
        if cadence > 20 : window_length = 101 # For 30 minute data
        elif cadence > 5 : window_length = 303 # for 10 minute data 
        else : window_length = 2003 # for eventual 120-s data 

    Tmag = fits.open(args.filename)[0].header['TESSMAG']
    estimated_noise_ppm = noise_est(Tmag, model='10th')
    if 1:
        # while True:
        #     for seg in segments : 
        #         try : 
        #             data_masked['TREND'][seg] = bruce.data.flatten_data_with_function(data_masked['TIME'][seg] + 2457000, data_masked['PDCSAP_FLUX'][seg], 
        #                                                                     SG_iter=iter, SG_window_length=window_length, SG_sigma=sigmaclip,
        #                                                                     SG_polyorder=3, SG_deriv=0, SG_delta=1.)
        #         except : 
        #             try : data_masked['TREND'][seg] = np.poly1d(np.polyfit(data_masked['TIME'][seg] - np.median(data_masked['TIME'][seg]), data_masked['PDCSAP_FLUX'][seg], 1))(data_masked['TIME'][seg] - np.median(data_masked['TIME'][seg]))
        #             except : data_masked['TREND'][seg] = np.ones(len(data_masked['TREND'][seg]))*np.median(data_masked['PDCSAP_FLUX'][seg])

        #         if QLP==1 : 
        #             trend =  data_masked['SAP_FLUX']/data['KSPSAP_FLUX']
        #             mask = ((data_masked['TIME'][seg]-data_masked['TIME'][seg].min())<1.5) | ((data_masked['TIME'][seg]-data_masked['TIME'][seg].max())>-1.5)
        #             trend_seg = data_masked['TREND'][seg]
        #             trend_seg[mask] = np.interp(data_masked['TIME'][seg][mask], data_masked['TIME'], trend)
        #             data_masked['TREND'][seg] = trend_seg

        while True:
            for seg in segments:
                try:
                    data_masked['TREND'][seg] = bruce.data.flatten_data_with_function(
                        data_masked['TIME'][seg] + 2457000,
                        data_masked['PDCSAP_FLUX'][seg],
                        SG_iter=iter,
                        SG_window_length=window_length,
                        SG_sigma=sigmaclip,
                        SG_polyorder=3,
                        SG_deriv=0,
                        SG_delta=1.
                    )
                except Exception:
                    try:
                        t = data_masked['TIME'][seg]
                        f = data_masked['PDCSAP_FLUX'][seg]
                        tc = t - np.median(t)

                        data_masked['TREND'][seg] = np.poly1d(
                            np.polyfit(tc, f, 1)
                        )(tc)
                    except Exception:
                        data_masked['TREND'][seg] = np.ones(
                            len(data_masked['TREND'][seg])
                        ) * np.median(data_masked['PDCSAP_FLUX'][seg])

                if QLP == 1:
                    t = data_masked['TIME'][seg]
                    trend = data_masked['SAP_FLUX'] / data['KSPSAP_FLUX']

                    mask = ((t - t.min()) < 1.5) | ((t - t.max()) > -1.5)

                    trend_seg = data_masked['TREND'][seg].copy()
                    trend_seg[mask] = np.interp(
                        t[mask],
                        data_masked['TIME'],
                        trend
                    )
                    data_masked['TREND'][seg] = trend_seg


            _,f,_, _ = bruce.data.bin_data(np.array(data_masked['TIME']), np.array(data_masked['PDCSAP_FLUX'] / data_masked['TREND']), 1/24)
            from astropy.stats import sigma_clip
            mask = ~sigma_clip(f, masked=True).mask
            measured_noise_ppm = 1e6*np.std(f[mask])
            print('Window length {:}    noise  {:,}    estimated {:,}    [{:}]'.format(window_length, int(measured_noise_ppm), int(estimated_noise_ppm), measured_noise_ppm/estimated_noise_ppm))
            if args.adaptive_filter==False : 
                print('Not using adaptive filter - breaking normalisation')
                break
            if (measured_noise_ppm/estimated_noise_ppm) > 1.5 : 
                window_length /= 2
                window_length = int(window_length)
                if not window_length%2 : window_length +=1
                if window_length < 2 : break
            else : break
    # else:
    #     for seg in segments :
    #         data_masked['TREND'][seg] = np.ones(len(data_masked[seg]))*np.median(data['DET_FLUX'][seg])

    #exit()


    # Now lets do a Lomb-scargle and BLS
    if args.bls:
        print('Running BLS...', flush=True)
        BLS = BLS(np.array(data_masked['TIME'] + 2457000), np.array(data_masked['PDCSAP_FLUX'] / data_masked['TREND']), 
                            oversample=oversample, frequency_factor=frequency_factor, durations = durations, objective=objective,
                            minimum_n_transit=minimum_n_transit, minimum_period=minimum_period, maximum_period=maximum_period,
                            sde=bls_sde, max_peaks=max_bls_peaks, seperation=bls_peak_seperation)

    # if args.ls:
    #     print('Running LS...', flush=True)
    #     LS = bruce2.utils.LS(np.array(data_masked['TIME'] + 2457000), np.array(data_masked['PDCSAP_FLUX']), BLS['period'],
    #                         sde=ls_sde, max_peaks=max_ls_peaks, seperation=ls_peak_seperation)

    if args.mf:
        print('Running MF...',flush=True)
        if sector > 26:
            print('Extended mission data, binning data to 30 mins for detection.')
            t_, f_, fe_, c_ = bruce.data.bin_data(data_masked['TIME'] + 2457000, data_masked['PDCSAP_FLUX']/data_masked['TREND'], 0.5/24)
            MFpeaks, MFmeta = find_mono_events(t_, f_, mad = MF_mad)
            MFpeaks = np.interp(t_[MFpeaks], data_masked['TIME'] + 2457000 ,np.arange(len(data_masked))).astype(int)
        else:
            MFpeaks, MFmeta = find_mono_events(data_masked['TIME'] + 2457000, data_masked['PDCSAP_FLUX']/data_masked['TREND'], mad = MF_mad)
        if args.onlyone and (len(MFpeaks)!=1):
            print('Found {:} events but only want 1, so quitting.'.format(len(MFpeaks)))
            if args.delete_files:
                print('Deleting input files')
                os.system('rm -vf {:}'.format(args.filename))
                os.system('rm -vf {:}'.format(tpf_filename))
            exit()


    # Initialise the transit model
    transit_model =  get_transit_model()
    if args.bls:
        try:
            # Now fit BLS periods
            print('Fitting BLS...', flush=True)
            BLS['fitted_params'] = []
            for i in range(len(BLS['peaks'])):
                print('Fitting BLS PEAK {:}...'.format(i+1), flush=True)
                # Initialse the parameters
                params =  get_transit_model_params(t_zero = BLS['epoch'][BLS['peaks']][i].value,
                                                            period = BLS['period'][BLS['peaks']][i].value,
                                                            radius_1 = convert_width_and_period_to_radius_1(BLS['duration'][BLS['peaks']][i].value, BLS['period'][BLS['peaks']][i].value),
                                                            k = convert_depth_to_k(BLS['depth'][BLS['peaks']][i]),
                                                            b = 0.5,
                                                            zp = 1.0,
                                                            cadence = np.median(np.gradient(data_masked['TIME'])),
                                                            h_1=h_1, h_2=h_2)
                #print(params)
                # Create the mask and fit the data
                phase = bruce.data.phase_times(data_masked['TIME'] + 2457000, BLS['epoch'][BLS['peaks']][i].value, BLS['period'][BLS['peaks']][i].value, 0.2)
                phase_width =  BLS['duration'][BLS['peaks']][i].value / BLS['period'][BLS['peaks']][i].value
                phase_widen= 1
                mask = (phase > -(phase_widen*phase_width))  & (phase < (phase_widen*phase_width))
                while len(data_masked['TIME'][mask]) < 10:
                    phase_widen += 1
                    mask = (phase > -(phase_widen*phase_width))  & (phase < (phase_widen*phase_width))
                out = fit_transit(np.array(data_masked['TIME'] + 2457000)[mask], np.array(data_masked['PDCSAP_FLUX'] / data_masked['TREND'])[mask], np.array(data_masked['PDCSAP_FLUX_ERR'] / data_masked['TREND'])[mask], 
                                            params=params, model=transit_model, report=bls_fit_report, method=bls_fit_method)
                BLS['fitted_params'].append(out)
        except Exception as e: 
            print('File : {:} has failed to BLS fit with error:\n{:}\n{:}'.format(args.filename, e, traceback.format_exc()))


    # if args.ls:
    #     try:
    #         # Now fit the LS periods
    #         print('Fitting LS...', flush=True)
    #         LS['fitted_params'] = []
    #         for i in range(len(LS['peaks'])):
    #             print('Fitting LS PEAK {:}...'.format(i+1), flush=True)
    #             phase = bruce2.data.phase_times(data_masked['TIME'] + 2457000, 0., BLS['period'][LS['peaks']][i].value, 0.2)
    #             mod, out = bruce2.utils.lmfit.fit_sin(2*np.pi*phase, np.array(data_masked['PDCSAP_FLUX']) - np.median(data_masked['PDCSAP_FLUX']), report=ls_fit_report)
    #             LS['fitted_model'] = mod
    #             LS['fitted_params'].append(out)
    #     except Exception as e: 
    #         print('File : {:} has failed to LS fit with error:\n{:}\n{:}'.format(args.filename, e, traceback.format_exc()))


    if args.mf:
        try:
            # Now fit the MF events
            print('Fitting MF events...', flush=True)
            MFmeta['fitted_params'] = []
            for i in range(len(MFpeaks)):
                print('Fitting MF event {:}...'.format(i+1), flush=True)
                params =  get_transit_model_params(t_zero = data_masked['TIME'][MFpeaks[i]] + 2457000 ,
                                                            period = 30.,
                                                            radius_1 = convert_width_and_period_to_radius_1(MFmeta['widths'][i], 30.),
                                                            k = convert_depth_to_k(1 - (data_masked['PDCSAP_FLUX']/data_masked['TREND'])[MFpeaks[i]]),
                                                            b = 0.5,
                                                            zp = 1.0,
                                                            cadence =np.median(np.gradient(data_masked['TIME'])),
                                                            h_1=h_1, h_2=h_2) 
                params['period'].vary = False
                data_event_mask = (data_masked['TIME']>(data_masked['TIME'][MFpeaks[i]] - MFmeta['widths'][i])) & (data_masked['TIME']<(data_masked['TIME'][MFpeaks[i]] + MFmeta['widths'][i]))
                out = fit_transit(np.array(data_masked['TIME'] + 2457000)[data_event_mask], np.array(data_masked['PDCSAP_FLUX'] / data_masked['TREND'])[data_event_mask], np.array(data_masked['PDCSAP_FLUX_ERR'] / data_masked['TREND'])[data_event_mask], 
                                                    params=params, model=transit_model, report=False, method=args.mf_method)
                MFmeta['fitted_params'].append(out)

        except Exception as e: 
            print('File : {:} has failed to MF fit with error:\n{:}\n{:}'.format(args.filename, e, traceback.format_exc()))
        

    if args.mf:
        print('Writing MF fits', flush=True)
        # Now we need to save the output
        MF_TABLE_NAMES  =  ['tic_id','run', 'sector', 'j', 
                            't_zero', 'period', 'radius_1', 'k', 'b', 'zp',
                            'h_1', 'h_2',
                            'ldc_1', 'ldc_2',
                            'MF_width', 'fitted_width', 'depth', 
                            'x_cen', 'y_cen',
                            'threshold', 'distance', 
                            'chisqr',   'red_chisqr', 'AIC',       'BIC']
        MF_TABLE_DTYPES = [int,int, int, int,
                        np.float64, np.float32,np.float32,np.float32,np.float32,np.float32,
                        np.float32, np.float32,
                        np.float32,np.float32,
                        np.float32,np.float32,np.float32,
                        np.float32,np.float32,
                        np.float32,np.float32,
                        np.float32,np.float32,np.float32,np.float32]
        MF_TABLE = Table(names = MF_TABLE_NAMES, dtype = MF_TABLE_DTYPES)
        for i in range(len(MFpeaks)):
            c, a = bruce.sampler.h1h2_to_ca(MFmeta['fitted_params'][i].params['h_1'].value, MFmeta['fitted_params'][i].params['h_1'].value)
            fitted_width = bruce.binarystar.transit_width(MFmeta['fitted_params'][i].params['radius_1'].value, 
                                                    MFmeta['fitted_params'][i].params['k'].value,
                                                    MFmeta['fitted_params'][i].params['b'].value, period=MFmeta['fitted_params'][i].params['period'].value)
            row = [TICID,args.RUN,sector, i,
                MFmeta['fitted_params'][i].params['t_zero'].value, MFmeta['fitted_params'][i].params['period'].value, MFmeta['fitted_params'][i].params['radius_1'].value, MFmeta['fitted_params'][i].params['k'].value, MFmeta['fitted_params'][i].params['b'].value, MFmeta['fitted_params'][i].params['zp'].value,
                MFmeta['fitted_params'][i].params['h_1'].value, MFmeta['fitted_params'][i].params['h_1'].value,
                c,a,
                MFmeta['widths'][i],  fitted_width,  MFmeta['depths'][i],
                np.median(data_masked['MOM_CENTR1']), np.median(data_masked['MOM_CENTR2']),
                MFmeta['threshold'], 100,
                MFmeta['fitted_params'][i].chisqr, MFmeta['fitted_params'][i].redchi, MFmeta['fitted_params'][i].aic, MFmeta['fitted_params'][i].bic]
            MF_TABLE.add_row(row)
        MF_TABLE.write(png_base + 'MF.fits' , overwrite=True, format='fits')


    if args.bls:
        print('Writing BLS fits', flush=True)

        # now we need to save the BLS results
        BLS_TABLE_NAMES  = ['tic_id', 'run','sector', 'j', 'sde',
                            'bls_t_zero', 'bls_period', 'bls_width', 'bls_depth',
                            't_zero', 'period', 'radius_1', 'k', 'b', 'zp',
                            'h_1', 'h_2',
                            'ldc_1', 'ldc_2',
                            'width', 'density','x_cen', 'y_cen',
                            'chisqr',   'red_chisqr', 'AIC',       'BIC']
        BLS_TABLE_DTYPES = [int, int, int,int,np.float32,
                        np.float32,np.float32,np.float32,np.float32,
                        np.float64, np.float32,np.float32,np.float32,np.float32,np.float32,
                        np.float32,np.float32,
                        np.float32,np.float32,
                        np.float32,np.float32,np.float32,np.float32,
                        np.float32,np.float32,np.float32,np.float32]
        BLS_TABLE = Table(names = BLS_TABLE_NAMES, dtype = BLS_TABLE_DTYPES)
        for i in range(len(BLS['peaks'])):
            c, a = bruce.sampler.h1h2_to_ca(BLS['fitted_params'][i].params['h_1'].value, BLS['fitted_params'][i].params['h_1'].value)
            fitted_width = bruce.binarystar.transit_width(BLS['fitted_params'][i].params['radius_1'].value, 
                                                    BLS['fitted_params'][i].params['k'].value,
                                                    BLS['fitted_params'][i].params['b'].value, period=BLS['fitted_params'][i].params['period'].value)
            density = bruce.binarystar.stellar_density(radius_1=BLS['fitted_params'][i].params['radius_1'].value, 
                                                    P=BLS['fitted_params'][i].params['period'].value, q=0, e = 0., w = np.pi/2.)
            row = [TICID, args.RUN,sector, i,BLS['sde'][BLS['peaks']][i],
                    BLS['epoch'][BLS['peaks'][i]].value, BLS['period'][BLS['peaks'][i]].value, BLS['duration'][BLS['peaks'][i]].value, BLS['depth'][BLS['peaks'][i]],
                    BLS['fitted_params'][i].params['t_zero'].value, BLS['fitted_params'][i].params['period'].value, BLS['fitted_params'][i].params['radius_1'].value, BLS['fitted_params'][i].params['k'].value, BLS['fitted_params'][i].params['b'].value, BLS['fitted_params'][i].params['zp'].value,
                    BLS['fitted_params'][i].params['h_1'].value, BLS['fitted_params'][i].params['h_2'].value,
                    c,a,
                    fitted_width, density, np.median(data_masked['MOM_CENTR1']), np.median(data_masked['MOM_CENTR2']),
                    BLS['fitted_params'][i].chisqr, BLS['fitted_params'][i].redchi, BLS['fitted_params'][i].aic, BLS['fitted_params'][i].bic]
            BLS_TABLE.add_row(row)
        BLS_TABLE.write(png_base + 'BLS.fits' , overwrite=True, format='fits')


    # if args.ls:
    #     print('Writing LS fits', flush=True)

    #     # Now save LS data
    #     LS_TABLE_NAMES  = ['tic_id', 'run', 'j', 
    #                         'period', 'sde', 
    #                         'amplitude', 'frequency', 'shift',
    #                         'x_cen', 'y_cen',
    #                         'chisqr',   'red_chisqr', 'AIC',       'BIC']
    #     LS_TABLE_DTYPES = [int,int, int,
    #                     np.float32,np.float32,
    #                     np.float64, np.float32,np.float32,
    #                     np.float32,np.float32,
    #                     np.float32,np.float32,np.float32,np.float32]
    #     LS_TABLE = Table(names = LS_TABLE_NAMES, dtype = LS_TABLE_DTYPES)

    #     for i in range(len(LS['peaks'])):
    #         row = [TICID, args.RUN, i,
    #         BLS['period'][LS['peaks']][i].value, LS['sde'][LS['peaks']][i],
    #         LS['fitted_params'][i].params['amplitude'].value, LS['fitted_params'][i].params['frequency'].value, LS['fitted_params'][i].params['shift'].value,
    #         np.median(data_masked['MOM_CENTR1']), np.median(data_masked['MOM_CENTR2']),
    #         LS['fitted_params'][i].chisqr, LS['fitted_params'][i].redchi, LS['fitted_params'][i].aic, LS['fitted_params'][i].bic]
            
    #         LS_TABLE.add_row(row)
    #     LS_TABLE.write(png_base + 'LS.fits' , overwrite=True, format='fits')   

    print('Updating fits', flush=True)
    def update_fits_table(fits_file, NDET):
        with fits.open(fits_file, 'update') as f:
            f[1].header['TICID'] = TICID
            f[1].header['SECTOR'] = sector
            f[1].header['CAMERA'] = camera
            f[1].header['CCD'] = ccd
            f[1].header['NDET'] = NDET
            f[1].header['RUN'] = args.RUN

    if args.mf : update_fits_table(png_base + 'MF.fits', len(MFpeaks))
    if args.bls : update_fits_table(png_base + 'BLS.fits', len(BLS['peaks']))
    # if args.ls : update_fits_table(png_base + 'LS.fits', len(LS['peaks']))
    


    def plot_data(ax, x, y, segments, line_plot=True):
        for seg in segments:
            ax.scatter(x[seg], y[seg], c=datacolor, s = 0.2)
            if line_plot : ax.plot(x[seg], y[seg], color=datacolor, lw=0.2, ls='--')


    if args.mf:
        try:
            # Plot thumbnails
            # Create the figure
            print('Plotting MF thumbail', flush=True)
            f, ax = plt.subplots(nrows=1, ncols=1, figsize = thumnnail_figsize)
            #ax.errorbar(data_masked['TIME'], data_masked['PDCSAP_FLUX']/data_masked['TREND'], yerr= data_masked['PDCSAP_FLUX_ERR']/data_masked['TREND'], 
            #            fmt='.', color=datacolor, markersize=datasize, linewidth=1)
            plot_data(ax, data_masked['TIME'], data_masked['PDCSAP_FLUX']/data_masked['TREND'], segments)

            ax.set(xlabel= 'BJD - 2457000 [S{:}]'.format(sector), ylabel='PDCSAP FLUX', xlim=[np.min(data_masked['TIME']), np.max(data_masked['TIME'])])
            for i in range(len(MFpeaks)):
                ax.axvspan(data_masked['TIME'][MFpeaks[i]]-MFmeta['widths'][i]/2, data_masked['TIME'][MFpeaks[i]]+MFmeta['widths'][i]/2, alpha=mf_axvspan_alpha, color='red')
                ax.annotate('E{:}'.format(i+1) , (data_masked['TIME'][MFpeaks[i]], (data_masked['PDCSAP_FLUX']/data_masked['TREND'])[MFpeaks[i]]), (data_masked['TIME'][MFpeaks[i]] - 2., (data_masked['PDCSAP_FLUX']/data_masked['TREND'])[MFpeaks[i]]) , bbox = props, arrowprops=arrowprops)
            ax.axhline(MFmeta['threshold'], c='r', ls='--', alpha = 0.3, label='Threshold = {:,} mmag [{:,} PPM]'.format(int((1 - MFmeta['threshold'])*1e3), int((1 - 10**(-0.4*(1 - MFmeta['threshold'])))*1e6) ))
            ax.legend()

            ax_twin = ax.twinx()
            for i in range(len(MFpeaks)):
                mask_bkg = (data_masked['TIME'] > (data_masked['TIME'][MFpeaks[i]]-10*MFmeta['widths'][i]))
                mask_bkg = mask_bkg & (data_masked['TIME'] < (data_masked['TIME'][MFpeaks[i]]+10*MFmeta['widths'][i]))
                X = data_masked['TIME'][mask_bkg]
                Y = data_masked['SAP_BKG'][mask_bkg]
                Y = (Y - Y.min()) / (Y.max() - Y.min())
                ax_twin.plot(data_masked['TIME'][MFpeaks[i]] + 10*(X - data_masked['TIME'][MFpeaks[i]]),Y,'orange', lw=0.4, zorder=0, alpha = 0.8)
                ax_twin.set_ylim(-0.1,1.5)


            plt.subplots_adjust(left=0.09, bottom=0.148, right=0.995, top=0.983, wspace=None, hspace=None)
            plt.savefig(png_base + 'MONOFIND_THUMBNAIL.webp', dpi=thumnnail_dpi,transparent=True)
            plt.close()

            def get_MF_full_base(tmin = -np.inf, tmax = np.inf):
                mask= (data['TIME'] > tmin) & (data['TIME'] < tmax)
                mask_masked = (data_masked['TIME'] > tmin) & (data_masked['TIME'] < tmax)

                f, ax = plt.subplots(nrows=6, ncols=1, figsize = MF_figsize, sharex=True)


                ax[0].yaxis.set_major_formatter('{x:9<5.3f}')
                #ax[0].errorbar(data_masked['TIME'][mask_masked], data_masked['PDCSAP_FLUX'][mask_masked]/data_masked['TREND'][mask_masked], yerr= data_masked['PDCSAP_FLUX_ERR'][mask_masked]/data_masked['TREND'][mask_masked], 
                #            fmt='.', color=datacolor, markersize=datasize, linewidth=1)
                plot_data(ax[0], data_masked['TIME'], data_masked['PDCSAP_FLUX']/data_masked['TREND'], segments)
                ax[0].set(ylabel='NORMALISED FLUX', xlim=[np.min(data['TIME'][mask]), np.max(data['TIME'][mask])])
                ax[0].xaxis.set_tick_params(which='both', labelbottom=True)


                ax[1].yaxis.set_major_formatter('{x:9<5.0f}')
                #ax[1].errorbar(data_masked['TIME'][mask_masked], data_masked['PDCSAP_FLUX'][mask_masked], yerr= data_masked['PDCSAP_FLUX_ERR'][mask_masked], 
                #            fmt='.', color=datacolor, markersize=datasize, linewidth=1)
                plot_data(ax[1], data_masked['TIME'], data_masked['PDCSAP_FLUX'], segments)
                trend_segments = bruce.data.find_nights_from_data(data_masked['TIME'][mask_masked], dx_lim=0.2)
                for seg in trend_segments:
                    ax[1].plot(data_masked['TIME'][mask_masked][seg], data_masked['TREND'][mask_masked][seg], 'r--', zorder=10, lw=1, label='Savgol filter' if seg[0]==0 else None, alpha = 0.5)
                ax[1].set(ylabel='PDCSAP FLUX')
                ax[1].legend()

                ax[2].yaxis.set_major_formatter('{x:9<5.0f}')
                #ax[2].errorbar(data_masked['TIME'][mask_masked], data_masked['SAP_FLUX'][mask_masked], yerr= data_masked['PDCSAP_FLUX_ERR'][mask_masked], 
                #            fmt='.', color=datacolor, markersize=datasize, linewidth=1)
                plot_data(ax[2], data_masked['TIME'], data_masked['SAP_FLUX'], segments)

                ylim = ax[2].get_ylim()

                ax[2].errorbar(data['TIME'][mask], data['SAP_FLUX'][mask], yerr= data['PDCSAP_FLUX_ERR'][mask], 
                            fmt='.', color=datacolor, markersize=datasize, linewidth=1, alpha = masked_alpha, zorder=0)
                ax[2].set_ylim(ylim)
                ax[2].set(ylabel='SAP FLUX')


                ax[3].yaxis.set_major_formatter('{x:9<5.0f}')
                ax[3].errorbar(data_masked['TIME'][mask_masked], data_masked['SAP_BKG'][mask_masked], yerr= data_masked['SAP_BKG_ERR'][mask_masked], 
                            fmt='.', color=datacolor, markersize=datasize, linewidth=1)
                ax[3].errorbar(data['TIME'][mask], data['SAP_BKG'][mask], yerr= data['SAP_BKG_ERR'][mask], 
                            fmt='.', color=datacolor, markersize=datasize, linewidth=1,alpha = masked_alpha, zorder=0)
                ax[3].set(ylabel='SAP BKG')
                ax[3].xaxis.set_tick_params(which='both', labelbottom=True)

                ax[4].yaxis.set_major_formatter('{x:9<5.2f}')
                # ax[4].errorbar(data_masked['TIME'][mask_masked], data_masked['PSF_CENTR1'][mask_masked], yerr = data_masked['PSF_CENTR1_ERR'][mask_masked], label = 'PSF_CENTR1',fmt='b.', markersize=1,)
                # ax[4].errorbar(data_masked['TIME'][mask_masked], data_masked['MOM_CENTR1'][mask_masked], yerr = data_masked['MOM_CENTR1_ERR'][mask_masked], label = 'MOM_CENTR1',fmt='g.', markersize=1)
                ax[4].errorbar(data_masked['TIME'][mask_masked], data_masked['PSF_CENTR1'][mask_masked], yerr = None, label = 'PSF_CENTR1',fmt='b.', markersize=1,)
                ax[4].errorbar(data_masked['TIME'][mask_masked], data_masked['MOM_CENTR1'][mask_masked], yerr = None, label = 'MOM_CENTR1',fmt='g.', markersize=1)
                ylim = ax[4].get_ylim()
                # ax[4].errorbar(data['TIME'][mask], data['PSF_CENTR1'][mask], yerr = data['PSF_CENTR1_ERR'][mask], label = None,fmt='b.', markersize=1,alpha = masked_alpha, zorder=0)
                # ax[4].errorbar(data['TIME'][mask], data['MOM_CENTR1'][mask], yerr = data['MOM_CENTR1_ERR'][mask], label = None,fmt='g.', markersize=1,alpha = masked_alpha, zorder=0)
                ax[4].errorbar(data['TIME'][mask], data['PSF_CENTR1'][mask], yerr = None, label = None,fmt='b.', markersize=1,alpha = masked_alpha, zorder=0)
                ax[4].errorbar(data['TIME'][mask], data['MOM_CENTR1'][mask], yerr = None, label = None,fmt='g.', markersize=1,alpha = masked_alpha, zorder=0)
                ax[4].set(ylabel='X', ylim=ylim)
                ax[4].legend()

                ax[5].yaxis.set_major_formatter('{x:9<5.2f}')
                # ax[5].errorbar(data_masked['TIME'][mask_masked], data_masked['PSF_CENTR2'][mask_masked], yerr = data_masked['PSF_CENTR2_ERR'][mask_masked], label = 'PSF_CENTR2',fmt='b.', markersize=1,)
                # ax[5].errorbar(data_masked['TIME'][mask_masked], data_masked['MOM_CENTR2'][mask_masked], yerr = data_masked['MOM_CENTR2_ERR'][mask_masked], label = 'MOM_CENTR2',fmt='g.', markersize=1)
                ax[5].errorbar(data_masked['TIME'][mask_masked], data_masked['PSF_CENTR2'][mask_masked], yerr = None, label = 'PSF_CENTR2',fmt='b.', markersize=1,)
                ax[5].errorbar(data_masked['TIME'][mask_masked], data_masked['MOM_CENTR2'][mask_masked], yerr = None, label = 'MOM_CENTR2',fmt='g.', markersize=1)
                ylim = ax[5].get_ylim()
                # ax[5].errorbar(data['TIME'][mask], data['PSF_CENTR2'][mask], yerr = data['PSF_CENTR2_ERR'][mask], label = None,fmt='b.', markersize=1,alpha = masked_alpha, zorder=0)
                # ax[5].errorbar(data['TIME'][mask], data['MOM_CENTR2'][mask], yerr = data['MOM_CENTR2_ERR'][mask], label = None,fmt='g.', markersize=1,alpha = masked_alpha, zorder=0)
                ax[5].errorbar(data['TIME'][mask], data['PSF_CENTR2'][mask], yerr = None, label = None,fmt='b.', markersize=1,alpha = masked_alpha, zorder=0)
                ax[5].errorbar(data['TIME'][mask], data['MOM_CENTR2'][mask], yerr = None, label = None,fmt='g.', markersize=1,alpha = masked_alpha, zorder=0)
                ax[5].set(ylabel='Y', xlabel = 'BJD - 2457000 [S{:}]'.format(sector), ylim=ylim)
                ax[5].legend()

                f.align_ylabels(ax)
                return f, ax

            print('Plotting MF base', flush=True)
            f ,ax = get_MF_full_base()
            for i in range(len(MFpeaks)):
                for j in range(len(ax)) : 
                    ax[j].axvspan(data_masked['TIME'][MFpeaks[i]]-MFmeta['widths'][i]/2, data_masked['TIME'][MFpeaks[i]]+MFmeta['widths'][i]/2, alpha=mf_axvspan_alpha, color='red')
                ax[0].annotate('E{:}'.format(i+1) , (data_masked['TIME'][MFpeaks[i]], (data_masked['PDCSAP_FLUX']/data_masked['TREND'])[MFpeaks[i]]), (data_masked['TIME'][MFpeaks[i]] - 2., (data_masked['PDCSAP_FLUX']/data_masked['TREND'])[MFpeaks[i]]) , bbox = props, arrowprops=arrowprops)
            ax[0].axhline(MFmeta['threshold'], c='r', ls='--', alpha = 0.3, label='Threshold = {:,} mmag [{:,} PPM]'.format(int((1 - MFmeta['threshold'])*1e3), int((1 - 10**(-0.4*(1 - MFmeta['threshold'])))*1e6) ))
            ax[0].legend()
            plt.subplots_adjust(left=0.08, bottom=0.03, right=0.995, top=0.995, wspace=None, hspace=None)
            plt.savefig(png_base + 'MONOFIND.webp', dpi=thumnnail_dpi,transparent=True)
            plt.close()


            for i in range(len(MFpeaks)):
                print('Plotting MF event {:}'.format(i+1), flush=True)
                tmin = data_masked['TIME'][MFpeaks[i]]-2*MFmeta['widths'][i]
                tmax = data_masked['TIME'][MFpeaks[i]]+2*MFmeta['widths'][i]
                f ,ax = get_MF_full_base(tmin=tmin, tmax=tmax)
                for j in range(len(ax)) : 
                    ax[j].axvspan(data_masked['TIME'][MFpeaks[i]]-MFmeta['widths'][i]/2, data_masked['TIME'][MFpeaks[i]]+MFmeta['widths'][i]/2, alpha=mf_axvspan_alpha, color='red')

                t_ = np.linspace(tmin, tmax, 10000)
                model =  transit_model.eval(x = t_, params=MFmeta['fitted_params'][i].params)
                ax[0].plot(t_, model, 'r', zorder=0, lw=1)
                ax[1].plot(t_, model*np.interp(t_, data_masked['TIME'], data_masked['TREND']), 'r',  zorder=0, lw=1)

                plt.subplots_adjust(left=0.08, bottom=0.03, right=0.995, top=0.995, wspace=None, hspace=None)
                plt.savefig(png_base + 'MONOFIND_E{:}.webp'.format(i+1), dpi=thumnnail_dpi,transparent=True)
                plt.close()


                # Now make the tpf_plot
                if tpf is not None:
                    # First, get the cadence
                    cadence_Tm2 = int(np.interp(data_masked['TIME'][MFpeaks[i]] - 2*MFmeta['widths'][i], tpf.time.btjd, tpf.cadenceno))
                    cadence_Tm1 = int(np.interp(data_masked['TIME'][MFpeaks[i]] - MFmeta['widths'][i], tpf.time.btjd, tpf.cadenceno))
                    cadence_T0 = int(np.interp(data_masked['TIME'][MFpeaks[i]]-MFmeta['widths'][i]/2, tpf.time.btjd, tpf.cadenceno))
                    cadence_Tc = int(np.interp(data_masked['TIME'][MFpeaks[i]], tpf.time.btjd, tpf.cadenceno))
                    cadence_T4 = int(np.interp(data_masked['TIME'][MFpeaks[i]]+MFmeta['widths'][i]/2, tpf.time.btjd, tpf.cadenceno))
                    cadence_Tp1 = int(np.interp(data_masked['TIME'][MFpeaks[i]]+MFmeta['widths'][i], tpf.time.btjd, tpf.cadenceno))
                    cadence_Tp2 = int(np.interp(data_masked['TIME'][MFpeaks[i]]+2*MFmeta['widths'][i], tpf.time.btjd, tpf.cadenceno))

                    cadence_Tm2 = tpf.cadenceno[np.argmin(np.abs(cadence_Tm2-tpf.cadenceno))]
                    cadence_Tm1 = tpf.cadenceno[np.argmin(np.abs(cadence_Tm1-tpf.cadenceno))]
                    cadence_T0 = tpf.cadenceno[np.argmin(np.abs(cadence_T0-tpf.cadenceno))]
                    cadence_Tc = tpf.cadenceno[np.argmin(np.abs(cadence_Tc-tpf.cadenceno))]
                    cadence_T4 = tpf.cadenceno[np.argmin(np.abs(cadence_T4-tpf.cadenceno))]
                    cadence_Tp1 = tpf.cadenceno[np.argmin(np.abs(cadence_Tp1-tpf.cadenceno))]
                    cadence_Tp2 = tpf.cadenceno[np.argmin(np.abs(cadence_Tp2-tpf.cadenceno))]

                    fig, ax = make_mf_tpf_plot_1(cadence_T0, cadence_Tc, cadence_T4, tpf)
                    fig.savefig(png_base + 'MONOFIND_E{:}_TPF1.webp'.format(i+1), dpi=thumnnail_dpi,transparent=True)
                    plt.close(fig)

                    fig, ax = make_mf_tpf_plot_2(cadence_Tm1, cadence_T0, cadence_Tc, cadence_T4, cadence_Tp1, tpf)
                    fig.savefig(png_base + 'MONOFIND_E{:}_TPF2.webp'.format(i+1), dpi=thumnnail_dpi,transparent=True)
                    plt.close(fig)

                    fig, ax = make_mf_tpf_plot_3(cadence_Tm2, cadence_Tc, cadence_Tp2, tpf)
                    fig.savefig(png_base + 'MONOFIND_E{:}_TPF3.webp'.format(i+1), dpi=thumnnail_dpi,transparent=True)
                    plt.close(fig)     
        except Exception as e: 
            print('File : {:} has failed to MF render fit with error:\n{:}\n{:}'.format(args.filename, e, traceback.format_exc()))


    if args.bls:
        try:
            # Now plot the BLS thumnail
            print('Plotting BLS thumbail', flush=True)
            f, ax = plt.subplots(nrows=1, ncols=1, figsize = thumnnail_figsize)
            def plot_bls_sde_base(ax,idx = None):
                ax.semilogx(BLS['period'], BLS['sde'], c='k', lw=0.1)
                ax.set(xlabel= 'PERIOD [DAY]', ylabel='SDE', xlim=[np.min(BLS['period'].value), np.max(BLS['period'].value)], ylim = (0, None))

                # Now annotate detections
                if idx is None:
                    for i in range(len(BLS['peaks'])):
                        ax.annotate('P{:}\nP = {:.5f} d'.format(i+1, BLS['period'][BLS['peaks'][i]].value) , ( BLS['period'][BLS['peaks'][i]].value, BLS['sde'][BLS['peaks'][i]]), ( BLS['period'][BLS['peaks'][i]].value + 0.9, BLS['sde'][BLS['peaks'][i]]) , bbox = props, arrowprops=arrowprops)
                else : 
                    i = idx
                    ax.annotate('P{:}\nP = {:.5f} d'.format(i+1, BLS['period'][BLS['peaks'][i]].value) , ( BLS['period'][BLS['peaks'][i]].value, BLS['sde'][BLS['peaks'][i]]), ( BLS['period'][BLS['peaks'][i]].value + 0.9, BLS['sde'][BLS['peaks'][i]]) , bbox = props, arrowprops=arrowprops)

                ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
                ax.xaxis.set_minor_formatter(NullFormatter())
                ax.axhline(bls_sde, ls='--', c='r', label = 'SDE {:.1f}\nPeaks {:.0f}'.format(bls_sde, len(BLS['peaks'])))
                ax.legend()
                return ax 

            # Plot the base thumnail
            ax = plot_bls_sde_base(ax)
            plt.subplots_adjust(left=0.08, bottom=0.157, right=0.995, top=0.983, wspace=None, hspace=None)
            plt.savefig(png_base + 'BLS_THUMBNAIL.webp', dpi=thumnnail_dpi,transparent=True)
            plt.close()

            # Now plot the peak thumbnail
            for i in range(len(BLS['peaks'])):
                print('Plotting BLS thumbail {:}'.format(i+1), flush=True)
                f, ax = plt.subplots(nrows=1, ncols=2, figsize = thumnnail_figsize)
                ax[0] = plot_bls_sde_base(ax[0], idx = i)
                phase = bruce.data.phase_times(data_masked['TIME'] + BTJD, BLS['epoch'][BLS['peaks'][i]].value, BLS['period'][BLS['peaks'][i]].value, phase_offset=0.2)
                phase_width = BLS['duration'][BLS['peaks'][i]].value /  BLS['period'][BLS['peaks'][i]].value
                mask = (phase > -phase_width) & (phase < phase_width)
                #ax[1].errorbar(phase[mask], data_masked['PDCSAP_FLUX'][mask]/data_masked['TREND'][mask], yerr= data_masked['PDCSAP_FLUX_ERR'][mask]/data_masked['TREND'][mask], 
                #        fmt='.', color=datacolor, markersize=datasize, linewidth=1)
                plot_data(ax[1], phase, data_masked['PDCSAP_FLUX']/data_masked['TREND'], segments, line_plot=False)
                ax[1].set(xlabel= 'PHASE', ylabel='NORMALISED FLUX', xlim=[-1.5*phase_width, 1.5*phase_width])

                t_ = np.linspace(-phase_width,phase_width, 1000)
                phase_params =  get_transit_model_phased_params(BLS['fitted_params'][i].params)
                model = transit_model.eval(params=phase_params, x=t_)
                ax[1].plot(t_, model, 'r', zorder=0, lw=1)


                plt.subplots_adjust(left=0.08, bottom=0.157, right=0.995, top=0.983, wspace=0.22, hspace=None)

                plt.savefig(png_base + 'BLS{:}_THUMBNAIL.webp'.format(i+1), dpi=thumnnail_dpi,transparent=True)
                plt.close()


            # Now plot the BLS full
            def plot_bls_full_base(ax, idx = None):
                # plot the normalised
                ax['A'].yaxis.set_major_formatter('{x:9<5.3f}')
                #ax['A'].errorbar(data_masked['TIME'], data_masked['PDCSAP_FLUX']/data_masked['TREND'], yerr= data_masked['PDCSAP_FLUX_ERR']/data_masked['TREND'], 
                #            fmt='.', color=datacolor, markersize=datasize, linewidth=1)
                plot_data(ax['A'], data_masked['TIME'], data_masked['PDCSAP_FLUX']/data_masked['TREND'], segments, line_plot=True)
                ax['A'].set(ylabel='NORMALISED FLUX', xlim=[np.min(data['TIME']), np.max(data['TIME'])])
                ax['A'].xaxis.set_tick_params(which='both', labelbottom=True)


                ax['B'].yaxis.set_major_formatter('{x:9<5.0f}')
                #ax['B'].errorbar(data_masked['TIME'], data_masked['PDCSAP_FLUX'], yerr= data_masked['PDCSAP_FLUX_ERR'], 
                #            fmt='.', color=datacolor, markersize=datasize, linewidth=1)
                plot_data(ax['B'], data_masked['TIME'], data_masked['PDCSAP_FLUX'], segments, line_plot=True)
                ax['B'].plot(data_masked['TIME'], data_masked['TREND'], 'r--', zorder=10, lw=1, label='Savgol filter', alpha = 0.5)
                ax['B'].set(ylabel='PDCSAP FLUX')
                ax['B'].legend()

                ax['C'].yaxis.set_major_formatter('{x:9<5.0f}')
                #ax['C'].errorbar(data_masked['TIME'], data_masked['SAP_FLUX'], yerr= data_masked['PDCSAP_FLUX_ERR'], 
                #            fmt='.', color=datacolor, markersize=datasize, linewidth=1)
                plot_data(ax['C'], data_masked['TIME'], data_masked['SAP_FLUX'], segments, line_plot=True)
                ylim = ax['C'].get_ylim()
                ax['C'].errorbar(data['TIME'], data['SAP_FLUX'], yerr= data['PDCSAP_FLUX_ERR'], 
                            fmt='.', color=datacolor, markersize=datasize, linewidth=1, alpha = masked_alpha, zorder=0)
                ax['C'].set_ylim(ylim)
                ax['C'].set(ylabel='SAP FLUX')

                ax['D'].yaxis.set_major_formatter('{x:9<5.0f}')
                #ax['D'].errorbar(data_masked['TIME'], data_masked['PDCSAP_FLUX']-data_masked['SAP_FLUX'], yerr= data_masked['PDCSAP_FLUX_ERR'], 
                #            fmt='.', color=datacolor, markersize=datasize, linewidth=1)
                plot_data(ax['D'], data_masked['TIME'], data_masked['PDCSAP_FLUX']-data_masked['SAP_FLUX'], segments, line_plot=True)
                ax['D'].set(ylabel='CBV CORRECTION', xlabel = 'BJD - {:}'.format(BTJD))

                # Now pot the SDE
                ax['E'] = plot_bls_sde_base(ax['E'],idx = idx)

                ax['F'].plot(BLS['period'],BLS['power'], 'k', lw=0.1 )
                ax['F'].plot(BLS['period'],BLS['trend'], 'r', lw=0.6, zorder = 10, label = 'TREND' )
                ax['F'].set(xlabel = 'PERIOD [DAY]', ylabel = "POWER")
                ax['F'].legend()

                for axs in [ax['A'], ax['B'], ax['C'], ax['D']]:
                    axs.set_xlim(np.min(data['TIME']), np.max(data['TIME']))


                if idx is not None:
                    i = idx
                    phase = bruce.data.phase_times(data_masked['TIME'] + BTJD, BLS['fitted_params'][i].params['t_zero'].value, BLS['fitted_params'][i].params['period'].value, phase_offset=0.2)
                    phase_width = BLS['duration'][BLS['peaks'][i]].value /  BLS['period'][BLS['peaks'][i]].value
                    mask = (phase > -phase_width) & (phase < phase_width)
                    #ax['G'].errorbar(phase[mask], data_masked['PDCSAP_FLUX'][mask]/data_masked['TREND'][mask], yerr= data_masked['PDCSAP_FLUX_ERR'][mask]/data_masked['TREND'][mask], 
                    #        fmt='.', color=datacolor, markersize=datasize, linewidth=1)
                    plot_data(ax['G'], phase, data_masked['PDCSAP_FLUX']/data_masked['TREND'], segments, line_plot=False)
                    ax['G'].set(xlabel= 'PHASE', ylabel='NORMALISED FLUX', xlim=[-1.5*phase_width, 1.5*phase_width])
                    t_ = np.linspace(-phase_width,phase_width, 1000)
                    phase_params =  get_transit_model_phased_params(BLS['fitted_params'][i].params)
                    model = transit_model.eval(params=phase_params, x=t_)
                    ax['G'].plot(t_, model, 'r', zorder=0, lw=1)

                    mask = (phase > (0.5-phase_width)) & (phase < (0.5+phase_width))
                    #ax['H'].errorbar(phase[mask], data_masked['PDCSAP_FLUX'][mask]/data_masked['TREND'][mask], yerr= data_masked['PDCSAP_FLUX_ERR'][mask]/data_masked['TREND'][mask], 
                    #        fmt='.', color=datacolor, markersize=datasize, linewidth=1)
                    plot_data(ax['H'], phase, data_masked['PDCSAP_FLUX']/data_masked['TREND'], segments, line_plot=False)
                    t_ = t_ + 0.5
                    model = transit_model.eval(params=phase_params, x=t_)
                    ax['H'].plot(t_, model, 'r', zorder=0, lw=1)
                    ax['H'].set(xlabel= 'PHASE',  xlim=[-1.5*phase_width+0.5, 1.5*phase_width+0.5])

                    phase = bruce.data.phase_times(data_masked['TIME'] + BTJD, BLS['epoch'][BLS['peaks'][i]].value, BLS['period'][BLS['peaks'][i]].value, phase_offset=0.2)
                    mask = (phase > -phase_width) & (phase < phase_width)
                    #ax['I'].errorbar(phase[mask], data_masked['PDCSAP_FLUX'][mask]/data_masked['TREND'][mask], yerr= data_masked['PDCSAP_FLUX_ERR'][mask]/data_masked['TREND'][mask], 
                    #        fmt='.', color=datacolor, markersize=datasize, linewidth=1)
                    plot_data(ax['I'], phase, data_masked['PDCSAP_FLUX']/data_masked['TREND'], segments, line_plot=False)
                    ax['I'].set(xlabel= 'PHASE', ylabel='NORMALISED FLUX', xlim=[-1.5*phase_width, 1.5*phase_width])
                    t_ = t_ - 0.5
                    model = 1 + box_model(t_, -BLS['depth'][BLS['peaks'][i]],0., BLS['duration'][BLS['peaks'][i]].value/BLS['period'][BLS['peaks'][i]].value,off=0.)
                    ax['I'].plot(t_, model, 'r', zorder=0, lw=1)

                    mask = (phase > (0.5-phase_width)) & (phase < (0.5+phase_width))
                    t_ = t_ + 0.5
                    #ax['J'].errorbar(phase[mask], data_masked['PDCSAP_FLUX'][mask]/data_masked['TREND'][mask], yerr= data_masked['PDCSAP_FLUX_ERR'][mask]/data_masked['TREND'][mask], 
                    #        fmt='.', color=datacolor, markersize=datasize, linewidth=1)
                    plot_data(ax['J'], phase, data_masked['PDCSAP_FLUX']/data_masked['TREND'], segments, line_plot=False)
                    model = 1 + box_model(t_, - BLS['depth'][BLS['peaks'][i]],0., BLS['duration'][BLS['peaks'][i]].value/BLS['period'][BLS['peaks'][i]].value,off=0.)
                    ax['J'].plot(t_, model, 'r', zorder=0, lw=1)
                    ax['J'].set(xlabel= 'PHASE', xlim=[-1.5*phase_width+0.5, 1.5*phase_width+0.5])


                    for j in range(-99,99):
                        epoch = BLS['epoch'][BLS['peaks'][i]].value + j*BLS['period'][BLS['peaks'][i]].value - BTJD
                        if (epoch < np.min(data['TIME'])) or (epoch > np.max(data['TIME'])) : continue
                        for axs in [ax['A'],ax['B'],ax['C'],ax['D']]:

                            axs.axvspan(epoch - BLS['duration'][BLS['peaks'][i]].value/2, epoch + BLS['duration'][BLS['peaks'][i]].value/2, alpha=mf_axvspan_alpha, color='red')

            print('Plotting BLS base', flush=True)
            # plot the full BLS base
            f = plt.figure(0, clear=True, figsize = BLS_figsize)
            layout = """AAEE
                        BBFF
                        CC..
                        DD.."""
            ax = f.subplot_mosaic(layout)
            plot_bls_full_base(ax)  
            plt.subplots_adjust(left=0.048, bottom=0.036, right=0.995, top=0.995, wspace=0.22, hspace=None)
            plt.savefig(png_base + 'BLS.webp', dpi=thumnnail_dpi,transparent=True)
            plt.close()


            # Now plot the peaks
            for i in range(len(BLS['peaks'])):
                print('Plotting BLS peak {:}'.format(i+1), flush=True)
                f = plt.figure(0, clear=True, figsize = BLS_figsize)
                layout = """AAEE
                            BBFF
                            CCGH
                            DDIJ"""
                ax = f.subplot_mosaic(layout)
                plot_bls_full_base(ax, idx=i)  


                plt.subplots_adjust(left=0.048, bottom=0.036, right=0.995, top=0.995, wspace=0.22, hspace=None)
                plt.savefig(png_base + 'BLS_P{:}.webp'.format(i+1), dpi=thumnnail_dpi,transparent=True)
                plt.close()

        except Exception as e: 
            print('File : {:} has failed to BLS render fit with error:\n{:}\n{:}'.format(args.filename, e, traceback.format_exc()))


    # if args.ls:
    #     try:
    #         # Now make the LS thumbnail
    #         print('Plotting LS thumbail', flush=True)
    #         f, ax = plt.subplots(nrows=1, ncols=1, figsize = thumnnail_figsize)
    #         def plot_ls_sde_base(ax,idx = None):
    #             ax.semilogx(BLS['period'], LS['sde'], c='k', lw=0.1)
    #             ax.set(xlabel= 'PERIOD [DAY]', ylabel='SDE', xlim=[np.min(BLS['period'].value), np.max(BLS['period'].value)], ylim = (0, None))

    #             # Now annotate detections
    #             if idx is None:
    #                 for i in range(len(LS['peaks'])):
    #                     ax.annotate('P{:}\nP = {:.5f} d'.format(i+1, BLS['period'][LS['peaks'][i]].value) , ( BLS['period'][LS['peaks'][i]].value, LS['sde'][LS['peaks'][i]]), ( BLS['period'][LS['peaks'][i]].value + 0.9, LS['sde'][LS['peaks'][i]]) , bbox = props, arrowprops=arrowprops)
    #             else : 
    #                 i = idx
    #                 ax.annotate('P{:}\nP = {:.5f} d'.format(i+1, BLS['period'][LS['peaks'][i]].value) , ( BLS['period'][LS['peaks'][i]].value, LS['sde'][LS['peaks'][i]]), ( BLS['period'][LS['peaks'][i]].value + 0.9, LS['sde'][LS['peaks'][i]]) , bbox = props, arrowprops=arrowprops)

    #             ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    #             ax.xaxis.set_minor_formatter(NullFormatter())
    #             ax.axhline(ls_sde, ls='--', c='r', label = 'SDE {:.1f}\nPeaks {:.0f}'.format(bls_sde, len(BLS['peaks'])))
    #             ax.legend()
    #             return ax 

    #         ax = plot_ls_sde_base(ax, idx=None)

    #         plt.subplots_adjust(left=0.08, bottom=0.157, right=0.995, top=0.983, wspace=None, hspace=None)
    #         plt.savefig(png_base + 'LS.webp', dpi=thumnnail_dpi)
    #         plt.close()


    #         for i in range(len(LS['peaks'])):
    #             print('Plotting LS thumbail {:}'.format(i+1), flush=True)
    #             f, ax = plt.subplots(nrows=1, ncols=2, figsize = thumnnail_figsize)
    #             ax[0] = plot_ls_sde_base(ax[0], idx = i)

    #             phase = bruce2.data.phase_times(data_masked['TIME'] + 2457000, 0., BLS['period'][LS['peaks']][i].value, 0.2) 
    #             #ax[1].errorbar(phase, data_masked['PDCSAP_FLUX'] - np.median(data_masked['PDCSAP_FLUX']), yerr= data_masked['PDCSAP_FLUX_ERR'], 
    #             #        fmt='.', color=datacolor, markersize=datasize, linewidth=1)
    #             plot_data(ax[1], phase, data_masked['PDCSAP_FLUX'] - np.median(data_masked['PDCSAP_FLUX']), segments, line_plot=False)

    #             t_ = np.linspace(-0.2,0.8, 1000)

    #             ax[1].set(xlabel= 'PHASE', ylabel='PDCSAP FLUX', xlim=[-0.2, 0.8])

    #             model = LS['fitted_model'].eval(x = t_*2*np.pi, params = LS['fitted_params'][i].params)
    #             ax[1].plot(t_, model, 'r', zorder=10, lw=1)

    #             plt.subplots_adjust(left=0.08, bottom=0.157, right=0.995, top=0.983, wspace=0.22, hspace=None)

    #             plt.savefig(png_base + 'LS_P{:}.webp'.format(i+1), dpi=thumnnail_dpi)
    #             plt.close()
    #     except Exception as e: 
    #         print('File : {:} has failed with error:\n{:}\n{:}'.format(args.filename, e, traceback.format_exc()))



    if args.searched:
        print('Saving search file')
        searched_table = Table(dtype=[('tic_id',int), ('run', int), ('sector', int),('ccd', int),('camera', int),
                                    ('window_length', int),('estimated_noise_ppm', int),
                                        ('measured_noise_ppm', int), ('mf_events', int),
                                        ('bls_peaks', int), ('ls_peaks', int)])
        searched_table.add_row([TICID, args.RUN, sector,ccd,camera,
                    window_length, int(estimated_noise_ppm), int(measured_noise_ppm),
                    len(MF_TABLE) if args.mf else 0,
                    len(BLS_TABLE) if args.bls else 0,
                    len(LS_TABLE) if args.ls else 0])
        searched_table.write(png_base + 'SEARCHED.fits', overwrite=True)

    if args.delete_files:
        print('Deleting input files')
        os.system('rm -vf {:}'.format(args.filename))
        os.system('rm -vf {:}'.format(tpf_filename))

    plt.close()
    gc.collect()
    
if __name__ == '__main__':
    main()
