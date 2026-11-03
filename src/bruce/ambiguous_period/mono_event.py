# Imports 
from unittest import result
import numpy as np
import bruce
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import multiprocess


from matplotlib.widgets import Button, Slider
from scipy.interpolate import UnivariateSpline
from lmfit import create_params, report_fit, minimize, Minimizer
import lmfit

from bruce.ambiguous_period.fitting import chi_fixed_period
from bruce.data.data_processing import flatten_data_with_function
from bruce.data import phase_times
from bruce.binarystar import transit_width

def group_data_by_epochs(data_list, data_list_labels, epoch_1, epoch_2):
    # First, lets order data by start point
    idx_sort = np.argsort([i.time.min() for i in data_list])
    data_list_sorted = data_list[idx_sort]
    data_list_labels_sorted = data_list_labels[idx_sort]

    # Now lets work out which data includes the transits
    # For this, we can assume that everything with start time above epoch time_2 is to br grouped
    max_epoch = max(epoch_1, epoch_2)

    data_to_be_grouped, data_to_be_grouped_labels = [],[] 
    other_data, other_data_labels = [],[]
    for i in range(len(data_list_sorted)):
        if data_list_sorted[i].time.min()<max_epoch:
            data_to_be_grouped.append(data_list_sorted[i])
            data_to_be_grouped_labels.append(data_list_labels_sorted[i])
        else:
            other_data.append(data_list_sorted[i])
            other_data_labels.append(data_list_labels_sorted[i])

    # Now concat
    time, flux, flux_err, w = [],[],[], []
    for i in data_to_be_grouped : 
        time.append(i.time)
        flux.append(i.flux)
        flux_err.append(i.flux_err)
        w.append(i.w)
    return_data = [photometry_time_series(np.concatenate(time), np.concatenate(flux),np.concatenate(flux_err), w=np.concatenate(w)), *other_data]
    return_labels = [','.join(data_to_be_grouped_labels), *other_data_labels]

    return return_data, return_labels
    









class photometry_time_series:
    def __init__(self, time=None, flux=None, flux_err=None, w=None, sky_bkg=None, filename=None):
        if time is None:
            self.load_data(filename)
            return

        self.time, self.flux, self.flux_err = time, flux, flux_err
        self.w = w
        self.sky_bkg = sky_bkg
        argsort = np.argsort(self.time)
        self.time, self.flux, self.flux_err= self.time[argsort], self.flux[argsort], self.flux_err[argsort]
        if w is not None : self.w = self.w[argsort]
                
    def plot_segments(self,dx_lim=10, data_alpha=0.5):
        try : segments = bruce.data.find_nights_from_data(self.time, dx_lim=dx_lim)
        except : segments = [np.arange(len(self.time), dtype=int)]
        fig, ax = plt.subplots(len(segments), 1, figsize = (6,4*len(segments)/3))
        ax = np.atleast_1d(ax)
        for i in range(len(segments)):
            ax[i].errorbar(self.time[segments[i]]-2457000, self.flux[segments[i]], yerr=self.flux_err[segments[i]], fmt='k.', markersize=1, alpha=data_alpha)
            ax[i].set(xlabel='Time [BTJD]', ylabel='Flux')
            if self.w is not None : 
                ax[i].plot(self.time[segments[i]]-2457000, self.w[segments[i]], c='orange', zorder=10, lw=1)
        plt.tight_layout()
        return fig, ax

        
    def flatten_data(self, median_bin_size=1, convolve_bin_size=1):
        self.w = bruce.data.median_filter(self.time, self.flux, bin_size=median_bin_size)
        self.w = bruce.data.convolve_1d(self.time, self.w, bin_size=convolve_bin_size)

    def flatten_data_old(self,window_width=0.2, sigmaclip=3, dx_lim=0.2):
        segments = bruce.data.find_nights_from_data(self.time, dx_lim=dx_lim)
        self.w = np.ones(self.flux.shape[0])
        for seg in segments : 
            # try : 

            try : 
                cadence = np.median(np.gradient(self.time[seg]))
                window_length_ = int(window_width/cadence)
                if window_length_ > self.time[seg].shape[0]:
                    window_length_ = int(0.8*self.time[seg].shape[0])
                print('W : ', window_length_)
                if window_length_ % 2 == 0: window_length_ +=1
                self.w[seg] = bruce.data.flatten_data_with_function(self.time[seg] + 2457000, self.flux[seg], 
                                                                        SG_iter=10, SG_window_length=window_length_, SG_sigma=sigmaclip,
                                                                        SG_polyorder=3, SG_deriv=0, SG_delta=1.)
            except : 
                self.w[seg] = np.polyval(np.polyfit(self.time[seg], self.flux[seg],1), self.time[seg])
        
    def fit_transit_event(self, 
                          t_zero, period,radius_1,k,b,
                          fitted_params = ['t_zero', 'radius_1', 'k', 'b', 'zp'],
                          fit_zp=True, fit_dilution=True, partial=False,
                          report=True, plot=False):

        params_dict = {
            't_zero': {
                'value': t_zero,
                'min': t_zero - 0.5,
                'max': t_zero + 0.5,
                'vary': True if 't_zero' in fitted_params else False
            },
            'period': {
                'value': period,
                'min': period - 0.1,
                'max': period + 0.1,
                'vary': True if 'period' in fitted_params else False
            },
            'radius_1' : {
                'value': radius_1,
                'min': 0.001,
                'max': 0.5,
                'vary': True if 'radius_1' in fitted_params else False
            },
            'k' : {
                'value': k,
                'min': 0.001,
                'max': 0.5,
                'vary': True if 'k' in fitted_params else False
            },
            'b': {
                'value': b,
                'min': 0.001,
                'max': 1.5,
                'vary': True if 'b' in fitted_params else False
            },
            'zp' : {
                'value': 1,
                'min': 0.99,
                'max': 1.01,
                'vary': True if 'zp' in fitted_params else False
            },
            'light_3' : {
                'value': 0,
                'min': 0,
                'max': 3,
                'vary': True if 'light_3' in fitted_params else False
            }
        }
        params = create_params(**params_dict)
        if partial:
            params['radius_1'].vary=False
            params['k'].vary=False
            params['b'].vary=False

        # params.add('__lnsigma', value=np.log(0.1), min=np.log(0.001), max=np.log(2))
        def residual(params, time, flux, flux_err, w, return_model=False):
            incl = np.arccos(params['radius_1']*params['b'])
            model = params['zp']*bruce.binarystar.lc(time, 
                                        t_zero=params['t_zero'],
                                        period = params['period'],
                                        radius_1=params['radius_1'],
                                        k=params['k'],
                                        incl=incl,
                                        light_3=params['light_3'])
            if return_model : return model
            else :     return ((flux/w) - model) / (flux_err/w)
        

        mini = Minimizer(residual, params, 
                         fcn_args=(self.time, self.flux, self.flux_err, self.w, False),
                         nan_policy='propagate')
        
        # first solve with Nelder-Mead algorithm
        out1 = mini.minimize(method='Nelder')

        # then solve with Levenberg-Marquardt using the
        # Nelder-Mead solution as a starting point
        out2 = mini.minimize(method='leastsq', params=out1.params)

        # sanitize errors 
        for p in out2.params:
            if out2.params[p].stderr is None : out2.params[p].stderr = 0.1*(out2.params[p].max - out2.params[p].max)

        # Reestimate the errors
        ci, trace = lmfit.conf_interval(mini, out2, sigmas=[1], trace=True)
        

        if report : 
            report_fit(out2)
            lmfit.printfuncs.report_ci(ci)

        if plot:
            if 'period' not in fitted_params:
                t_ref = out2.params['t_zero'].value
                cycle = int(np.round((np.median(self.time)-t_zero)/period))


                figi, axi = plt.subplots(1,1)
                axi.errorbar(self.time-t_ref, self.flux/self.w, self.flux_err/self.w, fmt='k.', markersize=1)
                t_ = np.linspace(self.time.min(), self.time.max(),1000)
                model = residual(params, t_, self.flux, self.flux_err, self.w, return_model=True)
                axi.plot(t_-t_ref, model, 'orange', lw=1)
                axi.set_xlabel('Time - {:}'.format(t_ref))
                axi.set_ylabel('Flux')
                #axi.set_title('Cycle {:} [initial]'.format(cycle))

                fig, ax = plt.subplots(1,1)
                ax.errorbar(self.time-t_ref, self.flux/self.w, self.flux_err/self.w, fmt='k.', markersize=1)
                t_ = np.linspace(self.time.min(), self.time.max(),1000)
                model = residual(out2.params, t_, self.flux, self.flux_err, self.w, return_model=True)
                ax.plot(t_-t_ref, model, 'orange', lw=1)
                ax.set_xlabel('Time - {:}'.format(t_ref))
                ax.set_ylabel('Flux')
                #ax.set_title('Cycle {:}'.format(cycle))

            return out2, ci, trace, fig, ax, figi, axi
        return out2, ci, trace



        






    def write_data(self,filename):
        if hasattr(self, 'w'):
            np.savetxt(filename, np.array([self.time, self.flux, self.flux_err, self.w]).T)
        else : 
            np.savetxt(filename, np.array([self.time, self.flux, self.flux_err]).T)

    def load_data(self, filename):
        d = np.loadtxt(filename).T
        if d.shape[0]==3 : self.time , self.flux, self.flux_err         = d[0], d[1], d[2]
        else :             self.time , self.flux, self.flux_err, self.w = d[0], d[1], d[2], d[3]



    def normalise_gui(self):

        import numpy as np
        import matplotlib.pyplot as plt

        from matplotlib.widgets import Button, Slider
        from scipy.interpolate import UnivariateSpline

        continuum_x = []
        continuum_y = []

        self.status = "skipped"
        self.w = None
        self.flux_norm = None

        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(10, 8),
            gridspec_kw={"height_ratios": [2, 1]}
        )

        plt.subplots_adjust(bottom=0.18)

        ax1.set_title(
            "Click continuum points. "
            "Fit spline. Done = accept. Skip = reject."
        )

        # -----------------------------
        # Top panel
        # -----------------------------

        ax1.plot(
            self.time,
            self.flux,
            '.k',
            ms=3
        )

        point_plot, = ax1.plot(
            [],
            [],
            'ro',
            ms=8
        )

        spline_line, = ax1.plot(
            [],
            [],
            'b-',
            lw=2
        )

        # -----------------------------
        # Bottom panel
        # -----------------------------

        norm_line, = ax2.plot(
            self.time,
            self.flux,
            '.k',
            ms=3
        )

        ax2.set_xlabel("Time")
        ax2.set_ylabel("Normalised Flux")

        # -----------------------------
        # Smoothness slider
        # -----------------------------

        slider_ax = plt.axes(
            [0.15, 0.08, 0.35, 0.03]
        )

        smooth_slider = Slider(
            slider_ax,
            "Smooth",
            0,
            1,
            valinit=0.01
        )

        # -----------------------------
        # Fit callback
        # -----------------------------

        def fit_spline(event=None):

            if len(continuum_x) < 4:
                print("Need at least 4 continuum points")
                return

            x = np.array(continuum_x)
            y = np.array(continuum_y)

            idx = np.argsort(x)

            x = x[idx]
            y = y[idx]

            spline = UnivariateSpline(
                x,
                y,
                s=smooth_slider.val * len(x)
            )

            model = spline(self.time)

            self.w = model
            self.flux_norm = self.flux / model

            spline_line.set_data(
                self.time,
                model
            )

            norm_line.set_ydata(
                self.flux_norm
            )

            ax1.relim()
            ax1.autoscale_view()

            ax2.relim()
            ax2.autoscale_view()

            fig.canvas.draw_idle()

        # -----------------------------
        # Mouse click callback
        # -----------------------------

        def onclick(event):

            if event.inaxes != ax1:
                return

            if event.xdata is None:
                return

            continuum_x.append(event.xdata)
            continuum_y.append(event.ydata)

            point_plot.set_data(
                continuum_x,
                continuum_y
            )

            fig.canvas.draw_idle()

        # -----------------------------
        # Clear callback
        # -----------------------------

        def clear(event):

            continuum_x.clear()
            continuum_y.clear()

            point_plot.set_data([], [])
            spline_line.set_data([], [])

            self.w = None
            self.flux_norm = None

            norm_line.set_ydata(
                self.flux
            )

            fig.canvas.draw_idle()

        # -----------------------------
        # Done callback
        # -----------------------------

        def done_callback(event):

            if self.w is None:
                print("No spline fitted yet.")
                return

            self.status = "accepted"

            plt.close(fig)

        # -----------------------------
        # Skip callback
        # -----------------------------

        def skip_callback(event):

            self.status = "skipped"

            self.w = None
            self.flux_norm = None

            plt.close(fig)

        # -----------------------------
        # Window close callback
        # -----------------------------

        def close_callback(event):

            if self.status != "accepted":

                self.status = "skipped"

                self.w = None
                self.flux_norm = None

        # -----------------------------
        # Buttons
        # -----------------------------

        fit_ax = plt.axes(
            [0.55, 0.05, 0.10, 0.06]
        )

        fit_button = Button(
            fit_ax,
            "Fit"
        )

        fit_button.on_clicked(
            fit_spline
        )

        clear_ax = plt.axes(
            [0.67, 0.05, 0.10, 0.06]
        )

        clear_button = Button(
            clear_ax,
            "Clear"
        )

        clear_button.on_clicked(
            clear
        )

        skip_ax = plt.axes(
            [0.79, 0.05, 0.10, 0.06]
        )

        skip_button = Button(
            skip_ax,
            "Skip"
        )

        skip_button.on_clicked(
            skip_callback
        )

        done_ax = plt.axes(
            [0.91, 0.05, 0.08, 0.06]
        )

        done_button = Button(
            done_ax,
            "Done"
        )

        done_button.on_clicked(
            done_callback
        )

        # -----------------------------
        # Connections
        # -----------------------------

        fig.canvas.mpl_connect(
            "button_press_event",
            onclick
        )

        fig.canvas.mpl_connect(
            "close_event",
            close_callback
        )

        plt.show()

        return self.w


    def split_and_normalise_transit_events(self, t_zero, period, width, dx_lim=0.2):
        phase = phase_times(self.time, t_zero, period, phase_offset=0.2)
        phase_width = width/period 
        mask = np.abs(phase)<(phase_width/2)
        print(mask.sum(), mask.shape)
        time, flux, flux_err = self.time[mask], self.flux[mask], self.flux_err[mask]

        data = []
        segs = bruce.data.find_nights_from_data(time, dx_lim=dx_lim)
        if len(segs)==0 : return data
        for seg in segs:
            print(len(seg))
            if len(seg)<10 : continue
            d = photometry_time_series(time[seg],flux[seg], flux_err[seg])
            d.normalise_gui()
            if d.status != "skipped":
                data.append(d)
        return data




def get_theta(name,theta,theta_names) : return theta[np.argwhere(theta_names==name)[0][0]]
def get_de(name,de, de_names) : return de.x[np.argwhere(de_names==name)[0][0]]

def report_best_params(xk, convergence):
    print(f"Current best parameters: {xk}, Convergence: {convergence}")
    




class mono_event():
    def __init__(self,t_cen, width, depth, data, 
                 name='Single-transit event',
                median_bin_size = 1,convolve_bin_size = 1):
        self.t_cen = t_cen
        self.width = width
        self.depth = depth
        self.data = data
        self.name=name
        self.median_bin_size = median_bin_size
        self.convolve_bin_size = convolve_bin_size

    def de_transit_width(self,):
        return bruce.binarystar.transit_width(get_de('radius_1',self.de_event_with_fixed_period,self.de_event_with_fixed_period.theta_names), get_de('k',self.de_event_with_fixed_period,self.de_event_with_fixed_period.theta_names), get_de('b',self.de_event_with_fixed_period,self.de_event_with_fixed_period.theta_names), period=self.fit_period)

    def de_get_epoch(self,):
        return get_de('t_zero',self.de_event_with_fixed_period,self.de_event_with_fixed_period.theta_names)
    def de_get_radius_1(self,):
        return get_de('radius_1',self.de_event_with_fixed_period,self.de_event_with_fixed_period.theta_names)
    def de_get_k(self,):
        return get_de('k',self.de_event_with_fixed_period,self.de_event_with_fixed_period.theta_names)
    def de_get_b(self,):
        return get_de('b',self.de_event_with_fixed_period,self.de_event_with_fixed_period.theta_names)
    
    def de_evaluate_model(self, period, data):
        t_zero = get_de('t_zero',self.de_event_with_fixed_period,self.de_event_with_fixed_period.theta_names)
        radius_1 = get_de('radius_1',self.de_event_with_fixed_period,self.de_event_with_fixed_period.theta_names)
        k = get_de('k',self.de_event_with_fixed_period,self.de_event_with_fixed_period.theta_names)
        b = get_de('b',self.de_event_with_fixed_period,self.de_event_with_fixed_period.theta_names)
        incl = bruce.sampler.incl_from_radius_1_b(radius_1,b)
        return bruce.binarystar.lc(data.time, data.flux, data.flux_err, t_zero=t_zero, period=period,radius_1 = radius_1, k=k, incl=incl, offset=False, jitter=0.)
    
    
    
    def de_phase_model(self,):
        phase_width = self.de_transit_width()/self.fit_period
        phase = np.linspace(-phase_width,phase_width,1000)
        radius_1 = get_de('radius_1',self.de_event_with_fixed_period,self.de_event_with_fixed_period.theta_names)
        k = get_de('k',self.de_event_with_fixed_period,self.de_event_with_fixed_period.theta_names)
        b = get_de('b',self.de_event_with_fixed_period,self.de_event_with_fixed_period.theta_names)
        incl = bruce.sampler.incl_from_radius_1_b(radius_1,b)
        return phase, bruce.binarystar.lc(phase, radius_1 = radius_1, k=k, incl=incl)
    
    def fit_event_with_fixed_period(self,nthreads=12, fit_period=30., plot=True, method='de', emcee_steps=10000, emcee_burn_in=7000, R1 = [1,0.1]):
        # Now convert parameters
        self.fit_period = fit_period
        self.k = np.sqrt(self.depth) 
        self.radius_1 = np.pi*self.width/self.fit_period
        
        theta, theta_names, bounds = [], [], []
        theta_names.append('t_zero'); theta.append(self.t_cen); bounds.append([self.t_cen-0.1,self.t_cen+0.1])
        theta_names.append('radius_1'); theta.append(np.clip(self.radius_1, 0.001,0.3)); bounds.append([0.001,0.3])
        theta_names.append('k'); theta.append(np.clip(self.k, 0.001,0.3)); bounds.append([0.001,0.3])
        theta_names.append('b'); theta.append(0.1); bounds.append([0.001,1.5])
        theta, theta_names, bounds = np.array(theta),  np.array(theta_names),  np.array(bounds)
        
        # Do the initial
        chi_initial = chi_fixed_period(theta, self.data, theta_names, False, self.fit_period, self.median_bin_size, self.convolve_bin_size)
        chi_initial_red = chi_initial / (self.data.time.shape[0]-len(theta))
        print('Initial Chi-Sqaured : {:.2f} [red {:.2f}]'.format(chi_initial, chi_initial_red))
        
        if plot:
            fig_initial, ax_initial, return_data = chi_fixed_period(theta, self.data, theta_names, True, self.fit_period, self.median_bin_size, self.convolve_bin_size)
            #fig_initial.suptitle(self.name + ' - Initial')
            fig_initial.suptitle(self.name, y=0.85, x=0.55, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', alpha=1.0), ha='center')
        
        # Do the DE fit
        # with multiprocess.Pool(processes=nthreads) as pool:
        #     self.de_event_with_fixed_period = differential_evolution(chi_fixed_period, bounds=bounds,x0 = theta, 
        #                                 disp=False, workers = pool.map,
        #                                maxiter=1000, polish=False, tol=0.001,
        #                                popsize=30, args=(self.data, theta_names, False, self.fit_period, self.median_bin_size, self.convolve_bin_size))
        #     self.de_event_with_fixed_period.theta_names = theta_names.copy()

        self.de_event_with_fixed_period = differential_evolution(chi_fixed_period, bounds=bounds,x0 = theta, 
                                    disp=False,
                                    maxiter=1000, polish=False, tol=0.001,
                                    popsize=30, args=(self.data, theta_names, False, self.fit_period, self.median_bin_size, self.convolve_bin_size))
        self.de_event_with_fixed_period.theta_names = theta_names.copy()




        print('Fitted parameters for {:}:'.format(self.name))
        for i, j in zip(theta_names, self.de_event_with_fixed_period.x) : print('{:} : {:}'.format(i,j))
        chi_final = chi_fixed_period(self.de_event_with_fixed_period.x, self.data, theta_names, False, self.fit_period, self.median_bin_size, self.convolve_bin_size)
        chi_final_red = chi_final / (self.data.time.shape[0]-len(theta))
        print('Final Chi-Sqaured : {:.2f} [red {:.2f}]'.format(chi_final, chi_final_red))


        # Now get errors
        # weights = np.exp(-(self.de_event_with_fixed_period.population_energies - np.min(self.de_event_with_fixed_period.population_energies)))

        res = self.de_event_with_fixed_period

        if hasattr(res, "population_energies"):
            energies = res.population_energies
        else:
            # Fallback for old SciPy
            # res.fun is the best solution value
            energies = np.array([res.fun])
        weights = np.exp(-(energies - np.min(energies)))
        weights /= np.sum(weights)


        mean = np.average(self.de_event_with_fixed_period.population, axis=0, weights=weights)
        cov  = np.cov(self.de_event_with_fixed_period.population.T, aweights=weights)
        self.x_uncertainties = np.sqrt(np.diag(cov))

        low = np.percentile(self.de_event_with_fixed_period.population, 16, axis=0)
        med = np.percentile(self.de_event_with_fixed_period.population, 50, axis=0)
        high = np.percentile(self.de_event_with_fixed_period.population, 84, axis=0)
        self.x_uncertainties_percentile = (med-low, high-med)

        # Now lets do width and depth
        self.x_aux = np.zeros((self.de_event_with_fixed_period.population.shape[0], 2))
        for i in range(self.x_aux.shape[0]):
            self.x_aux[i,0] = bruce.binarystar.transit_width(*self.de_event_with_fixed_period.population[i][1:],  period=30)
            
            self.x_aux[i,1] = 1 - bruce.binarystar.lc(t=np.array([0]), t_zero=0, period=30,
                                                            radius_1=self.de_event_with_fixed_period.population[i][1],
                                                            k = self.de_event_with_fixed_period.population[i][2],
                                                            incl=np.arccos(self.de_event_with_fixed_period.population[i][1]*self.de_event_with_fixed_period.population[i][3]))[0]


        low = np.percentile(self.x_aux, 16, axis=0)
        med = np.percentile(self.x_aux, 50, axis=0)
        high = np.percentile(self.x_aux, 84, axis=0)
        self.x_aux_percentile = (med-low, high-med)

        if plot:
            fig_final, ax_final, return_data = chi_fixed_period(self.de_event_with_fixed_period.x, self.data, theta_names, True, self.fit_period, self.median_bin_size, self.convolve_bin_size)
            #fig_final.suptitle(self.name + ' - Fitted')
            fig_final.suptitle(self.name, y=0.85,x=0.55, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', alpha=1.0), ha='center')


        if method=='de+emcee':

            def lnike(theta, R1 = [1,0.1]):

                L = np.zeros(4, dtype=np.float64)
                L[0] = -np.inf
                # t_zero ,radius_1, k, b
                # blobs = width, R2, depth
                if theta[1]<0 : return L
                if theta[1]>0.3 : return L
                if theta[2]<0 : return L
                if theta[2]>0.3 : return L
                if theta[3]<0 : return L
                if theta[3]>(1+theta[2]) : L
                L[0] = -0.5*chi_fixed_period(theta, self.data, theta_names, False, self.fit_period, self.median_bin_size, self.convolve_bin_size)
                L[1] = transit_width(radius_1=theta[1], k=theta[2], b=theta[3], period=30)
                L[2] = np.random.normal(*R1)*theta[2] # R2
                L[3] = 1 - bruce.binarystar.lc(t = np.array([0], dtype=np.float64), period=30, radius_1=theta[1], k = theta[2], incl = np.arccos(theta[1]*theta[3]))[0]
                return L

            ndim = len(theta)
            nwalkers = 4*ndim 
            p0 = self.de_event_with_fixed_period.x + 1e-5*np.random.randn(nwalkers, ndim)
            import emcee
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnike, args = (R1,))
            sampler.run_mcmc(p0, emcee_steps, progress=True)




            labels = ['t_zero', 'R1/a', 'k', 'b']
            flat_samples = sampler.get_chain(discard=emcee_burn_in, thin=15, flat=True)
            self.theta_emcee = []
            for i in range(ndim):
                mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
                self.theta_emcee.append(mcmc)
                q = np.diff(mcmc)
                txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
                txt = txt.format(mcmc[1], q[0], q[1], labels[i])
                # print(txt)
            self.theta_emcee  = np.array(self.theta_emcee )

            fig_emcee, ax_emcee, return_data_emcee = chi_fixed_period(self.theta_emcee[:,1], self.data, theta_names, True, self.fit_period, self.median_bin_size, self.convolve_bin_size)

            return fig_initial, ax_initial, fig_final, ax_final, fig_emcee, ax_emcee, return_data_emcee, sampler


        else:
            return fig_initial, ax_initial, fig_final, ax_final, return_data

