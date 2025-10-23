# Imports 
from unittest import result
import numpy as np
import bruce
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import multiprocess

from bruce.ambiguous_period.fitting import chi_fixed_period
from bruce.data.data_processing import flatten_data_with_function

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
    def __init__(self, time, flux, flux_err, w=None, sky_bkg=None):
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
            ax[i].errorbar(self.time[segments[i]], self.flux[segments[i]], yerr=self.flux_err[segments[i]], fmt='k.', markersize=1, alpha=data_alpha)
            ax[i].set(xlabel='Time [day]', ylabel='Flux')
            if self.w is not None : 
                ax[i].plot(self.time[segments[i]], self.w[segments[i]], c='orange', zorder=10, lw=1)
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
        
    def write_data(self,filename):
        if hasattr(self, 'w'):
            np.savetxt(filename, np.array([self.time, self.flux, self.flux_err, self.w]).T)
        else : 
            np.savetxt(filename, np.array([self.time, self.flux, self.flux_err]).T)



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
    
    def fit_event_with_fixed_period(self,nthreads=12, fit_period=30., plot=True):
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
        weights = np.exp(-(self.de_event_with_fixed_period.population_energies - np.min(self.de_event_with_fixed_period.population_energies)))
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

            return fig_initial, ax_initial, fig_final, ax_final, return_data


