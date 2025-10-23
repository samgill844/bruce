# Imports 
import numpy as np
import bruce, os
import matplotlib.pyplot as plt, gc
from itertools import combinations
from scipy.constants import G
import scipy.stats
from astropy.coordinates import SkyCoord, get_body
from astropy.time import Time, TimeDelta
from astropy.table import Table, Column
from astropy import units as u
from astroplan.utils import time_grid_from_range
from tqdm import tqdm 

from astroplan import Observer
from astropy.coordinates import  SkyCoord, get_body, get_sun
from astroquery.mast import Catalogs
from astropy.coordinates import SkyCoord, EarthLocation, AltAz

from bruce.data import bin_data
from bruce.ambiguous_period.mono_event import photometry_time_series


class ambiguous_period:
    def __init__(self, data, events=[], 
                 name='Period resolver',median_bin_size = 1,convolve_bin_size = 1):
        self.data = data
        self.events = events
        self.name=name
        self.median_bin_size = median_bin_size
        self.convolve_bin_size = convolve_bin_size

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
        time, flux, flux_err = [],[],[]
        for i in data_to_be_grouped : 
            time.append(i.time)
            flux.append(i.flux)
            flux_err.append(i.flux_err)
        return_data = [photometry_time_series(np.concatenate(time), np.concatenate(flux),np.concatenate(flux_err)), *other_data]
        return_labels = [','.join(other_data_labels), *other_data_labels]

        return return_data, return_labels


        

    def mask_and_filter_events(self,):
        self.event_mask = np.zeros(self.data.time.shape[0], dtype=bool)
        for i in range(len(self.events)):
            self.event_mask = self.event_mask | ((self.data.time>(self.events[i].de_get_epoch() - self.events[i].de_transit_width()/2)) &  (self.data.time<(self.events[i].de_get_epoch() + self.events[i].de_transit_width()/2)))
        self.event_mask = ~self.event_mask

#         w = bruce.data.median_filter(self.data.time[self.event_mask], self.flux[self.event_mask], bin_size=self.median_bin_size)
#         w = bruce.data.median_filter(self.time[self.event_mask], w, bin_size=self.median_bin_size)
#         self.w = np.interp(time, time[self.event_mask], w)
# #         plt.scatter(self.time, self.flux/self.w)
# #         plt.show()
        
    
    def are_multiples_within_tolerance(self, numbers, tolerance=0.01):
        if len(numbers) < 3:
            return [True]*len(numbers)

        smallest = min(numbers)

        vals = [abs(round(num / smallest) - (num / smallest)) for num in numbers]
        test = [num<tolerance for num in vals]
        
        return vals,test

        
    def calcualte_aliases(self,min_period=None, multi_epoch_check_tolerance=1e-3, dx_lim=1):
        # Sort the minmum period out
        if min_period is None : 
            start_end=[]
            for seg in bruce.data.find_nights_from_data(self.data.time, dx_lim=dx_lim): start_end.append([self.data.time[seg].max() , self.data.time[seg].min()])
            spans = [(i[0] - i[1]) for i in start_end]
            self.min_period = np.max(spans)
        else : self.min_period = min_period
            
        # Sort the aliases out
        self.unique_pairs = list(combinations(self.events, 2))
        self.unique_pairs = np.array([abs(i[0].de_get_epoch() - i[1].de_get_epoch()) for i in self.unique_pairs])
        self.max_period = np.min(self.unique_pairs)
           
        # Now work out the aliases
        self.P_max = int(np.floor(self.max_period/self.min_period))
        self.P_min = 1
        print(self.P_min,max(2,self.P_max+1),1, self.max_period/self.min_period, self.max_period,self.min_period, self.max_period/self.P_max)
        self.aliases = np.arange(self.P_min,max(self.P_max+1,2),1, dtype=int)[:]
        
        # Now check if there are more than 2 events if the sysrem is solved
        solutions=[]
        for i in range(self.aliases.shape[0]):
            period = self.max_period/self.aliases[i]
            if (np.abs(bruce.data.phase_times(self.unique_pairs/period,0,1,phase_offset=0.5))<multi_epoch_check_tolerance).sum()==len(self.events):
                solutions.append(i)
        if len(solutions)==1:
            print('System appears solved at alias P{:} with period {:}'.format(solutions[0], self.max_period/self.aliases[solutions[0]]))
            
            # Lets update
            self.aliases = np.array([self.aliases[solutions[0]]])
            
        elif len(solutions)>1:
            print('System appears solved with multiple periods:')
            for j in range(len(solutions)):
                print('- alias P{:} with period {:}'.format(solutions[j], self.max_period/self.aliases[solutions[j]]))

        return len(solutions)
    
    def calcualte_data_delta_L(self, data, p_value=0.0001, df=3,):
        _,self.height = bruce.template_match.get_delta_loglike_height_from_fap(p_value=p_value, df=df)

        epoch = self.events[0].de_get_epoch()
        width = self.events[0].de_transit_width()
        delta_L = np.zeros(self.aliases.shape[0])
        phase_model, phase_model_flux = self.events[0].de_phase_model()
        event_mask = np.zeros(data.time.shape[0], dtype=bool)
        for i in range(len(self.events)):
            event_mask = event_mask | ((data.time>(self.events[i].de_get_epoch() - self.events[i].de_transit_width()/2)) &  (data.time<(self.events[i].de_get_epoch() + self.events[i].de_transit_width()/2)))
        event_mask = ~event_mask
        
        for i in range(self.aliases.shape[0]):
            phase = bruce.data.phase_times(data.time, epoch , self.max_period/self.aliases[i], phase_offset=0.2)
            phase_ratio = self.events[0].fit_period/(self.max_period/self.aliases[i])
            phase_width = width/(self.max_period/self.aliases[i])
            # Now lets get the data in-transit
            mask = (phase[event_mask]>(-phase_width/2)) & (phase[event_mask]<(phase_width/2))
            phase_L = phase[event_mask][mask]
            time_L = data.time[event_mask][mask]
            flux_L = (data.flux[event_mask]/data.w[event_mask])[mask]
            flux_err_L = (data.flux_err[event_mask]/data.w[event_mask])[mask]
            argsort = np.argsort(phase_L)
            phase_L,flux_L,flux_err_L = phase_L[argsort],flux_L[argsort],flux_err_L[argsort]
            model_L = np.interp(phase_L, phase_model*phase_ratio, phase_model_flux)
            # Now compare the model to null model to get deltaL
            if time_L.shape[0]==0 : delta_L[i] = 0.
            else:
                model_L = bruce.sampler.loglike(flux_L, flux_err_L, model_L, jitter=0.0, offset=False)
                null_L = bruce.sampler.loglike(flux_L, flux_err_L, np.ones(flux_L.shape[0], dtype=np.float64), jitter=0.0, offset=False)
                delta_L[i] = model_L - null_L 
        return delta_L

    def plot_aliases(self,  phot_data=[], phot_data_labels=[]):
        epoch = self.events[0].de_get_epoch()
        width = self.events[0].de_transit_width()
                
        
        plt.rcParams['font.size'] = 3
        fig, ax = plt.subplots(self.aliases.shape[0], 1+len(phot_data), figsize=(2 + 2*len(phot_data),0.8*self.aliases.shape[0]))
        ax = np.atleast_2d(ax)
        if len(phot_data)==0 : ax = ax.T

        phase_model, phase_model_flux = self.events[0].de_phase_model()
        depth = phase_model_flux.max() - phase_model_flux.min()
        ylim = (1 - 2*depth, 1+depth)
        # t_bin ,f_bin, fe_bin, _ = bin_data(self.data.time, self.data.flux, 0.5/24)
        # w_bin = np.interp(t_bin, self.data.time, self.data.w)
        # event_mask_bin = np.zeros(t_bin.shape[0], dtype=bool)
        # for i in range(len(self.events)):
        #     event_mask_bin = event_mask_bin | ((t_bin>(self.events[i].de_get_epoch() - self.events[i].de_transit_width()/2)) &  (t_bin<(self.events[i].de_get_epoch() + self.events[i].de_transit_width()/2)))
        # event_mask_bin = ~event_mask_bin


        # Let's create the color mask
        alias_colours = np.empty((self.aliases.shape[0], 1 + len(phot_data)), dtype='<U11')
        alias_colours[:] = ''

        # Base red mask
        alias_colours[self.delta_L[0] < (-self.height)] = 'xkcd:salmon'

        for i in range(1, len(phot_data) + 1):
            # Lets get the red mask 
            red_mask = (alias_colours[:, :i]=='xkcd:salmon').any(axis=1) | (self.delta_L[i] < (-self.height))
            if True in (self.delta_L[i] >self.height):
                red_mask |= ~(self.delta_L[i] >self.height)

            green_mask = ~(alias_colours[:, :i]=='xkcd:salmon').any(axis=1) & (self.delta_L[i] >self.height)
            alias_colours[red_mask, i:] = 'xkcd:salmon'
            alias_colours[green_mask, i:] = 'xkcd:green'

        for i in range(alias_colours.shape[0]):
            for j in range(alias_colours.shape[1]):
                if len(alias_colours[i,j])>1 : ax[i,j].set_facecolor(alias_colours[i,j])
                title = 'Alias P{:} [{:.3f} d]\n{:}'.format(self.aliases[i], self.max_period/self.aliases[i], phot_data_labels[j])
                ax[i,j].set(xlabel='PHASE', ylabel='FLUX', title=title)

        

        for i in tqdm(range(self.aliases.shape[0])):
            phase = bruce.data.phase_times(self.data.time, epoch , self.max_period/self.aliases[i], phase_offset=0.2)
            # print(self.data.flux[self.event_mask])
            # print(self.data.flux)
            # print(self.data.w[self.event_mask])
            # print(self.data.w)
            # print(self.data.flux[self.event_mask]/self.data.w[self.event_mask])

            ax[i][0].errorbar(phase[self.event_mask], self.data.flux[self.event_mask]/self.data.w[self.event_mask], yerr=np.abs(self.data.flux_err[self.event_mask]/self.data.w[self.event_mask]), fmt='k.', alpha = 0.1)
            
            phase_ratio = self.events[0].fit_period/(self.max_period/self.aliases[i])
            ax[i][0].plot(phase_model*phase_ratio, phase_model_flux, c='orange', zorder=10)
            
            phase_width = width/(self.max_period/self.aliases[i])
            ax[i][0].set_xlim(-phase_width,phase_width)
            
            
            
            for j in range(len(phot_data)):
                phase = bruce.data.phase_times(phot_data[j].time, epoch , self.max_period/self.aliases[i], phase_offset=0.2)
                ax[i][j+1].errorbar(phase, phot_data[j].flux/phot_data[j].w, yerr=np.abs(phot_data[j].flux_err/phot_data[j].w), fmt='k.', alpha = 0.1)

                
        
                phase_ratio = self.events[0].fit_period/(self.max_period/self.aliases[i])
                phase_width = width/(self.max_period/self.aliases[i])
                mask = (phase>(-phase_width/2)) & (phase<(phase_width/2))
                if mask.sum()>0:
                    phase_ratio = self.events[0].fit_period/(self.max_period/self.aliases[i])
                    ax[i][j+1].plot(phase_model*phase_ratio, phase_model_flux, c='orange', zorder=10)
                ax[i][j+1].set_xlim(-phase_width,phase_width)
                ax[i][j+1].set_ylim(ylim)

                # if self.delta_L[j+1][i] > self.height : ax[i][j+1].set_facecolor('xkcd:green')
                # if self.delta_L[j+1][i] < (-self.height) : ax[i][j+1].set_facecolor('xkcd:salmon')

        plt.tight_layout()

        def alias_colours_2_mask(alias_colours):
            alias_colours_ = np.ones(alias_colours.shape, dtype=int)
            alias_colours_[alias_colours=='xkcd:salmon'] = 0
            alias_colours_[alias_colours=='xkcd:green'] = 2
            return alias_colours_
        
        self.alias_mask = alias_colours_2_mask(alias_colours)
        return fig, ax
        
    def calculate_alias_probability_period(self,M_sun=1.,R_sun=1.):
        # Get the alias periods
        P_days = np.array(self.max_period/self.aliases)
        
        # Convert inputs to SI units
        R_star = R_sun * 6.957e8  # Convert solar radii to meters
        M_star = M_sun * 1.989e30  # Convert solar masses to kg
        P_sec = P_days * 86400  # Convert days to seconds

        # Compute semi-major axis using Kepler's third law
        a = ((G * M_star * P_sec**2) / (4 * np.pi**2))**(1/3)
        
        probabilities = R_star / a
        
        # Now normalise
        probabilities = probabilities/probabilities.sum()
        
        return probabilities
    
    
    def stellar_density(self,P_days, R1_a, b, e=0, omega=np.pi/2):
        """
        Calculate the mean density of the host star from transit parameters,
        accounting for eccentricity and argument of periastron.

        Parameters:
        P_days : float
            Orbital period in days.
        R1_a : float
            Ratio of stellar radius to semi-major axis (R1/a).
        b : float
            Impact parameter.
        e : float, optional
            Orbital eccentricity (default is 0, circular orbit).
        omega : float, optional
            Argument of periastron in radians (default is 0).

        Returns:
        float: Stellar density in kg/m^3.
        """
        # Correct R1/a for eccentricity and argument of periastron
        f_e_omega = (1 - e**2) / (1 + e * np.sin(omega))
        a_R1_true = 1 / (R1_a * np.sqrt(1 - b**2) * f_e_omega)

        # Compute stellar density
        rho_star = (3 * np.pi / (G * (P_days * 86400)**2)) * (a_R1_true**3)

        return rho_star

    def calcualte_alias_probability_density(self, density, density_error):
        rho_calculated = np.array([self.stellar_density(i, self.events[0].de_get_radius_1() * (i/self.events[0].fit_period)**(-2/3), self.events[0].de_get_b(), e=0, omega=np.pi/2) for i in self.max_period/self.aliases])
        chi = (density-rho_calculated)**2 / (density_error**2)
        probability = 1 - scipy.stats.chi2.cdf(chi, 3)
        if probability.sum()>0 : return probability/probability.sum()
        else : return np.zeros(probability.shape[0])
    
    def calculate_probability_with_data(self,):
        probability = 1 - scipy.stats.chi2.cdf(self.delta_L, 3)
        probability = probability/probability.sum()
        return probability
    
    
    def calcualate_alias_probabilities(self,
                                       M_sun=None,R_sun=None, 
                                       density=None, density_error=None,
                                      plot=True):
        probability=np.ones(self.aliases.shape[0])
        probability_calc = np.zeros((2,self.aliases.shape[0]))
        if M_sun is not None :    probability_calc[0] = probability_calc[0] + self.calculate_alias_probability_period(M_sun=M_sun,R_sun=R_sun)
        if density is not None :  probability_calc[1] = probability_calc[1] + self.calcualte_alias_probability_density(density=density, density_error=density_error)

        if plot:
            prob_plot = probability_calc.sum(axis=0)
            prob_plot = prob_plot / prob_plot.sum()
            fig, ax = plt.subplots(1,1) #figsize=(6.4*1.4, 4.8) 
            for i in range(len(self.aliases)):
                if (self.delta_L[:,i]<-self.height).sum()>0 : c='r'
                elif (self.delta_L[:,i]>self.height).sum()>0 : c='g'
                else : c='k'
                plt.bar(self.max_period/self.aliases[i], 100*prob_plot[i],
                        width=0.5, color=c, edgecolor=c,
                       log=True,align="edge")
            ax.set_xscale('log')
            ax.set(xlabel='Period', ylabel='Normalised probability [%]', title='Most probable period is \nalias P{:} with period {:.5f} days'.format(self.aliases[np.argmax(prob_plot)], (self.max_period/self.aliases)[np.argmax(prob_plot)]))
            
            ax_twinx = ax.twinx()
            if probability_calc[0].sum() != 0 : ax_twinx.plot(self.max_period/self.aliases, probability_calc[0]/probability_calc[0].max(), label='Geometric probability', c='blue', alpha = 0.3, lw=1)
            if probability_calc[1].sum() != 0 : ax_twinx.plot(self.max_period/self.aliases, probability_calc[1]/probability_calc[1].max(), label='Density probability', c='orange', alpha = 0.3, lw=1)
            ax_twinx.legend()
            ax_twinx.set_yticks([])
            plt.tight_layout()
            return fig, ax, probability_calc
        if probability_calc.sum()>0 : return probability_calc
        else : return probability
        
    def count_delta_L_solutions(self,):
        solutions, periods = [], []
        for i in range(len(self.aliases)):
            # if (self.delta_L[:,i]>self.height).sum()>0 : 
            #     solutions.append(i)
            #     periods.append(self.max_period/self.aliases[i])
            if self.delta_L.sum(axis=0)[i]>0 : 
                solutions.append(i)
                periods.append(self.max_period/self.aliases[i])


                
        return periods, solutions 
 
    def evaluate_peaks(self,):
        plt.rcParams['font.size'] = 10
        fig,ax = plt.subplots(1,1)
        c = 0.5*np.ones(self.delta_L.shape[1])
        # c[self.delta_L<-self.height] = 0.
        # c[self.delta_L>self.height] = 1

        c[(self.delta_L<-self.height).sum(axis=0)>0] = 0.
        c[(self.delta_L>self.height).sum(axis=0)>0] = 1

        plt.scatter(self.max_period/self.aliases, c, c=c)
        plt.gca().set_xscale('log')
        plt.xlabel('Period [days]')
        plt.ylabel('0 - excluded\n0.5 - no evidence\n1 - Eveidence for transit')
        plt.tight_layout()
        return fig, ax, c

        
        
    def plot_solution(self,alias):
        alias += 1
        plt.rcParams['font.size'] = 10
        fig, ax = plt.subplots(1,1)
        phase = bruce.data.phase_times(self.data.time, self.events[0].de_get_epoch(), self.max_period/alias, phase_offset=0.2)
        width = self.events[0].de_transit_width()
        phase_width = width/(self.max_period/alias)
        ax.errorbar(phase, self.data.flux/ self.data.w, yerr=np.abs(self.data.flux_err/self.data.w), fmt='k.', alpha = 0.1)
        phase_model, phase_model_flux = self.events[0].de_phase_model()
        phase_ratio = self.events[0].fit_period/(self.max_period/alias)
        ax.plot(phase_model*phase_ratio, phase_model_flux, c='orange', zorder=10)
        ax.set_xlim(-phase_width, phase_width)
        ax.set(xlabel='Phase', ylabel='Flux', title='Solved with\nt_zero : {:.5f}\nperiod {:.5f}'.format(self.events[0].de_get_epoch(), self.max_period/alias))
        plt.tight_layout()
        return fig, ax 
        
    def transit_plan(self, 
                    start=Time.now(), end = Time.now()+TimeDelta(30, format='jd'), resolution = 1*u.minute,
                    tic_id=None, observatory='Paranal',
                    sun_max_alt=-15, target_min_alt=30, moon_min_seperation=20,
                    min_time_in_transit=None, min_frac_in_transit=None):
        # Now query tic8 
        tic8 = Catalogs.query_object('TIC{:}'.format(tic_id), radius=.02, catalog="TIC")[0]
        target = SkyCoord(tic8['ra']*u.deg, tic8['dec']*u.deg)
        
        time_range = time_grid_from_range((start,end), time_resolution=resolution)

        # Define observer's location (example: Mauna Kea, Hawaii)
        location = EarthLocation.of_site(observatory)  # Or use .from_geodetic(lon, lat, height)

        # Define the AltAz frame
        altaz_frame = AltAz(obstime=time_range, location=location)
        
        # Transform target coordinates to AltAz
        target_altaz = target.transform_to(altaz_frame)
        
        # Get Sun's ICRS coordinates
        sun_icrs = get_sun(time_range)

        # Transform the Sun's position to AltAz
        sun_altaz = sun_icrs.transform_to(altaz_frame)
        
        # Lets get the moon
        moon_icrs = get_body("moon", time_range, location)
        #moon_altaz = moon_icrs.transform_to(altaz_frame)
        
        # Moon-target_seperation
        moon_target_seperation = moon_icrs.separation(target)

        # Now get the time we can observe
        time_mask = (sun_altaz.alt.deg<sun_max_alt) & (target_altaz.alt.deg>target_min_alt) & (moon_target_seperation.deg>moon_min_seperation)
        #time_mask_sun = (sun_altaz.alt.deg<sun_max_alt) 
        
        # Now get the phases of all aliases
        phases = np.zeros((self.aliases.shape[0], len(time_range)))
        for i in range(len(phases)):
            phases[i] = bruce.data.phase_times(time_range.jd, self.events[0].de_get_epoch(), self.max_period/self.aliases[i], phase_offset=0.2)
        phase_widths = self.events[0].de_transit_width() / (self.max_period/self.aliases)
        
        # Now get a booleon mask of the transits
        visible_transits = np.abs(phases) < (phase_widths[:,np.newaxis]/2) # Make for a nice plot
        visible_transits[:,~time_mask] = 0.
        visible_transits[~(self.alias_mask[:,-1]==self.alias_mask.max())]=0 # exclude bad ones

        if visible_transits.sum()==0:
            print('No transists visible')
            return Table()
        
        transit_nights = []
        # Now unpick this
        for i in range(visible_transits.shape[0]):
            if visible_transits[i].sum()==0 : continue
                
            # Find unique nights of transits
            time_in_transit = time_range[visible_transits[i]==1]
            segments = bruce.data.find_nights_from_data(time_in_transit.jd, dx_lim=0.2)
            
            # Get the parameters
            nights = [Observer(EarthLocation.of_site(observatory)).sun_set_time(time_in_transit[j][0], which='previous').datetime.__str__()[:10] for j in segments]
            for j in range(len(nights)):
                event = {}
                event['aliasP'] = i
                event['aliasPer'] = self.max_period/self.aliases[i]
                event['time_in_transit']=(time_in_transit[segments[j]].shape[0]*resolution).to(u.day).value
                event['frac_in_transit']=event['time_in_transit']/self.events[0].de_transit_width()
                event['moon_mean'] = np.mean(moon_target_seperation[visible_transits[i]==1][segments[j]])
                event['night'] = nights[j]
                event['cycle_number'] = int(np.round((time_in_transit[segments[j]][0].jd - self.events[0].de_get_epoch()) / (self.max_period/self.aliases[i])))
                event['transit_center'] = self.events[0].de_get_epoch() + event['cycle_number']*(self.max_period/self.aliases[i])
                event['transit_width'] = self.events[0].de_transit_width()
                transit_nights.append([int(tic_id), str(observatory),float(sun_max_alt),float(target_min_alt),float(moon_min_seperation), str(event['night']) , int(event['aliasP']), float(event['aliasPer']), float(self.events[0].de_get_epoch()) , float(event['time_in_transit']),float(event['frac_in_transit']), float(event['moon_mean'].value), int(event['cycle_number']), float(event['transit_center']), float(event['transit_width'])])
            

        t = Table()
        names=['tic_id','observatory','sun_max_alt','target_min_alt','moon_min_seperation','night','aliasP', 'aliasPer','alias_epoch', 'time_in_transit', 'frac_in_transit', 'moon_mean_sep', 'cycle_number', 'transit_center', 'transit_width']
        dtypes = [int,   str,         float,         float,         float,                  str,    int,       float,    float,          float,            float,            float,              int,          float,           float ]
        for i in range(len(names)):
            t.add_column(Column(np.array(transit_nights)[:,i].astype(dtypes[i]), name=names[i]))
        if min_time_in_transit is not None : t = t[np.array(t['time_in_transit'], dtype=np.float32) > min_time_in_transit]
        if min_frac_in_transit is not None : t = t[np.array(t['frac_in_transit'], dtype=np.float32) > min_frac_in_transit]
        return t

    def plot_all_events(self, table,
                resolution = 1*u.minute, output_dir='.'):
        for i in range(len(table)) :
            output_file = output_dir + '/' + 'EVENTS_FROM_{:}_{:}.png'.format(table['observatory'][i], table['night'][i]) 
            if os.path.isfile(output_file) : continue 
            else:
                fig, _ = self.plot_event(table[i],resolution = resolution)
                plt.savefig(output_file)
                plt.close(fig)
                gc.collect()
                del fig
    
    def plot_event(self, row,
                resolution = 1*u.minute):    
        # Now query tic8 
        tic8 = Catalogs.query_object('TIC{:}'.format(row['tic_id']), radius=.02, catalog="TIC")[0]
        target = SkyCoord(tic8['ra']*u.deg, tic8['dec']*u.deg)
        
        midday = Time(row['night']+'T12:00:00')
        start = Observer(EarthLocation.of_site(row['observatory'])).sun_set_time(midday, which='next')
        end = Observer(EarthLocation.of_site(row['observatory'])).sun_rise_time(midday, which='next')
        time_range = time_grid_from_range((start,end), time_resolution=resolution)

        # Define the AltAz frame
        altaz_frame = AltAz(obstime=time_range, location=EarthLocation.of_site(row['observatory']))

        # Transform target coordinates to AltAz
        target_altaz = target.transform_to(altaz_frame)
        
        # Get Sun's ICRS coordinates
        #sun_icrs = get_sun(time_range)

        # Transform the Sun's position to AltAz
        #sun_altaz = sun_icrs.transform_to(altaz_frame)
        
        # Lets get the moon
        moon_icrs = get_body("moon", time_range, location=EarthLocation.of_site(row['observatory']))
        moon_altaz = moon_icrs.transform_to(altaz_frame)
        #moon_target_seperation = moon_icrs.separation(target)
        
        plt.rcParams['font.family'] = 'Times New Roman'
        fig, ax = plt.subplots(1,1, figsize=(8,4))
        target_up = target_altaz.alt.deg>float(row['target_min_alt'])
        target_in_transit = (time_range.jd > (float(row['transit_center'])-float(row['transit_width'])/2)) & (time_range.jd < (float(row['transit_center'])+float(row['transit_width'])/2))
        ax.plot_date(time_range.plot_date[target_up], target_altaz.alt.deg[target_up], 'k-', lw=1, alpha=1)
        ax.plot_date(time_range.plot_date[~target_up&~target_in_transit], target_altaz.alt.deg[~target_up&~target_in_transit], 'k--', lw=1, alpha = 0.4, label='TIC-{:}'.format(row['tic_id']))
        ax.plot_date(time_range.plot_date[target_in_transit&target_up], target_altaz.alt.deg[target_in_transit&target_up], 'r-', lw=1, alpha = 1)
        ax.plot_date(time_range.plot_date[target_in_transit&~target_up], target_altaz.alt.deg[target_in_transit&~target_up], 'r-', lw=1, alpha = 0.4)

        ax.plot_date([time_range.plot_date[target_in_transit][0]]*2, [target_altaz.alt.deg[target_in_transit][0]-5,target_altaz.alt.deg[target_in_transit][0]+5], 'r-', lw=1, alpha = 1)
        ax.plot_date([time_range.plot_date[target_in_transit][-1]]*2, [target_altaz.alt.deg[target_in_transit][-1]-5,target_altaz.alt.deg[target_in_transit][-1]+5], 'r-', lw=1, alpha = 1)

        xlim = ax.get_xlim()
        ylim=ax.get_ylim()
        ax.plot_date(time_range.plot_date, moon_altaz.alt.deg, 'b--', lw=1, alpha = 0.4, label='Moon')
        ax.set(xlim=xlim, ylim=ylim)
        
        
        title = 'TIC-{:}'.format(row['tic_id'])

        title +='\nVisible from {:} -> {:}'.format(time_range[target_up][0].datetime.__str__()[10:16],
                                                time_range[target_up][-1].datetime.__str__()[10:16])
        title +='\nTransit from {:} -> {:} [{:.0f} mins - {:.2f}%]'.format(time_range[target_in_transit][0].datetime.__str__()[10:16],
                                                time_range[target_in_transit][-1].datetime.__str__()[10:16],
                                                                (time_range[target_in_transit][-1]-time_range[target_in_transit][0]).jd*1440,
                                                                100*float(row['frac_in_transit']))
        title +='\nTransit visible from  {:} -> {:} [{:.0f} mins]'.format(time_range[target_in_transit&target_up][0].datetime.__str__()[10:16],
                                                time_range[target_in_transit&target_up][-1].datetime.__str__()[10:16],
                                                                (float(row['time_in_transit'])*1440))

        title +='\nObservatory : ' + row['observatory'] + ' [{:}]'.format(row['night'])
        title += '\nAlias P{:.0f} [{:.5f} days]'.format(float(row['aliasP']), float(row['aliasPer']))
        plt.xlabel('Time')
        plt.ylabel('Altitude')
        plt.title(title, ha='left', loc='left')
        plt.tight_layout()
        return fig, ax