import numpy as np
from bruce.ambiguous_period import photometry_time_series
import bruce 
# # --------------------------------------------------
# # Example transit data
# # --------------------------------------------------

# np.random.seed(42)

# time = np.linspace(-0.2, 3.2, 5000)

# # fake transit
# flux = np.ones_like(time)
# for i in range(5):
#     flux -= 0.02 * np.exp(-((time-i*0.5)/0.03)**2)

# # continuum trend
# flux *= (1 + 0.02*time + 0.05*time**2)

# # noise
# flux += np.random.normal(0, 0.001, len(flux))
# flux_err = np.random.uniform(0.0005,0.0015, len(flux))

# a = photometry_time_series(time, flux, flux_err)
# a.normalise_gui()
# print(a.w)
# data = a.split_and_normalise_transit_events(t_zero=0, period=0.5, width=0.08*2, dx_lim=0.05)
# print(data)



_,d,d_label,_ = bruce.ambiguous_period.download_tess_data(32000979)
t_zero, period, width = 2459243.572,	121.6668282, 5.233432187/24 
for data in d:
    try : 
        data_events = data.split_and_normalise_transit_events(t_zero=t_zero, period=period, width=width*2, dx_lim=0.05)
        print(data_events)
    except :  continue
