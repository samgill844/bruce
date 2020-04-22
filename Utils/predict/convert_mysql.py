from astropy.table import Table 
import numpy as np

t = Table.read('ngts_tran.dat', format='ascii')


new_table = []


for i in t:
    try:
        name = str(i['noi_id']) + '_' + i['lc_flag'] + '_' + i['tag']
        ra = float(i['ra_deg'])
        dec = float(i['dec_deg'])

        t_zero = float(float(i['epoch_sec'])/86400. + 6658.5 + 2450000.)    # epoch in HJD-2450000 / days
        period = float(float(i['period_sec'])/86400.)    # period/ days
        width = float(24*float(i['width_sec'])/86400.)   # width / days 
        new_table.append([name, ra, dec, t_zero, period, width])
    except : pass


new_table  = Table(np.array(new_table), names = ['name', 'ra', 'dec', 't_zero', 'period', 'width'], dtype = ['str', 'f8', 'f8', 'f8', 'f8', 'f8'])

new_table.write('ngts.csv', format='csv')