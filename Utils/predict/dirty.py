import os
from astropy.table import Table 
from multiprocessing import Pool

t = Table.read('ngts.csv', format='csv')
observatory='SAAO'
ntransits = 1

home = os.getcwd()
date = '2019-09-03'

def func(i):
    # Change to plans
    os.chdir('plans')

    args = [observatory, t['ra'][i], t['dec'][i], t['t_zero'][i], t['period'][i], t['width'][i], ntransits, t['name'][i], date, t['name'][i] ]

    # Make the call
    os.system('predict --observatory {:} --ra {:} --dec {:} --t_zero {:} --period {:} --width {:} --ntransits {:} --name {:} --plot --date {:} --utc > {:}.plan'.format(*args))

    os.chdir(home)


pool = Pool(10)
pool.map(func, range(len(t)))
pool.close()
pool.join() 
