from genericpath import isfile
from astropy.table import Table, unique 
import os
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.simbad import Simbad


__all__=['download_NASA_exoplanet_archive', 'load_NASA_exoplanet_archive', 'download_tois_ctois', 'load_toi_table', 'load_ctoi_table', 'load_tic8_info_for_target', 'load_gaia_table', 'query_simbad']

def download_NASA_exoplanet_archive():
    cache_dir = os.path.expanduser("~") + '/.cache/bruce'
    os.system('mkdir -p {:}'.format(cache_dir))
    cmd = 'wget "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+ps&format=csv" -O "{:}/confirmed_planets.csv"'.format(cache_dir)
    os.system(cmd)


def query_simbad(id):
    return Simbad.query_objectids('{:}'.format(id))



def load_NASA_exoplanet_archive(unique_entries=False):
    path = os.path.expanduser("~") + '/.cache/bruce2/confirmed_planets.csv'
    if not os.path.isfile(path) : download_NASA_exoplanet_archive()
    t = Table.read(path, format='csv')
    if unique_entries : t = unique(t, keys= 'pl_name', keep='last')
    return t



def download_tois_ctois(user = 'u1870241'):
    cache_dir = os.path.expanduser("~") + '/.cache/bruce2'
    toi_cmd_file = '{:}/toi_cmd.sh'.format(cache_dir)
    if not os.path.isfile(toi_cmd_file):
        with open(toi_cmd_file, 'w+') as f : f.write("mysql -h ngtsdb -u pipe -e 'select * from tess.tois;'")
    ctoi_cmd_file = '{:}/ctoi_cmd.sh'.format(cache_dir)
    if not os.path.isfile(ctoi_cmd_file):
        with open(ctoi_cmd_file, 'w+') as f : f.write("mysql -h ngtsdb -u pipe -e 'select * from tess.ctois;'")
    
    print('Downloading TOIs... ', end='')
    toi_file = '{:}/tois.dat'.format(cache_dir)
    os.system('ssh {:}@ngtshead.warwick.ac.uk < {:} > {:}'.format(user, toi_cmd_file, toi_file))
    ctoi_file = '{:}/ctois.dat'.format(cache_dir)
    os.system('ssh {:}@ngtshead.warwick.ac.uk < {:} > {:}'.format(user, ctoi_cmd_file, ctoi_file))

    for file in [toi_file, ctoi_file]:
        lines = open(file, 'r').readlines()[2:]
        with open(file, 'w+') as f : f.writelines(lines)

def load_toi_table():
    cache_dir = os.path.expanduser("~") + '/.cache/bruce2'
    toi_file = '{:}/tois.dat'.format(cache_dir)
    return Table.read(toi_file, format='ascii')

def load_ctoi_table():
    cache_dir = os.path.expanduser("~") + '/.cache/bruce2'
    ctoi_file = '{:}/ctois.dat'.format(cache_dir)
    return Table.read(ctoi_file, format='ascii')

def load_tic8_info_for_target(tic_id=None, user = 'u1870241'):
    if tic_id is None : return 
    cache_dir = os.path.expanduser("~") + '/.cache/bruce2'
    tic8_object_cmd = '{:}/tic8_object_cmd.sh'.format(cache_dir)

    with open(tic8_object_cmd, 'w+') as f : f.write("mysql -h ngtsdb -u pipe -e 'select * from catalogues.tic8 where tic_id=%s;'" % tic_id)
    output_file = '{:}/tic8_object.dat'.format(cache_dir)
    os.system('ssh {:}@ngtshead.warwick.ac.uk < {:} > {:}'.format(user, tic8_object_cmd, output_file))

    lines = open(output_file, 'r').readlines()[2:]
    with open(output_file, 'w+') as f : f.writelines(lines)

    return Table.read(output_file, format='ascii')


def load_gaia_table(tic_id=None, data_release=3):
    from astroquery.gaia import Gaia

    if tic_id is None : return 
    Gaia.MAIN_GAIA_TABLE = 'gaiadr{:}.gaia_source'.format(data_release)

    tic8 = load_tic8_info_for_target(tic_id=tic_id)
    coord = SkyCoord(ra=tic8['ra_deg'][0], dec=tic8['dec_deg'][0], unit=(u.degree, u.degree), frame='icrs')
    radius = u.Quantity(30, u.arcsec)
    j = Gaia.cone_search_async(coord, radius)
    r = j.get_results()
    return tic8, r[0]
