from genericpath import isfile
from astropy.table import Table, unique 
import os
from urllib.request import urlretrieve
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.simbad import Simbad


__all__=['download_NASA_exoplanet_archive', 'load_NASA_exoplanet_archive', 'download_tois_ctois', 'load_toi_table', 'load_ctoi_table', 'load_tic8_info_for_target', 'load_gaia_table', 'query_simbad']

def download_NASA_exoplanet_archive():
    cache_dir = os.path.expanduser("~/.cache/bruce")
    os.makedirs(cache_dir, exist_ok=True)
    urlretrieve(
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+ps&format=csv",
        os.path.join(cache_dir, "confirmed_planets.csv"),
    )


def query_simbad(id):
    return Simbad.query_objectids('{:}'.format(id))



def load_NASA_exoplanet_archive(unique_entries=False):
    path = os.path.expanduser("~") + '/.cache/bruce/confirmed_planets.csv'
    if not os.path.isfile(path) : download_NASA_exoplanet_archive()
    t = Table.read(path, format='csv')
    if unique_entries : t = unique(t, keys= 'pl_name', keep='last')
    return t



def download_tois_ctois(user=None):
    """Run the optional local-only TOI/CTOI downloader.

    The institutional implementation is not distributed with BRUCE.
    """
    from ._local_archive_loader import load_local_archive
    helper = load_local_archive().download_tois_ctois
    return helper() if user is None else helper(user=user)

def load_toi_table():
    cache_dir = os.path.expanduser("~") + '/.cache/bruce2'
    toi_file = '{:}/tois.dat'.format(cache_dir)
    return Table.read(toi_file, format='ascii')

def load_ctoi_table():
    cache_dir = os.path.expanduser("~") + '/.cache/bruce2'
    ctoi_file = '{:}/ctois.dat'.format(cache_dir)
    return Table.read(ctoi_file, format='ascii')

def load_tic8_info_for_target(tic_id=None, user=None):
    """Query TIC information through the optional local-only implementation."""
    if tic_id is None:
        return None
    from ._local_archive_loader import load_local_archive
    helper = load_local_archive().load_tic8_info_for_target
    return helper(tic_id=tic_id) if user is None else helper(tic_id=tic_id, user=user)


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
