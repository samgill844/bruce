

from astroquery.mast import Observations
from astroquery.mast import Observations
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table, Column
import numpy as np, re

def tess_products(ticid):
    # ------------------------------
    # Query MAST for TESS products
    # ------------------------------
    print(f"Querying TIC {ticid} from MAST...")

    obs = Observations.query_criteria(
        provenance_name=('TESS-SPOC', 'SPOC', 'QLP'),
        target_name=str(ticid)
    )

    # try:
    data = Observations.get_product_list(obs)
    # Keep only relevant files
    mask = np.zeros(len(data), dtype=bool)
    for ext in ('_lc.fits', '_llc.fits', '_tp.fits', '.pdf'):
        mask |= np.char.endswith(data['productFilename'], ext)
    data = data[mask]

    data['sector'] = np.zeros(len(data), dtype=int)
    for i in range(len(data)):
        # This pattern looks for 's' followed by one or more digits
        print(data['productFilename'][i])
        matches = matches = re.findall(r'[-_](?:s)(\d+)[-_]', data['productFilename'][i])
        data['sector'][i] = int(matches[0])  # 31 as an integer

    # except Exception:
    #     data = Table()


    # ------------------------------
    # Build JSON output
    # ------------------------------
    output = []

    for row in data:
        sector = row["sector"]
        output.append({
            "sector": int(sector) if sector is not None else None,
            "type": row["productSubGroupDescription"],
            "pipeline": row["provenance_name"],
            "filename": row["productFilename"],
            "url": f"https://mast.stsci.edu/api/v0.1/Download/file?uri={row['dataURI']}"
        })

    return  output 

if __name__ == "__main__":
    data = tess_products(1167538)










s = "hlsp_qlp_tess_ffi_s0005-0000000001167538_tess_v01_ll"

# This pattern looks for 's' followed by one or more digits
match = re.search(r's(\d+)', s)

if match:
    number = match.group(1)  # '0031' as a string
    number_int = int(number)  # 31 as an integer
    print(number, number_int)

