def main():
    from astropy.table import Table
    import sys
    import numpy as np
    from astropy.table import Table, vstack
    import matplotlib.pyplot as plt
    import bruce


    def get_constant(BJD, FLUX):
        # Now get the constant
        quarter_edges = np.linspace(np.min(BJD)-1e-3,np.max(BJD)+1e-3,5)
        dig = np.digitize(BJD, quarter_edges, right=True)
        medians = np.array([np.nanmedian(FLUX[dig==i]) for i in range(1,len(quarter_edges))])
        return np.nanmax(medians)

    def normalise_ngts_tables(tables, bin=5):
        N = [get_constant(t['BJD'],t['Flux']) for t in tables]
        for i, const in enumerate(N):
            tables[i]['FluxNorm'] = tables[i]['Flux']/const
            tables[i]['FluxNormErr'] = tables[i]['FluxErr']/const

        # Now bin
        t_total = vstack(tables)
        t_total.sort('BJD')
        t_bin, f_bin, fe_bin, c = bruce.data.bin_data(t_total['BJD'], t_total['FluxNorm'], bin/24/60)
        
        d = bruce.ambiguous_period.photometry_time_series(t_bin, f_bin, fe_bin)
        d.normalise_gui()
        return d


    tables = [Table.read(i) for i in sys.argv[1:-1]]
    d = normalise_ngts_tables(tables)
    d.write_data(sys.argv[-1])

