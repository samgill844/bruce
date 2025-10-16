#!/Users/sam/anaconda3/bin/ipython
from pyqtgraph.Qt import QtGui
import numpy as np, os#, pyqtgraph as pg
from astropy.table import Table, Column
from scipy.stats import median_abs_deviation
import pyqtgraph as pg
from astropy.io import fits
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QLineEdit , QCheckBox
from pyqtgraph.dockarea import *
from astroquery.mast import Catalogs
import warnings
warnings.filterwarnings("ignore")
from astroquery.mast import Catalogs
from astropy.wcs import WCS 
import argparse
from astroquery.mast import Tesscut
import lightkurve

from astropy.coordinates import SkyCoord


import urllib.request, json , glob, zipfile
from astroquery.mast import Catalogs
import tempfile

def parse_args():
    description = '''A program to searc spoc lightcurves for transit signals'''

    # Argument parser
    parser = argparse.ArgumentParser('spocsearch', description=description)

    parser.add_argument('-e',
                    "--tic_id",
                    help='The filename of the SPOC lightcurve',
                    default=None)

    parser.add_argument('-a', 
                    '--sector',
                    help='The sector',
                    default=None, type=int)

    parser.add_argument('-b', 
                    '--size',
                    help='The size of the cutout in pixels.',
                    default=10, type=int)

    parser.add_argument('-c', 
                    '--mask',
                    help='The pixel mask.',
                    default=None, type=str)

    parser.add_argument('-d',
                        '--coordinates',
                        help = 'The coordinates of the target.',
                        type = float,
                        nargs='+',
                        default=[])
    

    parser.add_argument('--tica', action="store_true", default=False)
    parser.add_argument('--all', action="store_true", default=False)
    parser.add_argument('--nogui', action="store_true", default=False)


    return parser.parse_args()


class Window():
    def __init__(self, TIME, FLUX,QUALITY_MASK, ticid, wcs, sector, mask, filename, type):
        # Set the defaults
        self.TIME = TIME
        self.FLUX = FLUX
        self.QUALITY_MASK = QUALITY_MASK
        self.ticid = ticid 
        self.wcs = wcs
        self.sector = sector
        self.lightkurve = lightkurve.targetpixelfile.TessTargetPixelFile(filename)
        self.lightkurve.row 

        if mask is None : self.calculate_pixel_aperture_mask()
        else : self.pixel_aperture_mask = np.load(mask)

        # Create the GUI
        self.app = pg.mkQApp("TESS TPF") 
        self.win = QtGui.QMainWindow()


        #################################################################
        # Create the window and the dock area
        #################################################################
        #self.area = DockArea()
        #self.win.setCentralWidget(self.area)
        #self.win.resize(1300,800)
        #self.win.showFullScreen()
        self.win.showMaximized()
        self.win.setStyleSheet('QCheckBox {background-color: #7BAD98}')

        title_text = '{:} S{:02} TPF [TIC-{:}]'.format(type, sector, self.ticid)
        self.win.setWindowTitle(title_text)

        #################################################################
        # Create the first dock which holds the plots in
        #################################################################
        #self.d1 = Dock('TIC-{:}'.format(ticid), size=(1000,800))     ## give this dock the minimum possible size

        #################################################################
        # Add the docks
        #################################################################
        #self.area.addDock(self.d1)      ## place d1 at left edge of dock area (it will fill the whole space since there are no other docks yet)
        #self.area.addDock(self.d_frame, 'right', self.d1 )     ## place d2 at right edge of dock area


        #################################################################
        # Create the image view
        ################################################################# 
        self.image_view = pg.ImageView()
        self.win.setCentralWidget(self.image_view)
        self.image_view_plotview = self.image_view.getView()


        self.image_view_roi = self.image_view.getRoiPlot()
        if args.tica: self.image_view_roi.setLabel('left', 'Flux [e]')
        else :        self.image_view_roi.setLabel('left', 'Flux [e/s]')
        self.image_view_roi.setLabel('bottom', 'BTJD')
        self.image_view_roi.setPos(3,3)

        self.image_view.setLevels(-500, 2000)

        self.lr = pg.LinearRegionItem([self.TIME[0], self.TIME[100]], bounds=[np.min(self.TIME), np.max(self.TIME)], movable=True)
        self.image_view_roi.addItem(self.lr)
        self.lr.sigRegionChanged.connect(self.update_plots)

        # Create the pixel lines
        self.pixel_lightcurve_pen = pg.mkPen((255,0,0,100))
        self.pixel_lines = [pg.PlotCurveItem(x=[], y=[],  pen =self.pixel_lightcurve_pen) for i in range(self.FLUX.shape[1]*self.FLUX.shape[2])]
        for line in self.pixel_lines : self.image_view_plotview.addItem(line)

        # Create the centroid scatter item
        self.centroid_scatter_plot_item_in = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush('blue'))
        self.centroid_scatter_plot_item_in.setZValue(10)
        self.image_view_plotview.addItem(self.centroid_scatter_plot_item_in)
        self.centroid_scatter_plot_item_out = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush('red'))
        self.centroid_scatter_plot_item_out.setZValue(9)
        self.image_view_plotview.addItem(self.centroid_scatter_plot_item_out)

        # Now create the checkbox to go with it
        self.plot_lc_check_box = QCheckBox("Plot pixel lightcurves")
        self.image_view.scene.addWidget(self.plot_lc_check_box)
        self.plot_lc_check_box.stateChanged.connect(self.plot_pixel_lcs)
        self.plot_lc_check_box.setStyleSheet('QCheckBox {background-color: #007BAD98; color: white}')

    
        # Create the source scatter item
        self.source_pen = pg.mkPen((0,255,0,100))
        self.source_scatter = pg.ScatterPlotItem(size=10, pen=self.source_pen)
        self.image_view_plotview.addItem(self.source_scatter)

        # Create the text
        self.sources_text = []

        # Make the source check boz
        self.plot_sources_check_box = QCheckBox("Plot sources")
        self.image_view.scene.addWidget(self.plot_sources_check_box)
        self.plot_sources_check_box.move(0, 20)
        self.plot_sources_check_box.stateChanged.connect(self.plot_sources)
        self.plot_sources_check_box.setStyleSheet('QCheckBox {background-color: #007BAD98; color: white}')

        # Now add the Tmin box 
        self.tmag_min_box = QLineEdit('0')
        self.tmag_min_cahce = '0'
        self.image_view.scene.addWidget(self.tmag_min_box)
        self.tmag_min_box.move(80, 40)
        self.tmag_min_box.setFixedWidth(50)

        self.tmag_minlabel = QLabel("min Tmag")
        self.image_view.scene.addWidget(self.tmag_minlabel)
        self.tmag_minlabel.move(0, 40)
        self.tmag_minlabel.setStyleSheet('QLabel {background-color: #007BAD98; color: white}')

        # Now add the Tmax box 
        self.tmag_max_box = QLineEdit('16')
        self.tmag_max_cache = '16'
        self.image_view.scene.addWidget(self.tmag_max_box)
        self.tmag_max_box.move(80, 60)
        self.tmag_max_box.setFixedWidth(50)

        self.tmag_maxlabel = QLabel("Max Tmag")
        self.image_view.scene.addWidget(self.tmag_maxlabel)
        self.tmag_maxlabel.move(0, 60)
        self.tmag_maxlabel.setStyleSheet('QLabel {background-color: #007BAD98; color: white}')
        
        # Make the pixel mask box
        self.pixel_mask_check_box = QCheckBox("Pixel mask")
        self.image_view.scene.addWidget(self.pixel_mask_check_box)
        self.pixel_mask_check_box.move(0, 80)
        self.pixel_mask_check_box.stateChanged.connect(self.plot_pixel_aperture)
        self.pixel_mask_check_box.setStyleSheet('QCheckBox {background-color: #007BAD98; color: white}')
        self.image_view.scene.sigMouseClicked.connect(self.update_mask_on_click)


        # Now plot the LC 
        self.plot_lc_button = QtGui.QPushButton('Plot pixel lightcurve')
        self.image_view.scene.addWidget(self.plot_lc_button)
        self.plot_lc_button.move(0, 100)
        self.plot_lc_button.setStyleSheet('QPushButton {background-color: #007BAD98; color: white}')
        self.plot_lc_button.clicked.connect(self.plot_pixel_lightcurve)

        # Now add the save pixel mask
        self.save_pixel_mask_button = QtGui.QPushButton('Save pixel mask')
        self.image_view.scene.addWidget(self.save_pixel_mask_button)
        self.save_pixel_mask_button.move(0, 130)
        self.save_pixel_mask_button.clicked.connect(self.save_pixel_mask)
        self.save_pixel_mask_button.setStyleSheet('QPushButton {background-color: #007BAD98; color: white}')

        # Now add the save lightcurve button
        self.save_lightcurve_button = QtGui.QPushButton('Save lightcurve')
        self.image_view.scene.addWidget(self.save_lightcurve_button)
        self.save_lightcurve_button.move(0, 160)
        self.save_lightcurve_button.clicked.connect(self.save_lightcurve)
        self.save_lightcurve_button.setStyleSheet('QPushButton {background-color: #007BAD98; color: white}')

        # Now add the spocfit  button
        self.fit_lightcurve_button = QtGui.QPushButton('Fit the lightcurve')
        self.image_view.scene.addWidget(self.fit_lightcurve_button)
        self.fit_lightcurve_button.move(0, 190)
        self.fit_lightcurve_button.clicked.connect(self.fit_lightcurve)
        self.fit_lightcurve_button.setStyleSheet('QPushButton {background-color: #007BAD98; color: white}')


        # Now add the fluxweight  button
        self.flux_weight_check_box = QCheckBox("Plot the flux weighting")
        self.image_view.scene.addWidget(self.flux_weight_check_box)
        self.flux_weight_check_box.move(0, 220)
        self.flux_weight_check_box.stateChanged.connect(self.plot_flux_weighting)
        self.flux_weight_check_box.setStyleSheet('QCheckBox {background-color: #007BAD98; color: white}')


        # Finally set anti alias
        pg.setConfigOptions(antialias=True)

        self.win.show()


    def update_plots(self,):
        self.plot_flux_weighting()
        self.plot_pixel_lcs()

    def plot_flux_weighting(self,):
        if self.flux_weight_check_box.isChecked():
            t_low, t_high = self.lr.getRegion()
            width = t_high - t_low
            inmask = (self.TIME > t_low) & (self.TIME < t_high)
            outmask = ((self.TIME > (t_low-width)) & (self.TIME < (t_high-width))) | ((self.TIME > (t_low+width)) & (self.TIME < (t_high+width)))
            x, y = self.lightkurve.estimate_centroids()
            x = np.array(x,dtype=np.float64) - self.lightkurve.column + 1
            y = np.array(y,dtype=np.float64) - self.lightkurve.row 
            self.centroid_scatter_plot_item_in.setData(x=x[inmask],y=y[inmask])
            self.centroid_scatter_plot_item_out.setData(x=x[outmask],y=y[outmask])
        else : 
            self.centroid_scatter_plot_item_in.clear()
            self.centroid_scatter_plot_item_out.clear()


    def fit_lightcurve(self,call=True):
        fname = 'TIC-{:}_S{:02}_TPF_lightcurve.fits'.format(self.ticid, self.sector)
        t = Table()
        t.add_column(Column(self.TIME, name='TIME'))
        sky_counts_per_pixel = np.median(self.FLUX[:,self.pixel_aperture_mask==1], axis=1)
        target_counts = self.FLUX[:,self.pixel_aperture_mask==2]
        target_counts = np.sum(target_counts, axis=1) - sky_counts_per_pixel*target_counts.shape[1]
        t.add_column(Column(target_counts, name='SAP_FLUX'))
        t.add_column(Column(target_counts, name='PDCSAP_FLUX'))
        t.add_column(Column(np.ones(len(t))*median_abs_deviation(t['SAP_FLUX']), name='PDCSAP_FLUX_ERR'))
        t.add_column(Column(np.sum(self.FLUX[:,self.pixel_aperture_mask==1], axis=1), name='SAP_BKG'))
        t.add_column(Column(np.ones(len(t))*median_abs_deviation(t['SAP_BKG']), name='SAP_BKG_ERR'))

        t.add_column(Column(self.QUALITY_MASK, name='QUALITY_MASK'))
        t.write(fname, overwrite=True)

        with fits.open(fname, 'update') as h:
            h[0].header['TICID'] = int(self.ticid)
            h[0].header['SECTOR'] = int(self.sector)

        if call : os.system('spocfit {:} &'.format(fname))
        else : return fname


    def save_lightcurve(self,):
        fname = 'TIC-{:}_S{:02}_TPF_lightcurve.fits'.format(self.ticid, self.sector)
        t = Table()
        t.add_column(Column(self.TIME, name='BTJD'))
        sky_counts_per_pixel = np.median(self.FLUX[:,self.pixel_aperture_mask==1], axis=1)
        target_counts = self.FLUX[:,self.pixel_aperture_mask==2]
        target_counts = np.sum(target_counts, axis=1) - sky_counts_per_pixel*target_counts.shape[1]
        t.add_column(Column(target_counts, name='FLUX'))
        t.add_column(Column(np.ones(len(t))*median_abs_deviation(t['FLUX']), name='FLUX_ERR'))
        t.add_column(Column(sky_counts_per_pixel, name='MEDIAN_BKG'))
        t.add_column(Column(np.sum(self.FLUX[:,self.pixel_aperture_mask==1], axis=1), name='TOTAL_BKG'))
        t.add_column(Column(self.QUALITY_MASK, name='QUALITY_MASK'))
        t.write(fname, overwrite=True)
        print('Saved to '+fname)

    def save_pixel_mask(self):
        fname = 'TIC-{:}_S{:02}_pixel_mask'.format(self.ticid, self.sector)
        np.save(fname, self.pixel_aperture_mask)
        print('Saved to '+fname+'.npy')


    def plot_pixel_lightcurve(self,):
        sky_counts_per_pixel = np.median(self.FLUX[:,self.pixel_aperture_mask==1], axis=1)
        target_counts = self.FLUX[:,self.pixel_aperture_mask==2]
        target_counts = np.sum(target_counts, axis=1) - sky_counts_per_pixel*target_counts.shape[1]
        plt = pg.plot()
        line1 = plt.plot(self.TIME, target_counts, pen='r')
        plt.setLabel('left', 'Flux', units ='e/s')
        plt.setLabel('bottom', 'Time', units ='BTJD')

    def update_mask_on_click(self, event):
        event = self.image_view_plotview.mapSceneToView(event.pos())
        i = int(np.floor(event.x()))
        j = int(np.floor(event.y()))
        self.pixel_aperture_mask[i,j] +=1
        if self.pixel_aperture_mask[i,j]==3 : self.pixel_aperture_mask[i,j] =0
        self.plot_pixel_aperture()

    def calculate_pixel_aperture_mask(self,):
        self.pixel_aperture_mask = np.zeros((self.FLUX.shape[1], self.FLUX.shape[2]), dtype = int)                
        self.pixel_aperture_mask[self.FLUX[30] < np.percentile(self.FLUX[30],30)] = 1
        self.pixel_aperture_mask[self.lightkurve.create_threshold_mask()] = 2


    def draw_data(self,):
        # update the image view 2 data
        self.image_view.setImage(self.FLUX, xvals=self.TIME)
        img_median=np.median(self.FLUX,)
        img_rms=1.48*np.median(np.abs(self.FLUX-img_median))
        zmin = img_median-img_rms*5
        zmax = img_median+img_rms*9
        self.image_view.setLevels(zmin, zmax)




    def plot_pixel_lcs(self,):

        if self.plot_lc_check_box.isChecked() : 
            count = 0
            for i in range(self.FLUX.shape[1]):
                for j in range(self.FLUX.shape[2]):
                    t_low, t_high = self.lr.getRegion()
                    mask = (self.TIME > t_low) & (self.TIME < t_high)
                    # Now need to scale data between j and j+1 in x
                    X = self.TIME[mask]
                    X = i  + (X - np.min(X)) / (np.max(X) - np.min(X))
                    # Y needs to be scaled between i and i+1 
                    Y =  -self.FLUX[mask,i,j]
                    Y = (j  + (Y - np.min(Y)) / (np.max(Y) - np.min(Y)))

                    self.pixel_lines[count].setData(x=X,y=Y, pen=pg.mkPen('red', width=3))
                    count +=1
        else :             
            for i in range(len(self.pixel_lines)) : self.pixel_lines[i].clear()

        self.image_view_plotview.update()

    def plot_pixel_aperture(self,):
        if self.pixel_mask_check_box.isChecked() : 
            count = 0
            for i in range(self.FLUX.shape[1]):
                for j in range(self.FLUX.shape[2]):
                    self.pixel_lines[count].clear()
                    if self.pixel_aperture_mask[i,j] > 0:
                        X = [i,i+ 1, i+1, i+0, i]
                        Y = [j,j,j+1,j+1, j]
                        if self.pixel_aperture_mask[i,j]==1 : self.pixel_lines[count].setData(x=X,y=Y, pen=pg.mkPen('b', width=5))
                        elif self.pixel_aperture_mask[i,j]==2 : self.pixel_lines[count].setData(x=X,y=Y, pen=pg.mkPen('red', width=5))

                        count +=1
        else :             
            for i in range(len(self.pixel_lines)) : self.pixel_lines[i].clear()

        self.image_view_plotview.update()


    def update_sources(self,):
            # No sources cached, lets get them
            self.sources = Catalogs.query_object('TIC{:}'.format(self.ticid), radius=0.00583333*np.max([self.FLUX.shape[1], self.FLUX.shape[2]]), catalog="TIC")
            self.sources_X, self.sources_Y = self.wcs.all_world2pix(self.sources['ra'], self.sources['dec'],1)

            mask = (self.sources_Y > 0) & (self.sources_Y < self.FLUX.shape[2]) & (self.sources_X > 0) & (self.sources_X < self.FLUX.shape[1]) & (self.sources['Tmag'] < float(self.tmag_max_cache)) & (self.sources['Tmag'] > float(self.tmag_min_cache))
            self.sources = self.sources[mask]
            self.sources_X, self.sources_Y = self.wcs.all_world2pix(self.sources['ra'], self.sources['dec'],1)

    def plot_sources(self,):
        if (not hasattr(self, 'sources')) or (self.tmag_max_cache!=self.tmag_max_box.text()) or (self.tmag_min_cache!=self.tmag_min_box.text()):
            self.tmag_max_cache = self.tmag_max_box.text()
            self.tmag_min_cache = self.tmag_min_box.text()

            # No sources cached, lets get them
            self.update_sources()

        if self.plot_sources_check_box.isChecked(): 
            self.source_scatter.setData(x = self.sources_Y, y=self.sources_X)
            self.sources_text = []
            for i in range(len(self.sources)):
                self.sources_text.append(pg.TextItem('TIC-{:}\nT={:.3f}'.format(self.sources['ID'][i], self.sources['Tmag'][i]), color=(0, 255, 0)))
                self.sources_text[-1].setPos(self.sources_Y[i]-0.5, self.sources_X[i]-1)
                self.image_view_plotview.addItem(self.sources_text[-1])
        else : 
            self.source_scatter.clear()
            for text in self.sources_text : text.setText('')
            self.sources_text = []


        
        self.image_view_plotview.update()




def main():

    # Parse the args
    args = parse_args()

    # Get the filename
    if len(args.coordinates) ==0 : args.coordinates = None
    else :                         args.coordinates = SkyCoord(*args.coordinates, unit="deg")
    if args.tic_id is not None : args.objectname = 'TIC {:}'.format(args.tic_id)
    else :                       args.objectname = None

    # Now query tic8 
    tic8 = Catalogs.query_object('TIC{:}'.format(args.tic_id), radius=.02, catalog="TIC")
    if len(tic8)==0 : raise ValueError('No obsject found :( ')
    tic8 = tic8[0]


    if args.sector is None:

        with urllib.request.urlopen('https://mast.stsci.edu/tesscut/api/v0.1/sector?ra={:}&dec={:}'.format(tic8['ra'], tic8['dec'])) as url:
            data = json.load(url)['results']
            if len(data)==0 : 
                print('No FFI data.')
                data_ffi=Table()
            else:
                keys = list(data[0].keys())
                data = [[i[j] for j in keys] for i in data]
                data_ffi = Table(np.array(data),names=keys)
  
        # Commented out as TIC not available now
        # print('https://mast.stsci.edu/tesscut/api/v0.1/sector?ra={:}&dec={:}&product=TICA'.format(float(tic8['ra']), float(tic8['dec']))) 
        # with urllib.request.urlopen('https://mast.stsci.edu/tesscut/api/v0.1/sector?ra={:}&dec={:}&product=TICA'.format(float(tic8['ra']), float(tic8['dec']))) as url:
        #     data = json.load(url)['results']
        #     if len(data)==0 : 
        #         print('No TICA data.')
        #         data_tica=Table()
        #     else:
        #         keys = list(data[0].keys())
        #         data = [[i[j] for j in keys] for i in data]
        #         data_tica = Table(np.array(data),names=keys)

        with urllib.request.urlopen('https://mast.stsci.edu/tesscut/api/v0.1/sector?ra={:}&dec={:}&product=SPOC'.format(tic8['ra'], tic8['dec'])) as url:
            data = json.load(url)['results']
            if len(data)==0 : 
                print('No SPOC data.')
                data_spoc=Table()
            else:
                keys = list(data[0].keys())
                data = [[i[j] for j in keys] for i in data]
                data_spoc = Table(np.array(data),names=keys)

        
        #if  (len(data_ffi)==0) and (len(data_spoc)==0)  and (len(data_tica)==0):
        if  (len(data_ffi)==0) and (len(data_spoc)==0):
            print('No data')
            exit()

        from astropy.table import vstack
        #total = vstack((data_ffi, data_spoc, data_tica)).group_by('sector')
        total = vstack((data_ffi, data_spoc)).group_by('sector')

        print(total)
        unique_sectors = total.groups.keys
        unique_camera = Column(np.zeros(len(unique_sectors), dtype=int), name='Camera')
        unique_ccd = Column(np.zeros(len(unique_sectors), dtype=int), name='CCD')
        unique_SPOC = Column(np.zeros(len(unique_sectors), dtype='|S1'), name='SPOC FFI')
        unique_TICA = Column(np.zeros(len(unique_sectors), dtype='|S1'), name='TICA FFI')
        unique_QLP_LC = Column(np.zeros(len(unique_sectors), dtype='|S1'), name='QLP LC')
        unique_TESS_SPOC_LC = Column(np.zeros(len(unique_sectors), dtype='|S1'), name='TESS-SPOC LC')
        unique_SPOC_LC = Column(np.zeros(len(unique_sectors), dtype='|S1'), name='SPOC LC')
        unique_SPOC_DVR = Column(np.zeros(len(unique_sectors), dtype='|S1'), name='SPOC LC DVR')

        for i in range(len(total.groups)):
            group = total.groups[i]
            unique_camera[i] = group['camera'][0]
            unique_ccd[i] = group['ccd'][0]
            for j in range(len(group)):
                if group['sectorName'][j][:4]=='tess' : unique_SPOC[i] = 'X'
                if group['sectorName'][j][:4]=='tica' : unique_TICA[i] = 'X'

        t = Table(masked=True)
        t.add_column(Column(np.array(unique_sectors).astype(int), name='Sector'))
        t.add_column(unique_camera)
        t.add_column(unique_ccd)
        t.add_column(unique_SPOC)
        t.add_column(unique_TICA)
        t.add_column(unique_QLP_LC)
        t.add_column(unique_TESS_SPOC_LC)
        t.add_column(unique_SPOC_LC)
        t.add_column(unique_SPOC_DVR)


        from astroquery.mast import Observations
        print('Qurying {:}'.format(args.tic_id))
        obsTable = Observations.query_criteria(provenance_name=('TESS-SPOC','SPOC', 'QLP'), target_name=args.tic_id)
        try: 
            data = Observations.get_product_list(obsTable)
            data_mask = np.zeros(len(data), dtype = bool)
            test = np.array([i[-8:]=='_lc.fits' for i in data['productFilename']], dtype = bool)
            data_mask = data_mask | test
            test = np.array([i[-9:]=='_llc.fits' for i in data['productFilename']], dtype = bool)
            data_mask = data_mask | test
            test = np.array([i[-8:]=='_tp.fits' for i in data['productFilename']], dtype = bool)
            data_mask = data_mask | test
            test = np.array([i[-4:]=='.pdf' for i in data['productFilename']], dtype = bool)
            data_mask = data_mask | test
            data = data[data_mask]
        except : data = Table()

        for i in range(len(t)):
            for j in range(len(data)):
                if (data['project'][j]=='QLP') and ('s{:04}'.format(t['Sector'][i]) in data['productFilename'][j]):
                    t['QLP LC'][i] = 'X'
                if (data['project'][j]=='TESS-SPOC') and ('s{:04}'.format(t['Sector'][i]) in data['productFilename'][j]):
                    t['TESS-SPOC LC'][i] = 'X'
                if (data['project'][j]=='SPOC') and ('s{:04}'.format(t['Sector'][i]) in data['productFilename'][j]):
                    t['SPOC LC'][i] = 'X'
                if ( (data['project'][j]=='TESS-SPOC') or (data['project'][j]=='SPOC') ) and (data['productSubGroupDescription'][j] in ['DVR', 'DVM', 'DVS']) and ('s{:04}'.format(t['Sector'][i]) in data['productFilename'][j]):
                    t['SPOC LC DVR'][i] = 'X'



        t.mask['SPOC FFI'] = t['SPOC FFI']!='X'
        t.mask['TICA FFI'] = t['TICA FFI']!='X'
        t.mask['QLP LC'] = t['QLP LC']!='X'
        t.mask['TESS-SPOC LC'] = t['TESS-SPOC LC']!='X'
        t.mask['SPOC LC'] = t['SPOC LC']!='X'
        t.mask['SPOC LC DVR'] = t['SPOC LC DVR']!='X'


        t.pprint(max_lines=1000)


    if args.all:
        print('Listing best data for each sector')
        print('1. SPOC LC (short cadence)')
        print('2. TESS SPOC LC (FFI)')
        print('3. QLP LC')
        print('4. SPOC FFIs   [our own LC]')
        print('5. TIC FFIs    [our own LC]')

        data_files, data_origin = [], []
        for i in range(len(t)):
            check=0
            if t['SPOC LC'][i]=='X' : 
                # Download the SPOC 2-min data
                for j in range(len(data)):
                    if (data['project'][j]=='SPOC') and ('s{:04}'.format(t['Sector'][i]) in data['productFilename'][j]) and (data['productFilename'][j][-8:]=='_lc.fits'):
                        out = Observations.download_products(data[j])
                        if out['Status'][0]=='COMPLETE':
                            os.system('cp {:} .'.format(out['Local Path'][0]))
                            data_files.append(out['Local Path'][0].split('/')[-1])
                            data_origin.append('SPOC short cadence LC')
                        check=1
                        break
            if check : continue

            if t['TESS-SPOC LC'][i]=='X' : 
                # Download the SPOC 2-min data
                for j in range(len(data)):
                    if (data['project'][j]=='TESS-SPOC') and ('s{:04}'.format(t['Sector'][i]) in data['productFilename'][j]) and (data['productFilename'][j][-8:]=='_lc.fits'):
                        out = Observations.download_products(data[j])
                        if out['Status'][0]=='COMPLETE':
                            os.system('cp {:} .'.format(out['Local Path'][0]))
                            data_files.append(out['Local Path'][0].split('/')[-1])
                            data_origin.append('SPOC FFI LC')

                        check=1
                        break
            if check : continue


            print(t['QLP LC'][i])
            if t['QLP LC'][i]=='X' : 
                # Download the SPOC 2-min data
                for j in range(len(data)):
                    print(data['productFilename'][j])
                    if (data['project'][j]=='QLP') and ('s{:04}'.format(t['Sector'][i]) in data['productFilename'][j]) and (data['productFilename'][j][-9:]=='_llc.fits'):
                        out = Observations.download_products(data[j])
                        print(out['Status'][0])
                        if out['Status'][0]=='COMPLETE':
                            os.system('cp {:} .'.format(out['Local Path'][0]))
                            data_files.append(out['Local Path'][0].split('/')[-1])
                            data_origin.append('QLP LC')
                        check=1
                        break
            if check : continue   



            if t['SPOC FFI'][i]=='X' : 
                # Download the SPOC 2-min data
                with tempfile.TemporaryDirectory() as tmpdirname:
                    cmd = 'wget -O {:}/test.zip "https://mast.stsci.edu/tesscut/api/v0.1/astrocut?ra={:}&dec={:}&y=10&x=10&sector={:}"'.format(tmpdirname, tic8['ra'], tic8['dec'], t['Sector'][i])
                    os.system(cmd)
                    #if os.stat('{:}/test.zip'.format(tmpdirname)).st_size==0 : 
                    #    exit()
                    try:
                        with zipfile.ZipFile('{:}/test.zip'.format(tmpdirname), 'r') as zip_ref : zip_ref.extractall(tmpdirname)
                        os.system('rm {:}/test.zip'.format(tmpdirname))
                        filename = glob.glob('{:}/*.fits'.format(tmpdirname))[0]
                        os.system('cp {:} .'.format(filename))
                        filename = filename.split('/')[-1]
                        hdu = fits.open(filename)

                        TIME = np.array(hdu[1].data['TIME'], dtype=np.float64)
                        FLUX = np.array(hdu[1].data['FLUX'], dtype=np.float64)
                        QUALITY_MASK = hdu[1].data['QUALITY'] ==0

                        TIME[np.isnan(TIME)]==10
                        FLUX[np.isnan(FLUX)]==10
                        TIME[np.isinf(TIME)]==10
                        FLUX[np.isinf(FLUX)]==10
                        TIME = TIME[QUALITY_MASK]
                        FLUX = FLUX[QUALITY_MASK]
                        FLUX = FLUX[~np.isnan(TIME)]
                        TIME = TIME[~np.isnan(TIME)]
                        QUALITY_MASK = QUALITY_MASK[QUALITY_MASK]
                        wcs = WCS(hdu[2].header, hdu)

                        app = Window(TIME,FLUX,QUALITY_MASK, ticid=args.tic_id, wcs=wcs, sector=t['Sector'][i], mask = args.mask, filename=filename, type='TICA' if args.tica else 'SPOC')
                        app.draw_data()
                        fname = app.fit_lightcurve(call=False)

                        data_files.append(fname)
                        data_origin.append('FFI custom LC')

                        check=1
                    except : pass
            if check : continue


            if t['TICA FFI'][i]=='X' : 
                # Download the SPOC 2-min data
                with tempfile.TemporaryDirectory() as tmpdirname:
                    cmd = 'wget -O {:}/test.zip "https://mast.stsci.edu/tesscut/api/v0.1/astrocut?ra={:}&dec={:}&y=10&x=10&product=TICA&sector={:}"'.format(tmpdirname, tic8['ra'], tic8['dec'], t['Sector'][i])
                    os.system(cmd)
                    #if os.stat('{:}/test.zip'.format(tmpdirname)).st_size==0 : 
                    #    exit()
                    with zipfile.ZipFile('{:}/test.zip'.format(tmpdirname), 'r') as zip_ref : zip_ref.extractall(tmpdirname)
                    os.system('rm {:}/test.zip'.format(tmpdirname))
                    filename = glob.glob('{:}/*.fits'.format(tmpdirname))[0]
                    os.system('cp {:} .'.format(filename))
                    filename = filename.split('/')[-1]
                    hdu = fits.open(filename)

                    TIME = np.array(hdu[1].data['TIME'], dtype=np.float64)
                    FLUX = np.array(hdu[1].data['FLUX'], dtype=np.float64)
                    QUALITY_MASK = hdu[1].data['QUALITY'] ==0

                    TIME[np.isnan(TIME)]==10
                    FLUX[np.isnan(FLUX)]==10
                    TIME[np.isinf(TIME)]==10
                    FLUX[np.isinf(FLUX)]==10
                    TIME = TIME[QUALITY_MASK]
                    FLUX = FLUX[QUALITY_MASK]
                    FLUX = FLUX[~np.isnan(TIME)]
                    TIME = TIME[~np.isnan(TIME)]
                    QUALITY_MASK = QUALITY_MASK[QUALITY_MASK]
                    wcs = WCS(hdu[2].header, hdu)

                    app = Window(TIME,FLUX,QUALITY_MASK, ticid=args.tic_id, wcs=wcs, sector=t['Sector'][i], mask = args.mask, filename=filename, type='TICA' if args.tica else 'SPOC')
                    app.draw_data()
                    fname = app.fit_lightcurve(call=False)

                    data_files.append(fname)
                    data_origin.append('TICA custom LC')
                    check=1
            if check : continue


            print('Sector {:} has no data'.format(t['Sector'][i]))
            data_files.append('')
            data_origin.append('Missing')
        
        print('Summary')
        for i in range(len(data_files)):
            print('Sector {:>5} : {:}'.format(t['Sector'][i], data_origin[i]))
        
        if not args.nogui: os.system('spocfit {:}'.format(' '.join(data_files)))
        else:
            from astropy.table import Table
            import bruce2
            class data:
                def __init__(self,filenames):
                    self.SG_window_length = 303
                    self.gradsplit = 0.1
                    self.SG_window_text = 2
                    self.load_data(filenames=filenames)

                def load_data(self, filenames):
                    filenames = np.atleast_1d(filenames)
                    print('Loading data from {:}'.format(','.join(filenames)))
                    self.data = None
                    self.sector_info = []
                    for i in range(len(filenames)):
                        data_ = Table.read(filenames[i])
                        if self.data is None : self.data = data_
                        else : self.data = vstack((self.data, data_))

                        try : self.sector_info.append({'sector' : fits.open(filenames[i])[0].header['SECTOR'], 'tmin' : np.min(data_['TIME']), 'tmax' : np.max(data_['TIME']) , 'sector_max' :  np.max(data_['PDCSAP_FLUX'])})
                        except : self.sector_info.append({'sector' : -1, 'tmin' : np.min(data_['TIME']), 'tmax' : np.max(data_['TIME']) , 'sector_max' :  np.max(data_['PDCSAP_FLUX'])})
                    
                    # Now filter the data
                    self.data = self.data[~np.isinf(self.data['TIME']) & ~np.isnan(self.data['TIME']) & ~np.isinf(self.data['SAP_FLUX']) & ~np.isnan(self.data['SAP_FLUX']) & (self.data['SAP_FLUX'] > 1) & (self.data['PDCSAP_FLUX'] > 1)]

                    # Now get the segments
                    self.segments = bruce2.data.find_nights_from_data(self.data['TIME'], float(self.gradsplit))

                    # now normalise
                    if 'PDC_SAP_NORM' not in self.data.colnames:
                        self.normalise_data()
                    else:
                        self.MAD = np.mean([median_abs_deviation(self.data['PDCSAP_FLUX'][self.segments[i]]/self.data['PDC_SAP_NORM'][self.segments[i]]) for i in range(len(self.segments))])
                        self.MFthreshold = 3*self.MAD
                        
                def normalise_data(self,):
                    # Check and add the normalisation column
                    if 'PDC_SAP_NORM' not in self.data.colnames : self.data.add_column(Column(np.zeros(len(self.data)), name='PDC_SAP_NORM'))

                    # Check SG_window_length
                    if not hasattr(self, 'SG_window_length') : self.SG_window_length = 101

                    # Now check it is odd 
                    if not self.SG_window_length&1 : self.SG_window_length +=1

                    # Now normalise
                    for i in range(len(self.segments)):
                        window_scale = int(float(self.SG_window_text) / np.median(np.gradient(self.data['TIME'][self.segments[i]])) )
                        if window_scale < 10 : window_scale = 10
                        if not window_scale&1 : window_scale +=1

                        try : self.data['PDC_SAP_NORM'][self.segments[i]] = bruce2.data.flatten_data_with_function(self.data['TIME'][self.segments[i]], self.data['PDCSAP_FLUX'][self.segments[i]], SG_window_length=window_scale, SG_iter=int(self.SG_iter_text.text()))
                        except : self.data['PDC_SAP_NORM'][self.segments[i]] = bruce2.data.flatten_data_with_function(self.data['TIME'][self.segments[i]], self.data['PDCSAP_FLUX'][self.segments[i]], method='poly1d')
                    # Now re-calculate MAD
                    self.MAD = np.mean([median_abs_deviation(self.data['PDCSAP_FLUX'][self.segments[i]]/self.data['PDC_SAP_NORM'][self.segments[i]]) for i in range(len(self.segments))])
                    self.MFthreshold = 3*self.MAD

            a = data(data_files)
            np.savetxt('TIC-{:}_TESS_DATA.txt'.format(args.tic_id), np.array([a.data['TIME'], a.data['PDCSAP_FLUX']/a.data['PDC_SAP_NORM'], a.data['PDCSAP_FLUX_ERR']]).T)
            print(a.data)








    if args.sector is None : 
        exit()

    
    if args.tica:
        with tempfile.TemporaryDirectory() as tmpdirname:
            cmd = 'wget -O {:}/test.zip "https://mast.stsci.edu/tesscut/api/v0.1/astrocut?ra={:}&dec={:}&y={:}&x={:}&product=TICA&sector={:}"'.format(tmpdirname, tic8['ra'], tic8['dec'], args.size, args.size, args.sector)
            os.system(cmd)
            if os.stat('{:}/test.zip'.format(tmpdirname)).st_size==0 : 
                print('Downloaded file is empty - was it observed?')
                exit()
            with zipfile.ZipFile('{:}/test.zip'.format(tmpdirname), 'r') as zip_ref : zip_ref.extractall(tmpdirname)
            os.system('rm {:}/test.zip'.format(tmpdirname))
            filename = glob.glob('{:}/*.fits'.format(tmpdirname))[0]
            os.system('cp {:} .'.format(filename))
            filename = filename.split('/')[-1]
            hdu = fits.open(filename)

    else:
        with tempfile.TemporaryDirectory() as tmpdirname:
            cmd = 'wget -O {:}/test.zip "https://mast.stsci.edu/tesscut/api/v0.1/astrocut?ra={:}&dec={:}&y={:}&x={:}&sector={:}"'.format(tmpdirname, tic8['ra'], tic8['dec'],args.size, args.size, args.sector)
            os.system(cmd)
            if os.stat('{:}/test.zip'.format(tmpdirname)).st_size==0 : 
                print('Downloaded file is empty - was it observed?')
                exit()
            with zipfile.ZipFile('{:}/test.zip'.format(tmpdirname), 'r') as zip_ref : zip_ref.extractall(tmpdirname)
            os.system('rm {:}/test.zip'.format(tmpdirname))
            filename = glob.glob('{:}/*.fits'.format(tmpdirname))[0]
            os.system('cp {:} .'.format(filename))
            filename = filename.split('/')[-1]
            hdu = fits.open(filename)

    TIME = np.array(hdu[1].data['TIME'], dtype=np.float64)
    FLUX = np.array(hdu[1].data['FLUX'], dtype=np.float64)
    QUALITY_MASK = hdu[1].data['QUALITY'] ==0

    TIME[np.isnan(TIME)]==10
    FLUX[np.isnan(FLUX)]==10
    TIME[np.isinf(TIME)]==10
    FLUX[np.isinf(FLUX)]==10
    TIME = TIME[QUALITY_MASK]
    FLUX = FLUX[QUALITY_MASK]
    FLUX = FLUX[~np.isnan(TIME)]
    TIME = TIME[~np.isnan(TIME)]
    QUALITY_MASK = QUALITY_MASK[QUALITY_MASK]
    wcs = WCS(hdu[2].header, hdu)

    app = Window(TIME,FLUX,QUALITY_MASK, ticid=args.tic_id, wcs=wcs, sector=args.sector, mask = args.mask, filename=filename, type='TICA' if args.tica else 'SPOC')
    app.draw_data()

    pg.exec()
    exit()





if __name__ == '__main__':
    main()