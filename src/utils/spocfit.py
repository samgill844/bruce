def main():


    import multiprocessing, multiprocess,  emcee, corner
    from opcode import hasconst
    from statistics import median
    from tkinter import E
    from pyqtgraph.Qt import QtGui, QtCore
    import numpy as np, sys, os, pyqtgraph as pg, bruce
    from astropy.table import Table, vstack, Column
    from scipy.stats import median_abs_deviation
    import pyqtgraph as pg
    from astropy.io import fits
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QSizePolicy, QSlider, QSpacerItem, \
        QVBoxLayout, QWidget, QLineEdit , QCheckBox, QProgressBar, QPlainTextEdit, QRadioButton
    # Enable High DPI scaling for better display on high-resolution screens
    try:
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    except Exception as e:
        print('Warning enabling High DPI attributes:', e)

    from pyqtgraph.dockarea import DockArea, Dock
    import pyqtgraph.exporters
    import tempfile , glob, time, pickle
    from astroquery.mast import Catalogs
    from astropy import constants
    from scipy.optimize import differential_evolution
    from scipy.stats import median_abs_deviation
    import warnings
    warnings.filterwarnings("ignore")


    def fit_lc_lnlike(theta, theta_names, bounds, time, flux, flux_err, ldc_1, ldc_2, ld_law, all_pars, all_pars_names):
        for i in range(len(theta)):
            if (theta[i] < bounds[i][0]) or (theta[i] > bounds[i][1]) : return -np.inf
        # Now check impack parameter
        if (theta[np.argwhere(theta_names=='b')[0][0]] if 'b' in theta_names else all_pars[np.argwhere(all_pars_names=='b')[0][0]])  > (1 + theta[np.argwhere(theta_names=='k')[0][0]] if 'k' in theta_names else all_pars[np.argwhere(all_pars_names=='k')[0][0]]) : return -np.inf

        # Now get the model
        incl = np.arccos(  (theta[np.argwhere(theta_names=='b')[0][0]] if 'b' in theta_names else all_pars[np.argwhere(all_pars_names=='b')[0][0]])       *  (theta[np.argwhere(theta_names=='radius_1')[0][0]] if 'radius_1' in theta_names else all_pars[np.argwhere(all_pars_names=='radius_1')[0][0]])  )

        model = (theta[np.argwhere(theta_names=='zp')[0][0]] if 'zp' in theta_names else all_pars[np.argwhere(all_pars_names=='zp')[0][0]])*bruce.binarystar.lc(np.array(time, dtype = np.float64),
                        t_zero=theta[np.argwhere(theta_names=='t_zero')[0][0]] if 't_zero' in theta_names else all_pars[np.argwhere(all_pars_names=='t_zero')[0][0]],
                        period=theta[np.argwhere(theta_names=='period')[0][0]] if 'period' in theta_names else all_pars[np.argwhere(all_pars_names=='period')[0][0]],
                        radius_1=theta[np.argwhere(theta_names=='radius_1')[0][0]] if 'radius_1' in theta_names else all_pars[np.argwhere(all_pars_names=='radius_1')[0][0]],
                        k=theta[np.argwhere(theta_names=='k')[0][0]] if 'k' in theta_names else all_pars[np.argwhere(all_pars_names=='k')[0][0]],
                        incl=incl,
                        c=ldc_1,
                        alpha=ldc_2,
                        ld_law=ld_law,
                        cadence = 0.00694444)

        # Now get the loglikeliehood
        return bruce.sampler.loglike(flux, flux_err, model, jitter=0., offset=False)


    class emcee_plot():
        def __init__(self, ndim, nsteps):
            self.app = pg.mkQApp("Gradiant Layout Example")
            self.view = pg.GraphicsView()
            self.l = pg.GraphicsLayout(border=(100,100,100))
            self.view.setCentralItem(self.l)
            self.view.show()
            self.view.setWindowTitle('pyqtgraph example: GraphicsLayout')
            self.view.resize(800,600)


    class Window():
        def __init__(self, title = None):
            # Set the defaults
            self.gradsplit = 0.1
            self.SG_iter = 5
            self.lc_model_data = None
            self.ld_law = 2

            # Create the GUI
            self.app = pg.mkQApp("SPOCFIT") 
            self.win = QtGui.QMainWindow()
            # Enable dock nesting and set modern Fusion style + theme toggle
            try:
                self.win.setDockNestingEnabled(True)
            except Exception:
                pass

            # Theme support: apply Fusion style and define light/dark palettes
            from PyQt5.QtWidgets import QAction
            def apply_light_theme():
                QApplication.setStyle('Fusion')
                pal = QApplication.palette()
                # light base adjustments
                pal.setColor(pal.Window, Qt.white)
                pal.setColor(pal.WindowText, Qt.black)
                pal.setColor(pal.Base, Qt.white)
                pal.setColor(pal.AlternateBase, Qt.lightGray)
                pal.setColor(pal.ToolTipBase, Qt.black)
                pal.setColor(pal.ToolTipText, Qt.white)
                pal.setColor(pal.Text, Qt.black)
                pal.setColor(pal.Button, Qt.white)
                pal.setColor(pal.ButtonText, Qt.black)
                QApplication.setPalette(pal)
                self._theme = 'light'

            def apply_dark_theme():
                QApplication.setStyle('Fusion')
                pal = QApplication.palette()
                # dark base adjustments
                pal.setColor(pal.Window, Qt.black)
                pal.setColor(pal.WindowText, Qt.white)
                pal.setColor(pal.Base, Qt.black)
                pal.setColor(pal.AlternateBase, Qt.darkGray)
                pal.setColor(pal.ToolTipBase, Qt.white)
                pal.setColor(pal.ToolTipText, Qt.white)
                pal.setColor(pal.Text, Qt.white)
                pal.setColor(pal.Button, Qt.darkGray)
                pal.setColor(pal.ButtonText, Qt.white)
                QApplication.setPalette(pal)
                self._theme = 'dark'

            # Attach the theme functions to self so other methods can call them
            self.apply_light_theme = apply_light_theme
            self.apply_dark_theme = apply_dark_theme
            # Default to light theme
            try:
                self.apply_light_theme()
            except Exception as e:
                print('Warning applying theme:', e)

            # Add a View menu with theme toggle
            menubar = self.win.menuBar() if hasattr(self.win, 'menuBar') else None
            if menubar is not None:
                viewMenu = menubar.addMenu('View')
                themeAction = QAction('Toggle Theme', self.win)
                def _toggle():
                    try:
                        if getattr(self, '_theme', 'light') == 'light':
                            self.apply_dark_theme()
                        else:
                            self.apply_light_theme()
                    except Exception as e:
                        print('Theme toggle error:', e)
                themeAction.triggered.connect(_toggle)
                viewMenu.addAction(themeAction)


            self.app.setStyleSheet("QLabel{font-size: 18pt;}")
            #################################################################
            # Create the window and the dock area
            #################################################################
            self.area = DockArea()
            self.win.setCentralWidget(self.area)
            #self.win.resize(1300,800)
            #self.win.showFullScreen()
            self.win.showMaximized()

            title_text = 'SPOCFIT'
            if title is not None: title_text += ' {:}'.format(title)
            self.win.setWindowTitle(title_text)

            #################################################################
            # Create the first dock which holds the plots in
            #################################################################
            self.d1 = Dock(title, size=(1000,800))     ## give this dock the minimum possible size
            self.d_segment_split = Dock("Segment\nsplit", size=(300,8))
            self.d_normalisation = Dock("Normalisation\nParameters", size=(300,50))
            self.d_write_data = Dock("Write\nParameters", size=(300,50))
            self.d_save_image = Dock("Save\nPlot", size=(300,50))
            self.d_fitted = Dock("Lightcurve\nParameters", size=(300,8))
            self.d_stellar = Dock("Stellar\nParameters", size=(300,100))
            self.d_fitting= Dock("Fitting", size=(300,100))

            #################################################################
            # Add the docks
            #################################################################
            self.area.addDock(self.d1)      ## place d1 at left edge of dock area (it will fill the whole space since there are no other docks yet)
            self.area.addDock(self.d_segment_split, 'right', self.d1 )     ## place d2 at right edge of dock area
            self.area.addDock(self.d_normalisation, 'bottom', self.d_segment_split)     ## place d2 at right edge of dock area
            self.area.addDock(self.d_write_data, 'bottom', self.d_normalisation)     ## place d2 at right edge of dock area
            self.area.addDock(self.d_save_image, 'bottom', self.d_write_data)     ## place d2 at right edge of dock area
            self.area.addDock(self.d_fitted, 'bottom', self.d_save_image)     ## place d2 at right edge of dock area
            self.area.addDock(self.d_stellar, 'bottom', self.d_fitted)     ## place d2 at right edge of dock area
            self.area.addDock(self.d_fitting, 'bottom', self.d_stellar)     ## place d2 at right edge of dock area

            #################################################################
            # Create the view and plots for the left size
            #################################################################
            self.view = pg.GraphicsLayoutWidget()
            self.w1 = self.view.addPlot()
            self.view.nextRow()
            self.w2 = self.view.addPlot()
            self.view.nextRow()
            self.w3 = self.view.addPlot()
            self.w1.setXLink(self.w2)
            self.w2.setXLink(self.w3)  

            # showing x and y grids
            self.w1.showGrid(x = True, y = True)
            self.w2.showGrid(x = True, y = True)

            # set properties of the label for y axis
            self.w1.setLabel('left', 'PDCSAP FLUX', units ='e/s')
            self.w2.setLabel('left', 'NORMALISED FLUX')
            self.w3.setLabel('left', 'SAP BKG FLUX', units ='e/s')

            # set properties of the label for x axis
            self.w1.setLabel('bottom', 'BTJD [JD - 2457000]')
            self.w2.setLabel('bottom', 'BTJD [JD - 2457000]')
            self.w3.setLabel('bottom', 'BTJD [JD - 2457000]')

            self.w1.addLegend(frame=False, colCount=2)
            self.w2.addLegend(frame=False, colCount=2)

            self.normed_roi = pg.ROI([-8, 14], [6, 5], parent=self.w2, pen=pg.mkPen('y'))
            self.normed_roi.addScaleHandle([0.5, 0], [0.5, 1])
            self.normed_roi.addScaleHandle([0.5, 1], [0.5, 0])
            self.normed_roi.addScaleHandle([0, 0.5], [1, 0.5])
            self.normed_roi.addScaleHandle([1, 0.5], [0., 0.5])
            self.normed_roi.addScaleHandle([0,0], [1,1])
            self.normed_roi.addScaleHandle([0,1], [1,0])
            self.normed_roi.addScaleHandle([1,0], [0,1])
            self.normed_roi.addScaleHandle([1,1], [0,0])
            self.normed_roi.setVisible(False)

            # now link it to update the text boxes 
            self.normed_roi.sigRegionChanged.connect(self.roiChangedEvent) 
            self.normed_roi.setSize([0.5,0.002])
            self.w2.addItem(self.normed_roi)

            # Finally ad view
            self.d1.addWidget(self.view)



            #################################################################
            # Now lets do the buttons, we have to use a Layout widget
            #################################################################
            self.button_view = pg.LayoutWidget()

            # Now lets do the gradsplit row
            self.button_view.addWidget(QLabel(text='Enter days\nto split'), row=0, col=0)

            self.gradsplitText = QLineEdit(str(self.gradsplit))
            self.button_view.addWidget(self.gradsplitText, row=0, col=1)

            self.gradsplitBtn = QtGui.QPushButton('GO')
            self.button_view.addWidget(self.gradsplitBtn, row=0, col=2)
            self.gradsplitBtn.clicked.connect(self.resegment_data)

            # Finally add the layout widget to d_segment_split
            self.d_segment_split.addWidget(self.button_view)


            #################################################################
            # Now we add the Layout widget for fitted parameters
            # Now lets do the buttons, we have to use a Layout widget
            #################################################################
            self.fitted_parameters_view = pg.LayoutWidget()

            # First, lets do the iterations
            self.fitted_parameters_view.addWidget(QLabel(text='SG iterations'), row=0, col=0)
            self.SG_iter_text = QLineEdit('5')
            self.fitted_parameters_view.addWidget(self.SG_iter_text, row=0, col=1)

            # Now lets do the scale 
            self.fitted_parameters_view.addWidget(QLabel(text='SG filter size [days]'), row=1, col=0)
            self.SG_window_text = QLineEdit('2')
            self.fitted_parameters_view.addWidget(self.SG_window_text, row=1, col=1)

            # Finally, ad the go button
            self.normBtn = QtGui.QPushButton('GO')
            self.fitted_parameters_view.addWidget(self.normBtn, row=1, col=2)
            self.normBtn.clicked.connect(self.renormalise_data)

            # Finally, add the view
            self.d_normalisation.addWidget(self.fitted_parameters_view)





            #################################################################
            # Next section is to write the data
            #################################################################
            self.write_data_view = pg.LayoutWidget()
            self.write_data_view.addWidget(QLabel(text='Data prefix'), row=0, col=0)

            # Now get the data prefix
            self.write_data_prefix = QLineEdit('{:}_SPOC'.format(title))
            self.write_data_view.addWidget(self.write_data_prefix, row=0, col=1,  rowspan=1, colspan=2)

            # Add the fits file writeout
            self.write_fits_Btn = QtGui.QPushButton('FITS')
            self.write_data_view.addWidget(self.write_fits_Btn, row=1, col=0)
            self.write_fits_Btn.clicked.connect(self.write_fits_data)

            # Add the flux file writeout
            self.write_flux_Btn = QtGui.QPushButton('FLUX')
            self.write_data_view.addWidget(self.write_flux_Btn, row=1, col=1)
            self.write_flux_Btn.clicked.connect(self.write_flux_data)

            # Now add the mag write out
            self.write_mag_Btn = QtGui.QPushButton('MAG')
            self.write_data_view.addWidget(self.write_mag_Btn, row=1, col=2)
            self.write_mag_Btn.clicked.connect(self.write_mag_data)

            # Finally, add the view
            self.d_write_data.addWidget(self.write_data_view)
    

            #################################################################
            # First, create the saveplotview
            #################################################################
            self.save_plot_view = pg.LayoutWidget()

            # set the prefix
            self.save_plot_view.addWidget(QLabel(text='Plot prefix'), row=0, col=0)
            self.save_plot_prefix = QLineEdit('{:}_SPOC'.format(title))
            self.save_plot_view.addWidget(self.save_plot_prefix, row=0, col=1)
    
            # Finally, ad the go button
            self.save_plot_Btn = QtGui.QPushButton('SAVE PLOT')
            self.save_plot_view.addWidget(self.save_plot_Btn, row=1, col=0,colspan=2)
            self.save_plot_Btn.clicked.connect(self.save_plot)

            # Finally, add the view
            self.d_save_image.addWidget(self.save_plot_view)




            #################################################################
            # Lightcurve parameters
            #################################################################
            # Now do the lightcurvve parameters
            self.lc_parameters_view = pg.LayoutWidget()

            # Now do Epoch
            self.lc_parameters_view.addWidget(QLabel(text='Epoch\n[BTJD]'), row=0, col=0)
            self.lc_pars_epoch_input = QLineEdit('{:.2f}'.format(np.median(self.data['TIME']) if hasattr(self, 'data') else 1566))
            self.lc_parameters_view.addWidget(self.lc_pars_epoch_input, row=0, col=1)

            # Now do period
            self.lc_parameters_view.addWidget(QLabel(text='Period\n[day]'), row=0, col=2)
            self.lc_pars_period_input = QLineEdit('30.')
            self.lc_parameters_view.addWidget(self.lc_pars_period_input, row=0, col=3)

            # Now do R1/a
            self.lc_parameters_view.addWidget(QLabel(text='R1 / a'), row=1, col=0)
            self.lc_pars_r1a_input = QLineEdit('0.2')
            self.lc_parameters_view.addWidget(self.lc_pars_r1a_input, row=1, col=1)

            # Now do k
            self.lc_parameters_view.addWidget(QLabel(text='R2 / R1'), row=1, col=2)
            self.lc_pars_r2r1_input = QLineEdit('0.2')
            self.lc_parameters_view.addWidget(self.lc_pars_r2r1_input, row=1, col=3)

            # Now do b
            self.lc_parameters_view.addWidget(QLabel(text='b'), row=1, col=4)
            self.lc_pars_b_input = QLineEdit('0.')
            self.lc_parameters_view.addWidget(self.lc_pars_b_input, row=1, col=5)

            # Now do h1 and h2
            self.lc_parameters_view.addWidget(QLabel(text='ldc 1'), row=2, col=0)
            self.lc_pars_ldc_1_input = QLineEdit('0.6536')
            self.lc_parameters_view.addWidget(self.lc_pars_ldc_1_input, row=2, col=1)

            self.lc_parameters_view.addWidget(QLabel(text='ldc 2'), row=2, col=2)
            self.lc_pars_ldc_2_input = QLineEdit('0.5739')
            self.lc_parameters_view.addWidget(self.lc_pars_ldc_2_input, row=2, col=3)

            # Now do zp
            self.lc_parameters_view.addWidget(QLabel(text='zp'), row=2, col=4)
            self.lc_pars_zp_input = QLineEdit('1.')
            self.lc_parameters_view.addWidget(self.lc_pars_zp_input, row=2, col=5)

            # Now add a check box to inddicate if it is a single transit
            self.lc_pars_single_check_box = QCheckBox("Single Transit? ")
            self.lc_pars_single_check_box.setChecked(True)
            self.lc_parameters_view.addWidget(self.lc_pars_single_check_box, row=6, col=0)
            self.lc_pars_single_check_box.stateChanged.connect(self.update_single_transit_par)

            self.get_tzero_toggle = QCheckBox("Click t_zero")
            self.get_tzero_toggle.setChecked(True)
            self.lc_parameters_view.addWidget(self.get_tzero_toggle, row=6, col=2)
            self.w1.scene().sigMouseClicked.connect(self.onClick)


            # Finally, ad the go button
            self.generate_lc_Btn = QtGui.QPushButton('GENERATE MODEL')
            self.lc_parameters_view.addWidget(self.generate_lc_Btn, row=7, col=0, colspan=2)
            self.generate_lc_Btn.clicked.connect(self.generate_lc_model)

            # Finally, ad the clear button
            self.clear_lc_Btn = QtGui.QPushButton('CLEAR MODEL')
            self.lc_parameters_view.addWidget(self.clear_lc_Btn, row=7, col=2, colspan=2)
            self.clear_lc_Btn.clicked.connect(self.clear_lc_model)

            # Finally, add the view
            self.d_fitted.addWidget(self.lc_parameters_view)


            #################################################################
            # Next up is stellar parameters
            # Now do the lightcurvve parameters
            #################################################################
            self.lc_stellar_view = pg.LayoutWidget()

            self.lc_stellar_view.addWidget(QLabel(text='TIC ID'), row=0, col=0)
            self.stellar_pars_tic_input = QLineEdit(title.split('-')[1])
            self.lc_stellar_view.addWidget(self.stellar_pars_tic_input, row=0, col=1)

            # self.lc_stellar_view.addWidget(QLabel(text='Tmag'), row=0, col=2)
            # self.stellar_pars_Tmag = QLineEdit()
            # self.lc_stellar_view.addWidget(self.stellar_pars_Tmag, row=0, col=3)
            self.stellar_pars_tic_reload_Btn = QtGui.QPushButton('Reload')
            self.lc_stellar_view.addWidget(self.stellar_pars_tic_reload_Btn, row=0, col=3)
            self.stellar_pars_tic_reload_Btn.clicked.connect(self.reload_tic_data)


            self.stellar_pars_tic_qry_Btn = QtGui.QPushButton('QUERY PARAMETERS')
            self.lc_stellar_view.addWidget(self.stellar_pars_tic_qry_Btn, row=1, col=0, colspan=2)
            self.stellar_pars_tic_qry_Btn.clicked.connect(self.query_tic_params)

            self.stellar_pars_ld_interp_Btn = QtGui.QPushButton('INTERP LD')
            self.lc_stellar_view.addWidget(self.stellar_pars_ld_interp_Btn, row=1, col=2, colspan=2)
            self.stellar_pars_ld_interp_Btn.clicked.connect(self.interp_ld_pars)

            self.lc_stellar_view.addWidget(QLabel(text='R1 [Rsol]'), row=2, col=0)
            self.stellar_pars_R1_input = QLineEdit()
            self.lc_stellar_view.addWidget(self.stellar_pars_R1_input, row=2, col=1)

            self.normed_roi_checkbox = QCheckBox("Inspect transit? ")
            self.lc_stellar_view.addWidget(self.normed_roi_checkbox, row=4, col=2)
            #self.normed_roi_checkbox.stateChanged.connect(self.update_normed_roi)

            self.lc_stellar_view.addWidget(QLabel(text='Width [hr]'), row=5, col=2)
            self.roi_width = QLineEdit()
            self.lc_stellar_view.addWidget(self.roi_width, row=5, col=3)

            self.lc_stellar_view.addWidget(QLabel(text='Depth [ppt]'), row=6, col=2)
            self.roi_depth = QLineEdit()
            self.lc_stellar_view.addWidget(self.roi_depth, row=6, col=3)

            self.lc_stellar_view.addWidget(QLabel(text='M1 [Msol]'), row=3, col=0)
            self.stellar_pars_M1_input = QLineEdit()
            self.lc_stellar_view.addWidget(self.stellar_pars_M1_input, row=3, col=1)

            self.lc_stellar_view.addWidget(QLabel(text='Teff [K]'), row=4, col=0)
            self.stellar_pars_Teff_input = QLineEdit('5777')
            self.lc_stellar_view.addWidget(self.stellar_pars_Teff_input, row=4, col=1)

            self.lc_stellar_view.addWidget(QLabel(text='[Fe/H]'), row=5, col=0)
            self.stellar_pars_FeH_input = QLineEdit('0.0')
            self.lc_stellar_view.addWidget(self.stellar_pars_FeH_input, row=5, col=1)

            self.lc_stellar_view.addWidget(QLabel(text='log g [dex]'), row=6, col=0)
            self.stellar_pars_logg_input = QLineEdit('4.44')
            self.lc_stellar_view.addWidget(self.stellar_pars_logg_input, row=6, col=1)

            # Finally, add the view
            self.d_stellar.addWidget(self.lc_stellar_view)


            #################################################################
            # Now do fitting routines here
            #################################################################
            self.fitting_view = pg.LayoutWidget()

            # First, lets do the iterations
            self.fitting_view.addWidget(QLabel(text='Free parameters:'), row=0, col=0, colspan=2)
            self.fitting_cut_data = QCheckBox("Cut data?")
            self.fitting_cut_data.setChecked(True)
            self.fitting_view.addWidget(self.fitting_cut_data, row=0, col=2, colspan=2)

            self.fitting_t_zero_free = QCheckBox("Epoch")
            self.fitting_t_zero_free.setChecked(True)
            self.fitting_view.addWidget(self.fitting_t_zero_free, row=1, col=0)
            self.fitting_period_free = QCheckBox("Period")
            self.fitting_view.addWidget(self.fitting_period_free, row=1, col=1)
            self.fitting_radius_1_free = QCheckBox("R1/a")
            self.fitting_view.addWidget(self.fitting_radius_1_free, row=1, col=2)
            self.fitting_radius_1_free.setChecked(True)
            self.fitting_k_free = QCheckBox("R2/R1")
            self.fitting_k_free.setChecked(True)
            self.fitting_view.addWidget(self.fitting_k_free, row=1, col=3)
            self.fitting_b_free = QCheckBox("b")
            self.fitting_b_free.setChecked(True)
            self.fitting_view.addWidget(self.fitting_b_free, row=1, col=4)
            self.fitting_zp_free = QCheckBox("zp")
            self.fitting_zp_free.setChecked(True)
            self.fitting_view.addWidget(self.fitting_zp_free, row=1, col=5)  

            self.fitting_view.addWidget(QLabel(text='Fit prefix'), row=2, col=0)

            # Now get the data prefix
            self.fit_prefix = QLineEdit('{:}_SPOC_FIT'.format(title))
            self.fitting_view.addWidget(self.fit_prefix, row=2, col=1,  rowspan=1, colspan=5)

            self.fitting_view.addWidget(QLabel(text='Pop size'), row=3, col=0, colspan=1)
            self.fit_popsize = QLineEdit('30')
            self.fitting_view.addWidget(self.fit_popsize, row=3, col=1, colspan=1)

            self.fitting_view.addWidget(QLabel(text='tol'), row=3, col=2, colspan=1)
            self.fit_tol = QLineEdit('0.001')
            self.fitting_view.addWidget(self.fit_tol, row=3, col=3, colspan=1)

            # self.fitting_view.addWidget(QLabel(text='Nproc'), row=0, col=4, colspan=1)
            # self.fit_nproc = QLineEdit(str(multiprocessing.cpu_count()))
            # self.fitting_view.addWidget(self.fit_nproc, row=0, col=5, colspan=1)
            self.plot_checkbox = QCheckBox("Plot during fit? ")
            self.fitting_view.addWidget(self.plot_checkbox,row=3, col=5, colspan=1)
            # self.fit_burn_in = QLineEdit('4000')
            # self.fitting_view.addWidget(self.fit_burn_in, row=3, col=5, colspan=1)


            # Now add the fit button
            self.fit_lc_Btn = QtGui.QPushButton('FIT')
            self.fitting_view.addWidget(self.fit_lc_Btn, row=4, col=0)
            self.fit_lc_Btn.clicked.connect(self.fit_lightcurve)

            # Now create the progress bar
            self.fitting_view_progress_bar = QProgressBar()
            self.fitting_view_progress_bar.setGeometry(200, 80, 250, 20)
            self.fitting_view.addWidget(self.fitting_view_progress_bar, row=4, col=1, colspan=5)

            # Now lets add a results box
            self.results_box = QPlainTextEdit('Results will appear here')
            self.fitting_view.addWidget(self.results_box, row=5, col=0, colspan=5, rowspan=4)

            self.fit_lc_write_Btn = QtGui.QPushButton('WRITE')
            self.fitting_view.addWidget(self.fit_lc_write_Btn, row=5, col=5)
            self.fit_lc_write_Btn.clicked.connect(self.write_results)

            #self.fit_lc_chains_Btn = QtGui.QPushButton('CHAINS')
            #self.fitting_view.addWidget(self.fit_lc_chains_Btn, row=6, col=5)
            #self.fit_lc_chains_Btn.clicked.connect(self.plot_chains)

            self.fit_lc_corner_Btn = QtGui.QPushButton('CORNER')
            self.fitting_view.addWidget(self.fit_lc_corner_Btn, row=7, col=5)
            self.fit_lc_corner_Btn.clicked.connect(self.plot_corner)

            # Finally, add the view
            self.d_fitting.addWidget(self.fitting_view)

            # Finally set anti alias
            pg.setConfigOptions(antialias=True)

            self.win.show()



        def onClick(self,event):
            if self.get_tzero_toggle.isChecked():
                items = self.w1.scene().items(event.scenePos())
                mousePoint = self.w1.vb.mapSceneToView(event._scenePos)
                self.lc_pars_epoch_input.setText(str( mousePoint.x() ))
                #print(mousePoint.x(), mousePoint.y())


        def roiChangedEvent(self,):
            width, depth = self.normed_roi.size()
            #self.roi_width.setText('{:.3f}'.format(width*24))
            #self.roi_width.setText('{:.3f}'.format(1e3*depth))

        def update_normed_roi(self,):
            # First update the position and width
            bottom = self.w2.getAxis('bottom').range
            left = self.w2.getAxis('left').range
            self.normed_roi.setPos([(bottom[0] + bottom[1])/2. , (left[0] + left[1])/2.], update=True, finish=True)
            #self.normed_roi.setSize([0.1*(bottom[1] - bottom[0]) , 0.1*(left[1] - left[0])])


            # Now change the visibility
            #self.normed_roi.setVisible(not self.normed_roi.isVisible())


        def plot_corner(self,):
            if not hasattr(self, 'results') : 
                print('No fit has been done yet!')
                return

            import matplotlib.pyplot as plt
            import corner 

            f = corner.corner(self.results['samples'], labels = self.results['theta_names'], truths = self.results['best_step'] )   
            filename = '{:}_CORNER.png'.format(self.fit_prefix.text())
            plt.savefig(filename)
            print('Saved figure to {:}'.format(filename))
            plt.close()

        def plot_chains(self,):
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(len(self.results['theta_names']), figsize=(5, 7*len(self.results['theta_names'])/3.), sharex=True)
            for i in range(len(self.results['theta_names'])):
                axes[i].semilogx(self.results['samples_raw'][:, :, i], "k", alpha=0.3)
                axes[i].set_ylabel(self.results['theta_names'][i])
                axes[i].axvline(self.results['burn_in'], c='r', ls='--')
            axes[-1].set_xlabel("Step number")
            axes[0].set_xlim(0, len(self.results['samples_raw'][:, :, 0]))
            plt.tight_layout()
            filename = '{:}_CHAINS.png'.format(self.fit_prefix.text())
            plt.savefig(filename)
            print('Saved figure to {:}'.format(filename))
            plt.close()

        def write_results(self,):
            if not hasattr(self, 'results') : 
                print('No fit has been done yet!')
                return

            results_copy = self.results['pars'].copy()
            results_copy['t_zero'][0] += 2457000

            filename = '{:}.p'.format(self.fit_prefix.text())
            pickle.dump( results_copy, open( '{:}'.format(filename), "wb" ) )
            print('Written to {:}'.format(filename))

            filename = '{:}.txt'.format(self.fit_prefix.text())
            with open(filename, 'w+') as f:
                f.write(self.results_box.toPlainText())
                f.close()
            print('Written to {:}'.format(filename))


        def fit_lightcurve(self, ):
            # Get a list of fittet parameters
            fitted_parameters = []
            p0 = []
            bounds = []
            all_pars, all_pars_names = [], []
            if self.fitting_t_zero_free.isChecked() : 
                fitted_parameters.append('t_zero')
                p0.append(float(self.lc_pars_epoch_input.text()))
                bounds.append([p0[-1]-0.1, p0[-1]+0.1] )
            all_pars.append(float(self.lc_pars_epoch_input.text()))
            all_pars_names.append('t_zero')
            if self.fitting_period_free.isChecked() : 
                fitted_parameters.append('period')
                p0.append(float(self.lc_pars_period_input.text()))  
                bounds.append([p0[-1]-0.01, p0[-1]+0.01] )
            all_pars.append(float(self.lc_pars_period_input.text()))
            all_pars_names.append('period')
            if self.fitting_radius_1_free.isChecked() : 
                fitted_parameters.append('radius_1')
                p0.append(float(self.lc_pars_r1a_input.text())) 
                bounds.append([0, 0.5] ) 
            all_pars.append(float(self.lc_pars_r1a_input.text()))
            all_pars_names.append('radius_1')
            if self.fitting_k_free.isChecked() : 
                fitted_parameters.append('k')
                p0.append(float(self.lc_pars_r2r1_input.text())) 
                bounds.append([0,0.5] )
            all_pars.append(float(self.lc_pars_r2r1_input.text()))
            all_pars_names.append('k')
            if self.fitting_b_free.isChecked() : 
                fitted_parameters.append('b')
                p0.append(float(self.lc_pars_b_input.text())) 
                bounds.append([0, 1.5] )
            all_pars.append(float(self.lc_pars_b_input.text()))
            all_pars_names.append('b')
            if self.fitting_zp_free.isChecked() : 
                fitted_parameters.append('zp')
                p0.append(float(self.lc_pars_zp_input.text())) 
                bounds.append([0.97, 1.03] )
            all_pars.append(float(self.lc_pars_zp_input.text()))
            all_pars_names.append('zp')

            p0 , fitted_parameters, bounds, all_pars, all_pars_names = np.array(p0) , np.array(fitted_parameters), np.array(bounds), np.array(all_pars), np.array(all_pars_names)

            # Now cut the data if needed
            if self.fitting_cut_data.isChecked():
                # first get the transit width
                width = bruce.binarystar.transit_width(float(self.lc_pars_r1a_input.text()),
                                                    float(self.lc_pars_r2r1_input.text()),
                                                    float(self.lc_pars_b_input.text()),
                                                    float(self.lc_pars_period_input.text()))
                pwidth = bruce.binarystar.transit_width(float(self.lc_pars_r1a_input.text()),
                                                    float(self.lc_pars_r2r1_input.text()),
                                                    float(self.lc_pars_b_input.text()),
                                                    1.)
                # print('Pwidth : ', pwidth)
                # print('width : ', width)

                # now check if period is for fitting
                if 'period' in fitted_parameters:
                    phase = bruce.data.phase_times(self.data['TIME'], float(self.lc_pars_epoch_input.text()), float(self.lc_pars_period_input.text()), phase_offset=0.5)
                    mask = (phase > -pwidth) & (phase < pwidth)
                else : mask = (self.data['TIME'] > (float(self.lc_pars_epoch_input.text()) - width)) & (self.data['TIME'] < (float(self.lc_pars_epoch_input.text()) + width))
            else : mask = np.ones(len(self.data), dtype = bool)
            data = [np.array(self.data['TIME'][mask],dtype=np.float64), np.array(self.data['PDCSAP_FLUX'][mask],dtype=np.float64)/np.array(self.data['PDC_SAP_NORM'][mask],dtype=np.float64), np.array(self.data['PDCSAP_FLUX_ERR'][mask],dtype=np.float64)/np.array(self.data['PDC_SAP_NORM'][mask],dtype=np.float64)]
            #print('data' , len(data), data)

            # Check the fitted  parameters
            if len(fitted_parameters) == 0 : 
                print('No parameters for fitting!')
                return 

            # Now check the data
            if len(data[0]) == 0 : 
                print('No data for fitting!')
                return 

            # Now get p0
            p0 =  p0 #+ 1e-4* np.random.randn(nwalkers, ndim)

            # Now reset the progress par
            self.fitting_view_progress_bar.reset()

            args = (fitted_parameters, bounds, np.copy(data[0]) , np.copy(data[1]) ,np.copy(data[2]) , float(self.lc_pars_ldc_1_input.text()), float(self.lc_pars_ldc_2_input.text()), self.ld_law, all_pars, all_pars_names)


            # Replace with DE
            nll = lambda *args: -fit_lc_lnlike(*args)
            #print(nll(p0, *args), flush=True)
            print('Starting now... ', flush=True)
            self.fitting_view_progress_bar.setRange(0,0)

            def record_best(xk, convergence=None):
                # Now set the results
                if 't_zero' in fitted_parameters : self.lc_pars_epoch_input.setText(str( xk.copy()[np.argwhere(fitted_parameters=='t_zero')[0][0]]) )
                if 'period' in fitted_parameters : self.lc_pars_period_input.setText(str( xk.copy()[np.argwhere(fitted_parameters=='period')[0][0]]) )
                if 'radius_1' in fitted_parameters : self.lc_pars_r1a_input.setText(str( xk.copy()[np.argwhere(fitted_parameters=='radius_1')[0][0]]) )
                if 'k' in fitted_parameters : self.lc_pars_r2r1_input.setText(str( xk.copy()[np.argwhere(fitted_parameters=='k')[0][0]]) )
                if 'b' in fitted_parameters : self.lc_pars_b_input.setText(str( xk.copy()[np.argwhere(fitted_parameters=='b')[0][0]]) )
                if 'zp' in fitted_parameters : self.lc_pars_zp_input.setText(str( xk.copy()[np.argwhere(fitted_parameters=='zp')[0][0]]) )
                if self.plot_checkbox.isChecked():
                    self.generate_lc_model()
                    QApplication.processEvents()

            de = differential_evolution(nll, bounds=bounds,x0 = p0, 
                                            disp=True,args=args,
                                        maxiter=1000, polish=False, tol=float(self.fit_tol.text()),
                                        popsize=int(self.fit_popsize.text()),
                                        callback=record_best)
        

            #self.fitting_view_progress_bar.setValue(1)
            self.fit_lc_Btn.setText('FIT')

            # Get the results
            #self.results = bruce.sampler.get_sampler_report(sampler, burn_in, fitted_parameters, name=self.fit_prefix.text())
            self.results = {'pars' : {}}
            self.results['out_text'] = ''
            self.results['samples'] = de.population
            self.results['theta_names'] = fitted_parameters
            self.results['best_step'] = de.x
            for i in range(len(fitted_parameters)) : 
                self.results['pars'][fitted_parameters[i]] = [de.x[i], np.std(de.population[:,i]), np.std(de.population[:,i])]
                low = de.x[i] - np.percentile(de.population[:,i],16)
                high = np.percentile(de.population[:,i],84) - de.x[i]
                btjd_conv = ''
                if (fitted_parameters[i]=='t_zero') and (de.x[i]<10000):
                    btjd_conv = ' [{:.5f}]'.format(de.x[i]+2457000)
                self.results['out_text'] += '{:} : {:.5f} - {:.5f} + {:.5f} {:}\n'.format(fitted_parameters[i], de.x[i],low,high,btjd_conv)

            # Now set the results
            if 't_zero' in fitted_parameters : self.lc_pars_epoch_input.setText(str( self.results['pars']['t_zero'][0] ))
            if 'period' in fitted_parameters : self.lc_pars_period_input.setText(str( self.results['pars']['period'][0] ))
            if 'radius_1' in fitted_parameters : self.lc_pars_r1a_input.setText(str( self.results['pars']['radius_1'][0] ))
            if 'k' in fitted_parameters : self.lc_pars_r2r1_input.setText(str( self.results['pars']['k'][0] ))
            if 'b' in fitted_parameters : self.lc_pars_b_input.setText(str( self.results['pars']['b'][0] ))
            if 'zp' in fitted_parameters : self.lc_pars_zp_input.setText(str( self.results['pars']['zp'][0] ))


            # Now re-daw the data
            self.generate_lc_model()


            # Now ger the derived text
            width = bruce.binarystar.transit_width( float(self.lc_pars_r1a_input.text()),
                                                                    float(self.lc_pars_r2r1_input.text()),
                                                                    float(self.lc_pars_b_input.text()),
                                                                    float(self.lc_pars_period_input.text()))
            text_derived = 'width = {:.2f} hr [{:.5f} d]'.format(24.*width, width )
            text_derived += '\nDensity = {:.2f} rho sun'.format(bruce.binarystar.stellar_density( float(self.lc_pars_r1a_input.text()),
                                                                    float(self.lc_pars_period_input.text())))
            depth = 1 - np.min([np.min(self.lc_model_data[i][1]) for i in range(len(self.segments)) if self.lc_model_data[i][1].size > 0 ])
            text_derived += '\nDepth = {:.5f}'.format(depth)

            try:
                R1 = float(self.stellar_pars_R1_input.text())
                sun2jup = constants.R_sun.to(constants.R_jup).value
                if 'k' in fitted_parameters:
                    text_derived += '\n{:} = {:.2f} + {:.2f} - {:.2f} Rsun'.format('R2', R1*self.results['pars']['k'][0],R1*self.results['pars']['k'][1] ,R1*self.results['pars']['k'][2] )
                    text_derived += '\n{:} = {:.2f} + {:.2f} - {:.2f} RJup'.format('R2', R1*sun2jup*self.results['pars']['k'][0],R1*sun2jup*self.results['pars']['k'][1] ,R1*sun2jup*self.results['pars']['k'][2] )

                else : 
                    text_derived += '\n{:} = {:.5f} Rsun'.format('R2', R1*self.results['pars']['k'][0] )
            except Exception as e: print(e)
            text_derived += '\n{:} {:} {:}'.format(de.x[np.argwhere(fitted_parameters=='t_zero')[0][0]]+2457000, width, depth )

            # For monos sheet

            text_derived += '\n\nAmberMonos\n{:},{:},{:},{:},{:},'.format(de.x[np.argwhere(fitted_parameters=='t_zero')[0][0]]+2457000, 
                                                   float(self.lc_pars_period_input.text()),
                                                   de.x[np.argwhere(fitted_parameters=='radius_1')[0][0]],
                                                   de.x[np.argwhere(fitted_parameters=='k')[0][0]],
                                                   de.x[np.argwhere(fitted_parameters=='b')[0][0]])
            try : text_derived += str(R1*sun2jup*self.results['pars']['k'][0]) + ','
            except : text_derived += ','
            text_derived += '{:},{:}'.format(width*24, depth )


            # Now write to results
            self.results_box.setPlainText(self.results['out_text'] + text_derived)
            self.fitting_view_progress_bar.setRange(0,1)




        def interp_ld_pars(self,):
            try : Teff = float(self.stellar_pars_Teff_input.text())
            except : Teff = 5777

            try : FeH = float(self.stellar_pars_FeH_input.text())
            except : FeH = 0.

            try : logg = float(self.stellar_pars_logg_input.text())
            except : logg = 4.44

            from pycheops import ld 
            interper = ld.stagger_power2_interpolator('TESS')

            try:
                c, alpha, _,_ = interper(Teff, logg, FeH)
                if np.isnan(c) or np.isnan(alpha) : raise ValueError('a')
            except : 
                print('Interpolation failed for parameters {:} {:} {:}'.format(self.stellar_pars_Teff_input.text(),
                                                                            self.stellar_pars_FeH_input.text(),
                                                                            self.stellar_pars_logg_input.text() ))
                return 

            # Now set the parameters
            self.lc_pars_ldc_1_input.setText(str(c))
            self.lc_pars_ldc_2_input.setText(str(alpha))



        def query_tic_params(self,):
            tic_id = self.stellar_pars_tic_input.text()
            print('Querying TIC database for TIC-{:}'.format(tic_id), flush=True)

            # Now make the quere
            try : catalogue = Catalogs.query_object('TIC{:}'.format(self.stellar_pars_tic_input.text()), radius=.02, catalog="TIC")
            except Exception as e: 
                print('Unable to query {:}'.format())
                print(e)

            # Now update R1
            par = float(catalogue['rad'][0])
            if ~np.isnan(par) and ~np.isinf(par) : self.stellar_pars_R1_input.setText(str(par))

            par = float(catalogue['mass'][0])
            if ~np.isnan(par) and ~np.isinf(par) : self.stellar_pars_M1_input.setText(str(par))

            par = float(catalogue['Teff'][0])
            if ~np.isnan(par) and ~np.isinf(par) : self.stellar_pars_Teff_input.setText(str(par))

            par = float(catalogue['MH'][0])
            if ~np.isnan(par) and ~np.isinf(par) : self.stellar_pars_FeH_input.setText(str(par))

            par = float(catalogue['logg'][0])
            if ~np.isnan(par) and ~np.isinf(par) : self.stellar_pars_logg_input.setText(str(par))

            par = float(catalogue['Tmag'][0])
            if ~np.isnan(par) and ~np.isinf(par) : self.stellar_pars_Tmag.setText(str(par))


            
            #a['rad', 'mass' 'Teff', 'MH', 'logg'

        def update_single_transit_par(self,):
            if self.lc_pars_single_check_box.isChecked() : self.ld_law = -2
            else : self.ld_law = 2

        def clear_lc_model(self,):
            self.lc_model_data = None

            # Now clear the axis
            self.w1.clear()
            self.w2.clear()
            self.w3.clear()

            # Now re-draw the data
            self.draw_data()

            # Now re-add the ROI
            self.w2.addItem(self.normed_roi)
            self.normed_roi.setVisible(False)
            #self.normed_roi_checkbox.setChecked(False)
            #self.update_normed_roi()

        def generate_lc_model(self,):
            try:
                # First, generate the parameters and check they are valid
                t_zero = float(self.lc_pars_epoch_input.text())
                period = float(self.lc_pars_period_input.text())
                radius_1 = float(self.lc_pars_r1a_input.text())
                k = float(self.lc_pars_r2r1_input.text())
                b = float(self.lc_pars_b_input.text())
                zp = float(self.lc_pars_zp_input.text())
                ldc_1 =  float(self.lc_pars_ldc_1_input.text())
                ldc_2 =  float(self.lc_pars_ldc_2_input.text())

                # Now iterate the segments
                self.lc_model_data = []
                for i in range(len(self.segments)):
                    t_ = np.arange(np.min(self.data['TIME'][self.segments[i]]), np.max(self.data['TIME'][self.segments[i]]), 0.000694444)

                    # Acceleration check
                    x_range, _ = self.w1.viewRange()
                    print(x_range)
                    print(np.max(self.data['TIME'][self.segments[i]]))
                    print(np.min(self.data['TIME'][self.segments[i]]))

                    if (np.max(self.data['TIME'][self.segments[i]]) < x_range[0]) or (np.min(self.data['TIME'][self.segments[i]]) > x_range[1]):
                        print('Dummying data')
                        self.lc_model_data.append([t_, zp*np.ones(t_.shape[0])])
                    else:
                        f_ = zp*bruce.binarystar.lc(t_, t_zero=t_zero, period=period, radius_1=radius_1, k=k, incl = np.arccos(radius_1*b),
                                            c=ldc_1, alpha=ldc_2, ld_law=self.ld_law, cadence = 0.00694444)
                        if (True in np.isnan(f_)) or (True in np.isinf(f_)) : raise ValueError('Nan in LC model')
                        self.lc_model_data.append([t_, f_])

            except Exception as e: 
                print('Generating the transit model failed because:')
                print(e)
                return 

            # Now clear the axis
            self.w1.clear()
            self.w2.clear()
            self.w3.clear()

            # Now re-draw the data
            self.draw_data()

            # Now re-add the ROI
            self.w2.addItem(self.normed_roi)
            self.normed_roi.setVisible(False)
            #self.normed_roi_checkbox.setChecked(False)
            #self.update_normed_roi()

            pass

        def write_fits_data(self,):
            self.data.write(self.write_data_prefix.text()+'.fits', overwrite=True, format='fits')

        def write_flux_data(self,):
            np.savetxt(self.write_data_prefix.text()+'_FLUX.dat', np.array([self.data['TIME'] + 2457000, self.data['PDCSAP_FLUX'],self.data['PDCSAP_FLUX_ERR']]).T)
            np.savetxt(self.write_data_prefix.text()+'_FLUX_NORMALISED.dat', np.array([self.data['TIME'] + 2457000, self.data['PDCSAP_FLUX'] / self.data['PDC_SAP_NORM'],self.data['PDCSAP_FLUX_ERR'] / self.data['PDC_SAP_NORM']]).T)

        def write_mag_data(self,):
            m, me = bruce.data.flux_to_mags(self.data['PDCSAP_FLUX'],self.data['PDCSAP_FLUX_ERR'])
            np.savetxt(self.write_data_prefix.text()+'_MAG.dat', np.array([self.data['TIME'] + 2457000, m,me]).T)
            m, me = bruce.data.flux_to_mags(self.data['PDCSAP_FLUX'] / self.data['PDC_SAP_NORM'],self.data['PDCSAP_FLUX_ERR'] / self.data['PDC_SAP_NORM'])
            np.savetxt(self.write_data_prefix.text()+'_MAG_NORMALISED.dat', np.array([self.data['TIME'] + 2457000, m,me]).T)


        def save_plot(self,):
            self.exporter = pg.exporters.ImageExporter(self.view.ci)
            self.exporter.export('{:}.png'.format(self.save_plot_prefix.text()))

        def resegment_data(self,):
            # First, clear the axis
            self.w1.clear()
            self.w2.clear()
            self.w3.clear()

            # Now get the segments
            self.segments = bruce.data.find_nights_from_data(self.data['TIME'], float(self.gradsplitText.text()))

            # Now re-draw the data
            self.draw_data()

            # Now re-add the ROI
            self.w2.addItem(self.normed_roi)
            self.normed_roi.setVisible(False)
        # self.normed_roi_checkbox.setChecked(False)
            #self.update_normed_roi()

        def renormalise_data(self,):
            # First, clear the axis
            self.w1.clear()
            self.w2.clear()
            self.w3.clear()

            # Now re-normalise the data
            self.normalise_data()

            # Now re-draw the data
            self.draw_data()

            # Now re-add the ROI
            self.w2.addItem(self.normed_roi)
            self.normed_roi.setVisible(False)
            #self.normed_roi_checkbox.setChecked(False)
            #self.update_normed_roi()           



        def reload_tic_data(self, ):
            self.load_data(int(self.stellar_pars_tic_input.text()))
            self.clear_lc_model()




        def load_data(self, tic_id ):
            data_table, data, data_labels, base_dir = bruce.ambiguous_period.data_retrieval.download_tess_data(tic_id, use_ffi=True)

            self.data = Table() 
            self.data.add_column(Column(np.concatenate( [i.time-2457000.0 for i in data]  ), name='TIME'))
            self.data.add_column(Column(np.concatenate( [i.flux/np.median(i.flux) for i in data]  ), name='SAP_FLUX'))
            self.data.add_column(Column(np.concatenate( [i.flux/np.median(i.flux)  for i in data]  ), name='PDCSAP_FLUX'))
            self.data.add_column(Column(np.concatenate( [i.flux_err/np.median(i.flux)  for i in data]  ), name='PDCSAP_FLUX_ERR'))
            self.data.add_column(Column(np.concatenate( [i.sky_bkg for i in data]  ), name='SAP_BKG'))
            self.data.add_column( np.ones(len(self.data))*median_abs_deviation(self.data['SAP_BKG']), name='SAP_BKG_ERR')

            # bind the labels
            self.sector_info = [] 
            for i in range(len(data)):
                self.sector_info.append({'sector' : data_labels[i] + ' (' + data_table['Source'][i] + ')', 
                                         'tmin' : data[i].time[0]-2457000.0, 
                                         'tmax' : data[i].time[-1]-2457000.0, 
                                         'sector_max' :  1 + 5*np.std(data[i].flux/np.median(data[i].flux))})
            

            # Now get the segments
            self.segments = bruce.data.find_nights_from_data(self.data['TIME'], float(self.gradsplitText.text()))

            # Now normalise 
            self.normalise_data()


            # now normalise
            # if 'PDC_SAP_NORM' not in self.data.colnames:
            #     self.normalise_data()
            # else:
            #     self.MAD = np.mean([median_abs_deviation(self.data['PDCSAP_FLUX'][self.segments[i]]/self.data['PDC_SAP_NORM'][self.segments[i]]) for i in range(len(self.segments))])
            #     self.MFthreshold = 3*self.MAD

                
        def normalise_data(self,):
            # Check and add the normalisation column
            if 'PDC_SAP_NORM' not in self.data.colnames : self.data.add_column(Column(np.zeros(len(self.data)), name='PDC_SAP_NORM'))

            # Check SG_window_length
            if not hasattr(self, 'SG_window_length') : self.SG_window_length = 101

            # Now check it is odd 
            if not self.SG_window_length&1 : self.SG_window_length +=1

            # Now normalise
            for i in range(len(self.segments)):
                try : window_scale = int(float(self.SG_window_text.text()) / np.median(np.gradient(self.data['TIME'][self.segments[i]])) )
                except : window_scale= 101
                if window_scale < 10 : window_scale = 10
                if not window_scale&1 : window_scale +=1

                try : self.data['PDC_SAP_NORM'][self.segments[i]] = bruce.data.flatten_data_with_function(self.data['TIME'][self.segments[i]], self.data['PDCSAP_FLUX'][self.segments[i]], SG_window_length=window_scale, SG_iter=int(self.SG_iter_text.text()))
                except : self.data['PDC_SAP_NORM'][self.segments[i]] = bruce.data.flatten_data_with_function(self.data['TIME'][self.segments[i]], self.data['PDCSAP_FLUX'][self.segments[i]], method='poly1d')
            # # Now re-calculate MAD
            # self.MAD = np.mean([median_abs_deviation(self.data['PDCSAP_FLUX'][self.segments[i]]/self.data['PDC_SAP_NORM'][self.segments[i]]) for i in range(len(self.segments))])
            # self.MFthreshold = 3*self.MAD

        def draw_data(self,):
            self.data_raw_plot_error =  [self.w1.addItem(pg.ErrorBarItem(x=self.data['TIME'][self.segments[i]], y=self.data['PDCSAP_FLUX'][self.segments[i]], top=self.data['PDCSAP_FLUX_ERR'][self.segments[i]], bottom=self.data['PDCSAP_FLUX_ERR'][self.segments[i]], beam=0.01))
                                for i in range(len(self.segments))]
            self.data_raw_plot_point =  [self.w1.plot(self.data['TIME'][self.segments[i]], self.data['PDCSAP_FLUX'][self.segments[i]], pen =pg.mkPen('w'))
                                for i in range(len(self.segments))]
                
            self.data_raw_norm =  [self.w1.plot(self.data['TIME'][self.segments[i]], self.data['PDC_SAP_NORM'][self.segments[i]], pen ='r', symbol =None, symbolPen ='r', symbolBrush = 0.2,name='Savgol filter' if i==0 else None)
                                for i in range(len(self.segments))]


            self.data_norm_plot_error =  [self.w2.addItem(pg.ErrorBarItem(x=self.data['TIME'][self.segments[i]], y=self.data['PDCSAP_FLUX'][self.segments[i]] / self.data['PDC_SAP_NORM'][self.segments[i]], top=self.data['PDCSAP_FLUX_ERR'][self.segments[i]]/self.data['PDC_SAP_NORM'][self.segments[i]], bottom=self.data['PDCSAP_FLUX_ERR'][self.segments[i]]/self.data['PDC_SAP_NORM'][self.segments[i]], beam=0.01))
                                for i in range(len(self.segments))]
            self.data_norm_plot_point =  [self.w2.plot(self.data['TIME'][self.segments[i]], self.data['PDCSAP_FLUX'][self.segments[i]]/self.data['PDC_SAP_NORM'][self.segments[i]], pen =pg.mkPen('w'))
                                for i in range(len(self.segments))]   
            

            self.sky_data_raw_plot_error =  [self.w3.addItem(pg.ErrorBarItem(x=self.data['TIME'][self.segments[i]], y=self.data['SAP_BKG'][self.segments[i]], top=self.data['SAP_BKG_ERR'][self.segments[i]], bottom=self.data['SAP_BKG_ERR'][self.segments[i]], beam=0.01))
                                for i in range(len(self.segments))]
            self.data_norm_plot_point =  [self.w3.plot(self.data['TIME'][self.segments[i]], self.data['SAP_BKG'][self.segments[i]],pen =pg.mkPen('r'))
                                for i in range(len(self.segments))]   
            self.w1.setLimits()


            self.zeropoint_line = pg.InfiniteLine(movable=False, angle=0, pen=pg.mkPen('r', width=1,style=QtCore.Qt.DashLine), hoverPen=pg.mkPen('r', width=3,))
            self.zeropoint_line.setPos([0, 1 ])
            self.w2.addItem(self.zeropoint_line)


            # self.threshold_line = pg.InfiniteLine(movable=False, angle=0, pen=pg.mkPen('r', width=3,), hoverPen=pg.mkPen('r', width=3,), name='MF threshold')
            # self.threshold_line.setPos([0, 1 - self.MFthreshold])
            # self.w2.addItem(self.threshold_line)

            for i in range(len(self.sector_info)):
                lr1 = pg.LinearRegionItem([self.sector_info[i]['tmin'], self.sector_info[i]['tmax']], movable=False,
                                        brush = pg.mkBrush((50,50,200,40)))
                self.w1.addItem(lr1) 
                lr2 = pg.LinearRegionItem([self.sector_info[i]['tmin'], self.sector_info[i]['tmax']], movable=False,
                                        brush = pg.mkBrush((50,50,200,40)))
                self.w2.addItem(lr2) 
                lr3 = pg.LinearRegionItem([self.sector_info[i]['tmin'], self.sector_info[i]['tmax']], movable=False,
                                        brush = pg.mkBrush((50,50,200,40)))
                self.w3.addItem(lr3) 

                text = pg.TextItem(text = self.sector_info[i]['sector'])
                self.w1.addItem(text)
                text.setPos(np.mean([self.sector_info[i]['tmin'], self.sector_info[i]['tmax']]), 1.01*self.sector_info[i]['sector_max'])


            # Now plot the lc model if we have it
            if self.lc_model_data is not None:
                for i in range(len(self.segments)):
                    self.w2.plot(self.lc_model_data[i][0], self.lc_model_data[i][1],pen =pg.mkPen('g'), name='Transit Model' if i==0 else None)
                    norm_interped = np.interp(self.lc_model_data[i][0], self.data['TIME'][self.segments[i]], self.data['PDC_SAP_NORM'][self.segments[i]])
                    self.w1.plot(self.lc_model_data[i][0], norm_interped*self.lc_model_data[i][1],pen =pg.mkPen('g'), name='Transit Model' if i==0 else None)

    
    # Create a plot with some random data
    app = Window(title='TIC-{:}'.format(sys.argv[1]))
    app.load_data(sys.argv[1])

    app.draw_data()


    pg.exec()

    exit()


if __name__ == '__main__':
    main()