import numpy as np
import pandas as pd
import pyqtgraph as pg
# from PyQt5.QtGui import QPoint, QPainterPath
from PyQt5.QtCore import QPointF
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QPushButton, QLineEdit, QFileDialog
from ec_code.phy_tools.utilities import crop_trace, autocorrelation
from sklearn.decomposition import PCA
import os
improt flammkuchen as fl


class SpikeSortingWidget(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ly_0_1 = QHBoxLayout()

        # Graphic settings
        all_col = (70, 150, 70, 20)
        sel_col = (255, 70, 70, 50)
        scatter_dot_size = 10
        plot_line_width = 1

        self.pre_int = 20
        self.fn = 8333.33333
        self.end_time = 103.35888

        self.m = []
        self.positions = []
        self.pca0 = []
        self.pca1 = []

        self.roi = pg.PolyLineROI(positions=[], closed=True)

        #################
        # PCA scatter box
        self.scatter_ly = QVBoxLayout()

        # Plot:
        self.scatter_plot_wg = pg.PlotWidget(autoscale=False)

        self.roi.sigRegionChangeFinished.connect(self.select_spikes)
        self.scatter_plot_wg.addItem(self.roi)

        self.pca_plot = pg.ScatterPlotItem(size=scatter_dot_size, pen=pg.mkPen(None), brush=pg.mkBrush(*all_col))
        self.pca_sel_plot = pg.ScatterPlotItem(size=scatter_dot_size, pen=pg.mkPen(None), brush=pg.mkBrush(*sel_col))
        self.scatter_plot_wg.addItem(self.pca_plot)
        self.scatter_plot_wg.addItem(self.pca_sel_plot)

        # Button
        self.clear_butt = QPushButton('Clear!')
        self.clear_butt.clicked.connect(self.remove_selected)

        self.scatter_ly.addWidget(self.scatter_plot_wg)
        self.scatter_ly.addWidget(self.clear_butt)

        ############
        # Spike plot
        self.spikes_wg = pg.PlotWidget(autoscale=False, connect='finite')
        self.spike_plot = pg.PlotCurveItem(pen=pg.mkPen(*all_col, width=plot_line_width), connect='finite')
        self.spikes_sel_plot = pg.PlotCurveItem(pen=pg.mkPen(*sel_col, width=plot_line_width), connect='finite')
        self.spikes_wg.addItem(self.spike_plot)
        self.spikes_wg.addItem(self.spikes_sel_plot)

        ############
        # Correlation plots
        self.corr_ly = QGridLayout()
        self.corr_plots = []
        self.corr_traces = []
        for i in range(4):
            plt = pg.PlotWidget(autoscale=False, connect='finite')
            trace = pg.PlotCurveItem(pen=pg.mkPen(*(120, 120, 120), width=2), connect='finite')
            self.corr_plots.append(plt)
            self.corr_traces.append(trace)
            self.corr_plots[-1].addItem(self.corr_traces[-1])
            self.corr_ly.addWidget(plt, np.mod(i, 2), i//2)

        ####################
        # Trace scatter plot
        self.trace_wg = pg.PlotWidget(autoscale=False, connect='finite')
        self.trace_plot = pg.ScatterPlotItem(size=scatter_dot_size, pen=pg.mkPen(None),
                                             brush=pg.mkBrush(*all_col))
        self.trace_sel_plot = pg.ScatterPlotItem(size=scatter_dot_size, pen=pg.mkPen(None),
                                                 brush=pg.mkBrush(*sel_col))
        self.trace_wg.addItem(self.trace_plot)
        self.trace_wg.addItem(self.trace_sel_plot)


        ###############
        # File controls
        self.control_ly = QHBoxLayout()

        self.file_butt = QPushButton('...')
        self.file_butt.clicked.connect(self.file_dialog)
        self.file_box = QLineEdit('/Users/luigipetrucco/Desktop/motor_ec/wholecell/17126029_32_vclamp_lpfilt.h5')
        self.load_butt = QPushButton('Load')
        self.load_butt.clicked.connect(self.load_new_file)

        self.save_butt = QPushButton('Save')
        self.save_butt.clicked.connect(self.save_file)

        self.control_ly.addWidget(self.file_butt)
        self.control_ly.addWidget(self.file_box)
        self.control_ly.addWidget(self.load_butt)
        self.control_ly.addWidget(self.save_butt)

        # Layout
        self.ly_0 = QVBoxLayout()
        self.ly_0_1.addWidget(self.spikes_wg, 1)
        self.ly_0_1.addLayout(self.scatter_ly, 1)
        self.ly_0_1.addLayout(self.corr_ly, 1)

        self.ly_0.addWidget(self.trace_wg, 1)
        self.ly_0.addLayout(self.ly_0_1, 1)
        self.ly_0.addLayout(self.control_ly)

        self.setLayout(self.ly_0)
        self.show()

    def file_dialog(self):
        wnd = QFileDialog()
        filename = wnd.getOpenFileName(directory='/Users/luigipetrucco/Desktop/motor_ec/exported', filter='*.h5')
        if filename[0]:
            self.file_box.setText(filename[0])

    def select_spikes(self):
        roiShape = self.roi.mapToItem(self.scatter_plot_wg.getPlotItem(), self.roi.shape())
        # Get list of all points inside shape
        points = [self.scatter_plot_wg.getPlotItem().mapFromView(QPointF(pt_x, pt_y))
                  for pt_x, pt_y in zip(self.pca0, self.pca1)]
        self.contained = [i for i, p in enumerate(points) if roiShape.contains(p)]

        self.update_sel_plots(np.array(self.contained))

    def get_xy(self, m):
        trace = np.concatenate((m, np.empty((m.shape[0], 1)) * np.nan), 1).flatten()
        time = np.tile((np.concatenate([np.arange(m.shape[1]), [np.nan]]) - self.pre_int) / self.fn, m.shape[0])
        return trace, time

    def load_new_file(self):
        filename = self.file_box.text()
        #try:
        data = fl.load(filename)

        self.data = data
        print(data.keys())
        df = data['data_dict']['trace']
        print(df)
        self.positions = data['data_dict']['spikes']
        spike_mat = crop_trace(np.array(df.cell_mV),
                               self.positions, pre_int=20, post_int=80, rebase=np.arange(25, 30))
        spike_mat[np.isnan(spike_mat)] = 0

        self.m = spike_mat

        self.update_figures()
        #except:
        #    print('Invalid file!')


    def update_sel_plots(self, idxs):
        if len(idxs) > 0:
            b, t = self.get_xy(self.m[idxs, :])
            self.spikes_sel_plot.setData(y=b, x=t)
            self.pca_sel_plot.setData(x=self.pca0[idxs], y=self.pca1[idxs])

            self.trace_sel_plot.setData(x=np.mod(self.positions[idxs]/self.fn, self.end_time),
                                        y=(self.positions[idxs]/self.fn) // self.end_time)

            # Autocorrs:
            mask = np.ones(len(self.positions)).astype('bool')
            mask[self.contained] = False
            train1 = self.positions[mask] / self.fn
            train2 = self.positions[self.contained] / self.fn

            for j, (t1, t2) in enumerate(zip([train1] * 2 + [train2] * 2, [train1, train2] * 2)):
                auto, bins = autocorrelation(spike_times=t1, spike_times2=t2,
                                             bin_width=15e-3, width=0.4e-0)
                self.corr_traces[j].setData(x=bins[1:], y=auto)

        else:
            self.spikes_sel_plot.setData(y=[], x=[])
            self.pca_sel_plot.setData(y=[], x=[])
            self.trace_sel_plot.setData(y=[], x=[])

    def update_figures(self):

        # Perform PCA:
        sel = self.m.T
        pca = PCA(n_components=2)
        pca.fit(sel - np.mean(sel, 0))
        self.pca0 = pca.components_[0, :]
        self.pca1 = pca.components_[1, :]

        # Default square:
        x_min, x_max = np.min(self.pca0), np.max(self.pca0)
        y_min, y_max = np.min(self.pca1), np.max(self.pca1)
        x0 = (x_max + x_min) / 2 - (x_max - x_min) / 4
        x1 = (x_max + x_min) / 2 + (x_max - x_min) / 4
        y0 = (y_max + y_min) / 2 - (y_max - y_min) / 4
        y1 = (y_max + y_min) / 2 + (y_max - y_min) / 4

        self.roi.setPoints([(x0, y1), (x0, y0), (x1, y0), (x1, y1)])
        self.pca_plot.setData(x=self.pca0, y=self.pca1)

        # spike shapes:
        b, t = self.get_xy(self.m)
        self.spike_plot.setData(y=b, x=t)

        # spike trace:
        self.trace_plot.setData(x=np.mod(self.positions / self.fn, self.end_time),
                                    y=(self.positions/self.fn) // self.end_time)

        # autocorr:
        # train1 = self.positions / self.fn
        # train2 = self.positions[self.contained] / self.fn
        # #for
        # auto, bins = autocorrelation(spike_times=train1, spike_times2=train1,
        #                              bin_width=10e-3, width=0.5e-0)
        # self.corr_traces[0].setData(x=bins[1:], y=auto)
        # auto, bins = autocorrelation(spike_times=train2, spike_times2=train2,
        #                              bin_width=10e-3, width=0.5e-0)
        # self.corr_traces[3].setData(x=bins[1:], y=auto)


    def remove_selected(self):

        mask = np.ones(len(self.positions)).astype('bool')
        mask[self.contained] = False
        self.m = self.m[mask, :]
        self.positions = self.positions[mask]
        print(len(self.positions))
        sel = self.m.T  # crop around the peak region...?
        pca = PCA(n_components=2)
        pca.fit(sel - np.mean(sel, 0))
        self.pca0 = pca.components_[0, :]
        self.pca1 = pca.components_[1, :]

        self.update_figures()

    def save_file(self):
        old_name = self.file_box.text()
        new_name = old_name[:-3] + 'clean.h5'

        self.data['data_dict']['spikes'] = self.positions
        print(len(self.data['data_dict']['spikes']))
        dd.io.save(new_name, self.data)
        print('saved!')
        #print(len(dd.io.load(old_name)['data_dict']['spikes']))
        #print(len(dd.io.load(new_name)['data_dict']['spikes']))


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import qdarkstyle

    #folder = '/Users/luigipetrucco/Desktop/motor_ec/wholecell/17126029_32_vclamp_lpfilt.h5r.h5'
    #fname = '17302002_11_n25.h5'

    app = QApplication([])
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    a = SpikeSortingWidget()

    app.exec_()
