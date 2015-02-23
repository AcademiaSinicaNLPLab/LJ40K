
#
# draw chart by using matplotlib (Enthought Canopy Python Package in win7)
# https://www.enthought.com/products/canopy/
#

import numpy as np
import matplotlib.pyplot as plt
import os, csv
import scipy.io as sciio
import drawingutils

class PlotChart:
        def __init__(self, chart_type, title, x_label='X', y_label='Y'):
            ''' chart_type: "multi-bar", '''
            self.title = title
            self.chart_type = chart_type
            self.x_label = x_label
            self.y_label = y_label
            self.feature_results = []
            self.x_ticks = ()
            self.num_group = 0

        def set_x_ticks(self, ticks):
            self.x_ticks = ticks
            self.num_group = len(ticks)

        def add_feature(self, feature_name, data):
            self.feature_results.append((feature_name, data))

        def get_feature_number():
            return len(self.feature_results)

        def _plot_bar_chart(self, file_name):

            fig, ax = plt.subplots()
            index = np.arange(self.num_group)
            bar_width = 0.1
            opacity = 0.4
            error_config = {'ecolor': '0.3'}

            rects = []
            i = 0
            for (feature, results) in self.feature_results:
                rects.append(plt.bar(index+i*bar_width, results, bar_width,
                                     alpha=opacity,
                                     color=drawingutils.pick_color(i),
                                     error_kw=error_config,
                                     label=feature))
                i = i + 1

            plt.xlabel(self.x_label, fontsize=18)
            plt.ylabel(self.y_label, fontsize=18)
            plt.title(self.title, fontsize=24)
            plt.xticks(index + bar_width, self.x_ticks, rotation='vertical')
            plt.margins(0.01)
            plt.ylim(0.4, 0.7)
            #plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1))
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=4,
                    ncol=4, borderaxespad=0.)
            plt.grid(True)

            plt.tight_layout()
            plt.show()
            #plt.savefig(file_name)

        def plot_and_save(self, file_name):
            {
                'multi-bar': self._plot_bar_chart(file_name)
            }[self.chart_type]
            
