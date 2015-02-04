
import sys
import os
import csv
from plotchart import PlotChart
import drawingutils

# python batchPlotCsv.py [file name] [fig title]

file_name = sys.argv[1]
fig_title = sys.argv[2]

DATA_DIR = 'Z:\\data\\MKLv2'
emotion_file_path = os.path.join(DATA_DIR, 'emotion.csv')
#file_name = 'exp_14_eval_result_15020211'
result_file_path = 'output\\%s.csv' % (file_name)
output_file_name = "output\\%s.png" % (file_name)

# get emotions
emotions = drawingutils.get_emotions_from_file(emotion_file_path)

myplot = PlotChart('multi-bar', fig_title, 'Emotions', 'Accuracy Rate')
myplot.set_x_ticks(emotions)

with open(result_file_path, 'rb') as result_csv:
    rows = csv.reader(result_csv)
    for row in rows:
        results = []
        for j, element in enumerate(row):
            if j == 0:
                feature_name = element
                continue
            results.append(float(element))
        myplot.add_feature(feature_name, results)

myplot.plot_and_save(output_file_name)
