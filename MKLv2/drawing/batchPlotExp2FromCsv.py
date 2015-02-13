
import os
import csv
from plotchart import PlotChart
import drawingutils

DATA_DIR = 'Z:\\data\\MKLv2'
emotion_file_path = os.path.join(DATA_DIR, 'emotion.csv')
file_name = 'exp_22_eval_result_15020215'
result_file_path = 'output\\%s.csv' % (file_name)
output_file_path = "output\\%s.png" % (file_name)

# get emotions
emotions = drawingutils.get_emotions_from_file(emotion_file_path)

myplot = PlotChart('multi-bar', '(1)-feature-type MKL', 'Emotions', 'Accuracy Rate')
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

myplot.plot_and_save(output_file_path)
