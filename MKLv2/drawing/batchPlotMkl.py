
import os
import csv
from plotchart import PlotChart
import drawingutils

DATA_DIR = 'Z:\\data\\MKLv2'
emotion_file_path = os.path.join(DATA_DIR, 'emotion.csv')
result_file_path = 'Z:\\github_repo\\LJ40K\\MKLv2\\drawing\\exp_1_eval_result_15012716.csv'

# get emotions
emotions = drawingutils.get_emotions_from_file(emotion_file_path)

myplot = PlotChart('multi-bar', '(1)-feature-type MKL', 'Emotions', 'Accuracy Rate')
myplot.set_x_ticks(emotions);

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

myplot.plot()
