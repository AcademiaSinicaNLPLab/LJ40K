
import csv

def get_emotions_from_file(file_path):
    emotions = ()
    with open(file_path, 'rb') as emotion_csv:
        csv_data = csv.reader(emotion_csv)
        for row in csv_data:
            for element in row:
                emotions = emotions + (element,)
    return emotions

def pick_color(idx):
    colors = ['c', 'm', 'g', 'r', 'b', 'y', 'k', 'w']
    return colors[idx]


