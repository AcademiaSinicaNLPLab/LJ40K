addpath('phog');

source_dir = 'png_input/images/emotion-imgs-1x1-rgba/pattern'
dest_dir = 'output/mats/rgba_phog'

ExtractPhogFeature(source_dir, dest_dir);

source_dir = dest_dir
dest_dir = 'output/csvs/rgba_phog'

BatchToCSV(source_dir, dest_dir)

disp('all done');

