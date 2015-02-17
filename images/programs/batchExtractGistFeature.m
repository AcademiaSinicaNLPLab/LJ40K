
addpath('gist');

source_dir = 'png_input/images/emotion-imgs-1x1-rgba/pattern'
dest_dir = 'output/mats/rgba_gist'

ExtractGistFeature(source_dir, dest_dir)

source_dir = dest_dir
dest_dir = 'output/csvs/rgba_gist'

BatchToCSV(source_dir, dest_dir)
    
disp('all done');





source_dir = 'png_input/images/emotion-imgs-1x1-rgb/pattern'
dest_dir = 'output/mats/rgb_gist'

ExtractGistFeature(source_dir, dest_dir)

source_dir = dest_dir
dest_dir = 'output/csvs/rgb_gist'

BatchToCSV(source_dir, dest_dir)
    
disp('all done');