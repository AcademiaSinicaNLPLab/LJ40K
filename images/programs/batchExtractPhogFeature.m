addpath('phog');

%source_dir = 'png_input/images/emotion-imgs-1x1-rgba/pattern'
%dest_dir = 'output/mats/rgba_phog'
source_dir = 'png_input/lj40k_png'
dest_dir = 'output_prediction/mats/rgba_phog'

ExtractPhogFeature(source_dir, dest_dir);

source_dir = dest_dir
dest_dir = 'output_prediction/csvs/rgba_phog'

BatchToCSV(source_dir, dest_dir)

disp('all done');



%addpath('phog');

%source_dir = 'png_input/images/emotion-imgs-1x1-rgb/pattern'
%dest_dir = 'output/mats/rgb_phog'

%ExtractPhogFeature(source_dir, dest_dir);

%source_dir = dest_dir
%dest_dir = 'output/csvs/rgb_phog'

%BatchToCSV(source_dir, dest_dir)

%disp('all done');


