

clear all
close all

dbstop if error;

% train it separately for each emotion
%features = {'TFIDF', 'keyword', 'image_rgba_gist', 'image_rgba_phog'};
mklv2_exp_1(1, 'E1', {'keyword', 'image_rgba_gist', 'image_rgba_phog', 'TFIDF'});
