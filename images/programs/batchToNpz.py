import sys, os, logging
sys.path.append('../..')
from feelit.features import LoadFile




if __name__ == '__main__':

	features = ['rgb_gist', 'rgb_phog', 'rgba_gist', 'rgba_phog']
	input_root = 'output/csvs'
	output_root = 'output/npzs'

	for f in features:

		input_path = os.path.join(input_root, f)
		output_path = os.path.join(output_root, f)

		lf = LoadFile(verbose=True)
		lf.loads(input_path, data_range=(None,800), amend=True)
		output_file = os.path.join(output_path, '%s_train' % f)
		lf.dump(output_file)



		lf = LoadFile(verbose=True)
		lf.loads(input_path, data_range=(-200,None), amend=True)
		output_file = os.path.join(output_path, '%s_test' % f)
		lf.dump(output_file)

		"""
		lf = LoadFile(verbose=True)
		lf.loads(input_path, data_range=(None,800), amend=True)
		output_file = os.path.join(output_path, '%s_1000' % f)
		lf.dump(output_file)
		"""
		