LJ40K
=====

Python modules for analyzing LJ40K emotion data

## System flow

![feelit flow](https://cloud.githubusercontent.com/assets/1659204/5698196/fd3873e8-9a42-11e4-803e-81c59a12c143.png)


## feelit/features.py

1. Load features from files

	```python
	>> from feelit.features import LoadFile
	>> lf = LoadFile(verbose=True)
	>> lf.loads(root="../emotion_imgs_threshold_1x1_rbg_out_amend/out_f1", data_range=800)
	>> lf.dump(path="data/image_rgb_gist.Xy", ext=".npz")
	```
2. Load features from mongodb

	```python
	>> from feelit.features import FetchMongo
	>> fm = FetchMongo(verbose=True)
	>> fm.fetch_transform('TFIDF', '53a1921a3681df411cdf9f38', data_range=800)
	>> fm.dump(path="data/TFIDF.Xy", ext=".npz")
	```

3. Fuse loaded features

	```python
	>> from feelit.features import Fusion
	>> fu = Fusion(verbose=True)
	>> fu.loads(a1, a2, ...)
	>> fu.fuse()
	>> fu.dump()
	```
4. Train a classifier

	```python
	>> from feelit.features import Learning
	>> l = Learning(verbose=True)
	>> l.load(path="data/DepPairs_LSA512+TFIDF_LSA512+keyword_LSA512+rgba_gist+rgba_phog.Xy.npz")
	>> l.kFold()
	>> l.save(root="results")
	```
