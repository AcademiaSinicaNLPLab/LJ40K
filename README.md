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
5. Train, Cross-validation and Test

	```python
	>> from feelit.features import Learning
	>> learner = Learning(verbose=args.verbose, debug=args.debug) 
        >> learner.set(X_train, y_train, feature_name)
        >>
        >> scores = {}
        >> for C in Cs:
        >> 	for gamma in gammas:
        >> 		score = learner.kFold(kfolder, classifier='SVM', 
        >>					kernel='rbf', prob=False, 
        >>					C=c, scaling=True, gamma=gamma)
        >>		scores.update({(c, gamma): score})
        >>
	>> best_C, best_gamma = max(scores.iteritems(), key=operator.itemgetter(1))[0]
	>> learner.train(classifier='SVM', kernel='rbf', prob=True, C=best_C, gamma=best_gamma, 
	>>		scaling=True, random_state=np.random.RandomState(0))
	>> results = learner.predict(X_test, yb_test, weighted_score=True, X_predict_prob=True, auc=True)
	```

