'''
models.py

Created by 

name: Federico Marcuzzi
e-mail: federico.marcuzzi@unive.it

date 20/02/2020
'''

import numpy as np

from collections import Counter
from sklearn import tree

from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError


# tree wrapper.
class TreeWrapper():
	def __init__(self):
		self.node_count = 0
		self.children_left = []
		self.children_right = []
		self.feature = []
		self.threshold = []
		self.value = []

# this class manages the attributes and methods of trained trees in a projection of the original dataset.
class ProjectedDecisionTreeClassifier():
	def __init__(self, criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None):
		self.tree_ = TreeWrapper()
		self.n_feat = 0
		self.n_features_ = 0
		self.feature_importances_ = []
		self.albero = tree.DecisionTreeClassifier(criterion, splitter, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, random_state, max_leaf_nodes, min_impurity_decrease, min_impurity_split, class_weight)

	def fit_aux(self):
		self.n_features_ = self.albero.n_features_
		self.feature_importances_build_ = self.albero.feature_importances_

		self.tree_.node_count = self.albero.tree_.node_count

		self.tree_.children_left = self.albero.tree_.children_left
		self.tree_.children_right = self.albero.tree_.children_right

		self.tree_.feature = np.asarray([(lambda x: self.idx[x] if x >=0 else -2)(f) for f in self.albero.tree_.feature])
		self.tree_.threshold = self.albero.tree_.threshold
		self.tree_.value = self.albero.tree_.value

		app = np.argmax(self.tree_.value,axis=2).reshape(-1)
		app[self.tree_.feature!=-2]=-2
		app[app==0]=-1
		self.tree_.leaves_labels = app

		self.feature_importances_ = np.zeros((self.n_feat, ))

		for idx_f,imp_f in zip(self.idx,self.feature_importances_build_):
			self.feature_importances_[idx_f] = imp_f

	def fit(self,X,y,idx):
		_, self.n_feat = np.shape(X)
		self.idx = np.copy(idx)
		self.idx.sort()
		self.albero.fit(X[:,self.idx],y)
		self.fit_aux()

	def predict(self,X):
		return self.albero.predict(X[:,self.idx])

	def score(self,X,y):
		return self.albero.score(X[:,self.idx],y)

	def decision_path(self,X,check_input=True):
		return self.albero.decision_path(X[:, self.idx],check_input)

# this class manages the attributes and methods of a forest with trees trained on a projection of the original dataset.
class ProjectedForest(BaseEstimator):
	def __init__(self):
		self.forest = []
		self.index = -1
		self.iter = []
		self.n_trees = 0

	def check_fit(self):
		if len(self.forest) == 0:
			raise NotFittedError('This RandomForestClassifier instance is not fitted yet. Call \'fit\' with appropriate arguments before using this method.')

	def __iter__(self):
		self.check_fit()
		self.iter.append(0)
		self.index += 1
		return self

	def __next__(self):
		self.n_trees = len(self.forest)
		self.check_fit()
		if self.iter[self.index] < self.n_trees:
			self.iter[self.index] += 1
			return self.forest[self.iter[self.index] - 1]
		else:
			del self.iter[self.index]
			self.index -= 1
			raise StopIteration

	def fit(self,X,y):
		self.max_label = Counter(y).most_common()[0][0]

	def predict(self,X):
		self.check_fit()
		predict = np.sum([tr.predict(X) for tr in self.forest],axis=0)

		if self.n_trees % 2 == 0:
			predict[predict==0] = self.max_label

		predict[predict<0] = -1
		predict[predict>0] = 1
		return predict

	def score(self,X,y):
		return np.sum(self.predict(X)==y) / len(y)

# this class is the implementation of our robust FPF ensemble method.
class FeaturePartitionedForest(ProjectedForest):
	def __init__(self,b,r=10,n_est=None,min_acc=None,random_state=None,max_leaf_nodes=None):  
		super().__init__()
		self.forest = []
		self.b = b
		self.n_est = n_est

		if n_est is not None:
			self.r = n_est // (2 * b + 1)
		else:
			self.r = r

		self.min_acc = min_acc
		self.random_state = random_state
		self.max_leaf_nodes = max_leaf_nodes
		np.random.seed(seed=random_state)

	def fit(self,X,y,X_val=None,y_val=None):
		super().fit(X,y)
		n_ist, n_feat = np.shape(X)
		idx_f = np.arange(n_feat)
		min_num_tree = 2 * self.b + 1

		# if the papameter min_acc is None, It is set with the majority class.
		if self.min_acc == None:
			self.min_acc = np.round(Counter(y).most_common()[0][1] / n_ist, decimals=3)

		if X_val == None:
			X_val = X
			y_val = y

		self.forest_counter = 0
		# for each round 'r' create a sub-forest of '2b + 1' trees.
		for _ in range(self.r):
			forest = []
			idx_f = np.copy(idx_f)
			# shuffles feature indexes.
			np.random.shuffle(idx_f)
			# performs robust partitioning: split the feature sets into '2b + 1' partitions.
			slice_idx = np.array_split(idx_f, min_num_tree)

			# for each partition trains a decision tree.
			for idx in slice_idx:
				clf = ProjectedDecisionTreeClassifier(random_state=self.random_state,max_leaf_nodes=self.max_leaf_nodes)
				clf.fit(X,y,idx)

				# verify that the tree contains at least one features.
				if len(clf.tree_.feature) > 1:
					forest.append(clf)

			# verifies that a sub-forest of '2b + 1' trees has been created.
			if len(forest) >= min_num_tree:
				predict = np.sum([tr.predict(X_val) for tr in forest],axis=0)
				predict[predict<0] = -1
				predict[predict>0] = 1
				
				acc = np.sum(predict==y_val) / len(y_val)
				# given an 'X_val' dataset if the accuracy of the sub-forest is below the minimum accuracy acceptance threshold, discard the forest.
				if acc > self.min_acc:
					self.forest += forest
					self.forest_counter += 1

		self.n_trees = len(self.forest)

		print('t_size: ',min_num_tree,'#frst: ',self.forest_counter,' #tr: ',self.n_trees)
		# if it does not generate at least one forest it raises an error
		if self.n_trees < 1:
			raise Exception('Error: a robust forest cannot be created with the specified parameters.')

# this class is the implementation of our robust FPF ensemble method.
class HierarchicalFeaturePartitionedForest(FeaturePartitionedForest):
	def __init__(self,b,r=10,n_est=None,min_acc=None,random_state=None,max_leaf_nodes=None):  
		super().__init__(b,r,n_est,min_acc,random_state,max_leaf_nodes)

	def predict(self,X):
		self.check_fit()
		predict_sum = np.zeros((X.shape[0],))
		rounded_forest = np.array_split(self.forest, self.forest_counter)

		for forest in rounded_forest:
			predict = np.sum([tr.predict(X) for tr in forest],axis=0)
			predict[predict<0] = -1
			predict[predict>0] = 1
			predict_sum += predict

		predict_sum[predict_sum<0] = -1
		predict_sum[predict_sum>0] = 1
		predict_sum[predict_sum==0] = self.max_label
		return predict_sum

# this class is the implementation of RSM ensemble method.
class RandomSubspaceMethod(ProjectedForest):
	def __init__(self,p=.2,n_trees=1,random_state=None,max_leaf_nodes=None):  
		super().__init__()
		self.forest = []
		self.p = p	
		self.n_trees = n_trees
		self.random_state = random_state
		self.max_leaf_nodes = max_leaf_nodes
		np.random.seed(seed=random_state)

	def fit(self,X,y):
		super().fit(X,y)
		_, n_feat = np.shape(X)
		idx_f = np.arange(n_feat)
		ft_sb_size = int(self.p * n_feat)

		if ft_sb_size < 1 or ft_sb_size > n_feat:
			print('Error: parameter "p" must be in [0,1].')

		self.forest_counter = 0
		for _ in range(self.n_trees):
			idx_f = np.copy(idx_f)
			# shuffles feature indexes.
			np.random.shuffle(idx_f)
			# performs boostrap sampling.
			slice_idx = idx_f[:ft_sb_size]
			# trains a tree with the features sample
			clf = ProjectedDecisionTreeClassifier(random_state=self.random_state,max_leaf_nodes=self.max_leaf_nodes)
			clf.fit(X,y,slice_idx)

			# verify that the tree contains at least one features.
			if len(clf.tree_.feature) > 1:
				self.forest.append(clf)