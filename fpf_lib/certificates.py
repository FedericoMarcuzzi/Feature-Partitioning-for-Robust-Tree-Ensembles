'''
certificates.py

Created by 

name: Federico Marcuzzi
e-mail: federico.marcuzzi@unive.it

date 20/02/2020
'''

import numpy as np
from itertools import combinations



'''
 FUNCTION -> : 'max_damage' computes the maximum damage matrix for fast_lower_bound and fast_lower_bound_hierarchical.
  *INPUT  -> dmg : a damage matrix of size n_ist X (2 * n_feat) w.r.t. a left attack or a right attack to a feature (f+ and f- path).
  *OUTPUT -> : a damage matrix summarising the maximum attack between f+ and f- path.
'''
def max_damage(dmg):
	_, n_feat_p2 = dmg.shape
	n_feat = n_feat_p2 // 2
	max_dmg = np.maximum(dmg[:, :n_feat], dmg[:, n_feat:n_feat_p2])
	return max_dmg


'''
 FUNCTION -> : 'decision_path_sets' computes the decision path sets of a given model.
  *INPUT  -> model : tree to inspect
  *INPUT  -> X : set of original instances
  *INPUT  -> check_input : 'decision_path' parameter (from 'scikit-learn: Allow to bypass several input checking. Don’t use this parameter unless you know what you do.')
  *OUTPUT -> : a one/zero matrix n_ist X (2 * n_feat). 0 in (x,f) if the tree use feature f to predict instance x. 1 otherwise.(f+ and f- path)
'''
def decision_path_sets(tr,X,err=None):
	n_ist, n_feat = X.shape
	feature = tr.tree_.feature
	threshold = tr.tree_.threshold
	akt_f = np.ones((n_ist, n_feat*2))

	path = tr.decision_path(X)
	arr = path.toarray()

	for i in np.arange(n_ist):
		mask = arr[i]==1
		for f, v in zip(feature[mask], threshold[mask]):
			if f >= 0:
				if X[i, f] <= v:
					akt_f[i, f] = 0
				else:
					akt_f[i, f + n_feat] = 0

	if err is not None:
		akt_f = np.logical_not(akt_f) * err.reshape(-1, 1)

	return akt_f


### CERTIFICATI FLAT
'''
 FUNCTION -> : 'accurate_lower_bound' calculate a slow but more accurate lower-bound of the model.
  *INPUT  -> model : forest to be certified
  *INPUT  -> X : set of original instances
  *INPUT  -> y : instances labels
  *INPUT  -> k_start : minimum attacker' budget
  *INPUT  -> k_end : maximum attacker' budget
  *OUTPUT -> : for each attack it returns the indexes of the instances attacked
'''
def accurate_lower_bound(model, X, y, k_start, k_end=None, maj_clss=None):
	if k_end == None:
		k_end = k_start

	n_ist, n_feat = X.shape
	d_n_feat = 2 * n_feat

	forest = [tr for tr in model]
	n_trees = len(forest)

	half_forest = n_trees // 2 + 1

	predict = np.transpose([tr.predict(X) for tr in model])
	init = predict==y.reshape(n_ist, 1)

	stacked = np.hstack([decision_path_sets(tr, X).T.reshape(n_ist * d_n_feat, 1) for tr in model])
	splitted = np.asarray(np.array_split(stacked, d_n_feat, axis=0))

	dict_broken = {}
	for k in np.arange(k_start, k_end+1):
		safe_inst = np.ones(n_ist)
		if k > 0:
			for idx_f in combinations(np.arange(d_n_feat), k):
				idx_f = np.asarray(idx_f)

				check_v = np.zeros((d_n_feat,1))
				check_v[idx_f] = 1
				check_v = np.array_split(check_v, 2, axis=0)
				check_v = check_v[0] * check_v[1]

				if np.sum(check_v) == 0:
					out = np.copy(init) * np.prod(splitted[idx_f], axis=0)
					summarized = np.sum(out, axis=1)

					# maj. clss. test
					if maj_clss is not None and n_trees % 2 == 0:
						idx_eq = np.where(summarized == n_trees // 2)
						summarized[idx_eq] += (y[idx_eq]==maj_clss)

					safe_inst *= (summarized >= half_forest)

		dict_broken[k] = np.where(safe_inst!=1)[0]

	return dict_broken

'''
 FUNCTION -> : 'fast_lower_bound' calculate a pessimistic but fast lower-bound of the robustness of the model.
  *INPUT  -> model : forest to be certified
  *INPUT  -> X : set of original instances
  *INPUT  -> y : instances labels
  *INPUT  -> k_start : minimum attacker' budget
  *INPUT  -> k_end : maximum attacker' budget
  *OUTPUT -> : for each attack it returns the indexes of the instances attacked
'''
def fast_lower_bound(model, X, y, k_start, k_end=None, maj_clss=None):
	if k_end == None:
		k_end = k_start

	n_ist, n_feat = X.shape

	forest = [tr for tr in model]
	n_trees = len(forest)

	half_forest = n_trees // 2 if n_trees % 2 == 0 else n_trees // 2 + 1

	predict = np.transpose([tr.predict(X) for tr in forest])
	init = (predict==y.reshape(n_ist, 1))
	error = np.sum(np.logical_not(init), axis=1)
	summarized = np.sum([decision_path_sets(tr, X, err) for tr, err in zip(model, init.T)], axis=0)

	max_dmg = max_damage(summarized)
	max_dmg_sorted = np.sort(max_dmg, axis=1)
	dict_broken = {}
	for k in np.arange(k_start, k_end+1):
		idx_atk = np.arange(n_feat-k, n_feat)
		atked = np.sum(max_dmg_sorted[:,idx_atk], axis=1) + error

		# maj. clss. test
		if maj_clss is not None and n_trees % 2 == 0:
			idx_eq = np.where(atked == n_trees // 2)
			atked[idx_eq] -= (y[idx_eq]==maj_clss)

		dict_broken[k] = np.where(atked >= half_forest)[0]

	return dict_broken



### CERTIFICATI HIERARCHICAL
'''
 FUNCTION -> : 'accurate_lower_bound_hierarchical' calculate a slow but more accurate lower-bound of the hierarchical model.
  *INPUT  -> model : forest to be certified
  *INPUT  -> X : set of original instances
  *INPUT  -> y : instances labels
  *INPUT  -> k_start : minimum attacker' budget
  *INPUT  -> k_end : maximum attacker' budget
  *OUTPUT -> : for each attack it returns the indexes of the instances attacked
'''
def accurate_lower_bound_hierarchical(model, X, y, k_start, k_end=None, maj_clss=None):
	if k_end == None:
		k_end = k_start

	n_ist, n_feat = X.shape
	d_n_feat = 2 * n_feat

	forest_counter = model.forest_counter
	forest = model.forest

	half_slot = forest_counter // 2 + 1
	b = model.b

	predict = np.transpose([tr.predict(X) for tr in forest])
	init = predict==y.reshape(n_ist, 1)

	stacked = np.hstack([decision_path_sets(tr, X).T.reshape(n_ist * d_n_feat, 1) for tr in model])
	splitted = np.asarray(np.array_split(stacked, d_n_feat, axis=0))

	dict_broken = {}
	for k in np.arange(k_start, k_end+1):
		safe_inst = np.ones(n_ist)
		for idx_f in combinations(np.arange(d_n_feat), k):
			idx_f = np.asarray(idx_f)

			check_v = np.zeros((d_n_feat,1))
			check_v[idx_f] = 1
			check_v = np.array_split(check_v, 2, axis=0)
			check_v = check_v[0] * check_v[1]

			if np.sum(check_v) == 0:
				out = np.copy(init) * np.prod(splitted[idx_f], axis=0)
				splitted_out = np.asarray(np.array_split(out, forest_counter, axis=1))
				summarized = np.sum(splitted_out, axis=2)
				summarized = np.sum(summarized >= b+1, axis=0)

				# maj. clss. test
				if maj_clss is not None and forest_counter % 2 == 0:
					idx_eq = np.where(summarized == forest_counter // 2)[0]
					summarized[idx_eq] += (y[idx_eq]==maj_clss)

				safe_inst *= (summarized >= half_slot)

		dict_broken[k] = np.where(safe_inst!=1)[0]

	return dict_broken

'''
 FUNCTION -> : 'fast_lower_bound_hierarchical' calculate a pessimistic but fast lower-bound of the robustness of the hierarchical model.
  *INPUT  -> model : forest to be certified
  *INPUT  -> X : set of original instances
  *INPUT  -> y : instances labels
  *INPUT  -> k_start : minimum attacker' budget
  *INPUT  -> k_end : maximum attacker' budget
  *OUTPUT -> : for each attack it returns the indexes of the instances attacked
'''
def fast_lower_bound_hierarchical(model, X, y, k_start, k_end=None, maj_clss=None):
	if k_end == None:
		k_end = k_start

	forest_counter = model.forest_counter
	forest = model.forest
	b = model.b
	half_slot = forest_counter // 2 + 1

	predict = np.asarray([tr.predict(X) for tr in forest])
	predict = predict==y

	slot_prd = np.asarray(np.array_split(predict, forest_counter, axis=0))
	prd_pr_slot = np.sum(slot_prd, axis=1).T - b
	prd_pr_slot[prd_pr_slot<0] = 0
	prd_pr_slot = np.sort(prd_pr_slot, axis=1)

	summarized = np.sum([decision_path_sets(tr, X, err) for tr, err in zip(model, predict)], axis=0)
	max_dmg = max_damage(summarized)
	max_dmg_sorted = np.sort(max_dmg, axis=1)
	max_dmg_sorted = np.flip(max_dmg_sorted, axis=1)

	dict_broken = {}
	for k in np.arange(k_end):
		for j in np.arange(forest_counter):
			m1 = np.minimum(prd_pr_slot[:,j], max_dmg_sorted[:,k])
			m2 = np.minimum(m1, 1)
			prd_pr_slot[:,j] -= m2
			prd_pr_slot[prd_pr_slot[:,k]<0,k] = 0
			max_dmg_sorted[max_dmg_sorted[:,k]>0,k] -= 1

		if k >= k_start - 1:
			prd = prd_pr_slot>0
			prd = np.sum(prd, axis=1)

			if maj_clss is not None and forest_counter % 2 == 0:
				idx_eq = np.where(prd == forest_counter // 2)[0]
				prd[idx_eq] += (y[idx_eq]==maj_clss)

			dict_broken[k+1] = np.where(prd < half_slot)[0]

	return dict_broken