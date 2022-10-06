'''
misc.py

Created by 

name: Federico Marcuzzi
e-mail: federico.marcuzzi@unive.it

date 20/02/2020
'''

import numpy as np
import pandas as pd


SEED = 7

# normalizes dataset in [0,1].
def normalize(X):
    X = (X - X.min(0)) / (X.max(0) - X.min(0))
    return np.nan_to_num(X)

def load_dataset(path, dataset):
	X = None
	y = None

	if "mnist" in dataset:
		classes = get_mnist_labels(dataset)
		X, y =  load_mnist(path)
		X, y = prepare_mnist(X, y, cls=classes)
	else:
		dataset_name =  ""
		if dataset == "bc":
			dataset_name = "breast_cancer"
		elif dataset == "sb":
			dataset_name = "spam_base"
		elif dataset == "wn":
			dataset_name = "wine"
		X, y = load_dataset_csv(path + "/" + dataset_name)
	
	unique_y = np.unique(y)
	y[y==unique_y[0]] = -1
	y[y==unique_y[1]] = 1
	return X, y

# loads dataset from csv.
def load_dataset_csv(path):
	data = np.asarray(pd.read_csv(path + ".csv", header=None))
	return data[:,:-1], data[:,-1]

def get_mnist_labels(string):
    string = string[5:]
    return [int(l) for l in string]

def load_mnist(path):
    X = np.load(path + "/mnist/mnist_dataset.npy")
    y = np.load(path + "/mnist/mnist_labels.npy")
    return X.astype(np.uint8), y.astype(np.int8)

def prepare_mnist(X, y, cls=np.arange(10)): 
    cls = np.asarray(cls)
    num_cls = cls.shape[0]
    cls_lab = [-1,1]

    if num_cls < 1:
        raise ValueError('The number of classes must be greater than 0')
        
    idx_cls = np.where(np.isin(y, cls))[0]
    X = X[idx_cls]
    y = y[idx_cls]

    for i in np.arange(num_cls):
        y[y==cls[i]] = cls_lab[i]

    return np.asarray(X), np.asarray(y)