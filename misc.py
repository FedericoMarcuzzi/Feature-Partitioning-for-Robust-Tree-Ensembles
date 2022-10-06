from sklearn.datasets import load_breast_cancer, load_svmlight_file
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import os
import dill
import numpy as np
import pandas as pd
import json

def load_wine(th=4):
    dataset = pd.read_csv("wine_quality_white.csv")
    data = dataset.to_numpy()
    X = data[:, : -1]
    y = data[:, -1]
    
    y[y<th] = -1
    y[y>=th] = 1
    return X, y

def load_diabetes():
    dataset = pd.read_csv("../diabetes/diabetes.csv")
    data = dataset.to_numpy()
    X = data[:, : -1]
    y = data[:, -1]
    
    y[y == 0] = -1
    return X, y

def load_ijcnn():
    dataset_train = load_svmlight_file("../ijcnn/ijcnn1.tr")
    dataset_test = load_svmlight_file("../ijcnn/ijcnn1.t")
    X_train = dataset_train[0].toarray()
    X_test = dataset_test[0].toarray()
    y_train = dataset_train[1]
    y_test = dataset_test[1]
    
    return X_train, X_test, y_train, y_test

def load_covtype():
    dataset = load_svmlight_file("../covtype/covtype.libsvm.binary.scale")
    X = dataset[0].toarray()
    y = dataset[1]
    y[y==2] = -1
    
    return X, y

def normalize(X):
    X = X - X.min(axis=0)
    X = X/X.max(axis=0)
    X = np.nan_to_num(X)
    return X

def str_to_json(ensamble, dim, lab=[-1,1]):
    n_trees = len(ensamble)
    i = 0
    str_tree = "\t" + str(dim) + ",\n\t["
    for l in lab:
        str_tree += str(l) + ", "
    
    str_tree = str_tree[:-2] + "],\n\t[\n"
    
    for dict_tree in ensamble:
        app = np.array(dict_tree['feature'])
        app+=1
        app[app<0] = 0
        dict_tree['feature'] = app
        
        app = np.array(dict_tree['children_left'])
        app+=1
        app[app<0] = 0
        dict_tree['children_left'] = app
        
        app = np.array(dict_tree['children_right'])
        app+=1
        app[app<0] = 0
        dict_tree['children_right'] = app
        
        str_f = '\t\t\t[' + ', '.join([str(v) for v in dict_tree['feature']]) + '],\n'
        str_t = '\t\t\t[' + ', '.join([str(v) for v in dict_tree['threshold']]) + '],\n'
        str_l = '\t\t\t[' + ', '.join([str(v) for v in dict_tree['children_left']]) + '],\n'
        str_r = '\t\t\t[' + ', '.join([str(v) for v in dict_tree['children_right']]) + '],\n'
        str_p = '\t\t\t[' + ', '.join([str(v) for v in dict_tree['prediction']]) + ']\n'
        
        i+=1
        if i < n_trees:
            str_tree += '\t\t[\n' + str_f + str_t + str_l + str_r + str_p + '\t\t],\n'
        else:
            str_tree += '\t\t[\n' + str_f + str_t + str_l + str_r + str_p + '\t\t]'
        
    str_ensamble = '[\n' + str_tree + '\n\t]\n]'
    print(str_ensamble)
    return str_ensamble

def _rec_(tree, dict_tree, idx_n):
    #print(tree)
    if 'split_index' in tree:
        dict_tree['feature'].append(tree['split_feature'])
        dict_tree['threshold'].append(tree['threshold'])
        dict_tree['children_left'].append(idx_n+1)
        dict_tree['children_right'].append(idx_n+2)
        dict_tree['prediction'].append(0)
        
        idx_n = _rec_(tree['left_child'], dict_tree, idx_n + 2)
        idx_n = _rec_(tree['right_child'], dict_tree, idx_n)
        return idx_n;
    else:
        dict_tree['feature'].append(-2)
        dict_tree['threshold'].append(-2)
        dict_tree['children_left'].append(-2)
        dict_tree['children_right'].append(-2)
        dict_tree['prediction'].append(tree['leaf_value'])
        return idx_n + 1

def lgbm_to_json(json_lightgbm, dim, filename):
    ensamble = []
    trees = json_lightgbm['tree_info']
    for tr in trees:
        dict_tree = {'feature' : [], 'threshold' : [], 'children_left' : [], 'children_right' : [], 'prediction' : []}
        _rec_(tr['tree_structure'], dict_tree, 0)
        labels = np.asarray(dict_tree['prediction'])
        labels[labels>0] = 1
        labels[labels<0] = -1
        dict_tree['prediction'] = labels.astype(int)
        ensamble.append(dict_tree)

    with open(filename, 'w') as f:
        f.write(str_to_json(ensamble, dim))

# prende in unput una lista di alberi
def sklearn_to_json(trees, dim, filename):
    ensamble = []
    classes = np.asarray([-1, 1])
    n_trees = len(trees)
    i = 0
    str_tree = ""
    for tr in trees:
        dict_tree = {}
        n_nodes = len(tr.tree_.value)
        n_labels = len(tr.tree_.value[0][0])   
        value = tr.tree_.value.reshape(n_nodes, n_labels)
        prediction = classes[np.argmax(value, axis=1)]
        
        dict_tree['feature'] = tr.tree_.feature
        dict_tree['threshold'] = tr.tree_.threshold
        dict_tree['children_left'] = tr.tree_.children_left
        dict_tree['children_right'] = tr.tree_.children_right
        dict_tree['prediction'] = prediction
        ensamble.append(dict_tree)

    with open(filename, 'w') as f:
        f.write(str_to_json(ensamble, dim))

# saves dataset in json format
def dataset_json(data, labels, filename):
    lbs = labels.tolist()
    dt = data.tolist()
    l = [lbs, dt]
    with open(filename, 'w') as f:
        f.write(json.dumps(l))

def rec_visit(node, dict_tree):
    input_id = len(dict_tree["feature"])

    '''
    if node!=None:
        print("START TREE")
        print(node.prediction)
        print(node.best_split_feature_id)
        print(node.best_split_feature_value)
        print(node.is_leaf())
        print("END TREE")
        print("FIGLI")
    '''

    if node!=None:
        print("FILL TREE ####", input_id, "#############")
        if node.is_leaf():
            dict_tree["feature"].append(-1)
            dict_tree["threshold"].append(-2)
            p = node.prediction
            p = -1 if p == 0 else 1
            dict_tree["prediction"].append(p)

            return input_id, [-1], [-1]
        else:
            
            dict_tree["feature"].append(node.best_split_feature_id)
            dict_tree["threshold"].append(node.best_split_feature_value)
            dict_tree["prediction"].append(0)

            node_id_l, l1l, l1r = rec_visit(node.left, dict_tree)
            node_id_r, l2l, l2r = rec_visit(node.right, dict_tree)

            return input_id, [node_id_l] + l1l + l2l, [node_id_r] + l1r + l2r

# prende in unput una lista di alberi
def treant_to_json(filename, dim, filename_out):
    with open(filename, 'rb') as model_file:
        model = dill.load(model_file)

    ensamble = []
    for tr in model.estimators:
        dict_tree = {"feature" : [], "threshold" : [], "children_left" : [], "children_right" : [], "prediction" : []}
        _, ltl, ltr = rec_visit(tr.root, dict_tree)

        dict_tree["children_left"] = ltl
        dict_tree["children_right"] = ltr

        print(dict_tree["feature"])
        print(dict_tree["threshold"])
        print(dict_tree["children_left"])
        print(dict_tree["children_right"])
        print(dict_tree["prediction"])
        ensamble.append(dict_tree)

    with open(filename_out, 'w') as f:
        f.write(str_to_json(ensamble, dim))

# prende in unput una lista di alberi
def icml2019_to_json(filename, dim, filename_out):
    with open(filename, 'rb') as model_file:
        model = dill.load(model_file)

    ensamble = []
    for tr in model:
        print(tr)
        dict_tree = {"feature" : [], "threshold" : [], "children_left" : [], "children_right" : [], "prediction" : []}
        _, ltl, ltr = rec_visit(tr.root, dict_tree)

        dict_tree["children_left"] = ltl
        dict_tree["children_right"] = ltr

        print(dict_tree["feature"])
        print(dict_tree["threshold"])
        print(dict_tree["children_left"])
        print(dict_tree["children_right"])
        print(dict_tree["prediction"])
        ensamble.append(dict_tree)

    with open(filename_out, 'w') as f:
        f.write(str_to_json(ensamble, dim))