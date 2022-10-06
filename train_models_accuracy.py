'''
test.py

Created by 

name: Federico Marcuzzi
e-mail: federico.marcuzzi@unive.it

date 20/02/2020
'''


import argparse
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from fpf_lib.models import *
from fpf_lib.misc import SEED, load_dataset, normalize

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", dest = "dataset", help="Dataset name") # bc, sb, wn
parser.add_argument("-df", "--dataset_folder", dest = "dataset_folder", help="Dataset folder")
parser.add_argument("-a", "--algo", dest = "algo", help="Learning algorithm") # ffpf, hfpf, rf, rsm
parser.add_argument("-b", "--b_param", dest ="b", help="Model' strength (*-FPF only)", type=int, default=0)
parser.add_argument("-r", "--r_param", dest = "r", help="Rounds number (*-FPF only)", type=int, default=0)
parser.add_argument("-p", "--p_param", dest = "p", help="Projections. size (RSM only)", type=float, default=0)
parser.add_argument("-ml", "--max_leaves", dest = "ml", help="Maximum tree leaves", type=int, default=8)
parser.add_argument("-n", "--num_estimators", dest = "n_est", help="Ensemble size", type=int, default=0)
parser.add_argument("-o", "--output", dest = "output", help="Model output path", default=None)

args = parser.parse_args()

# sanity check
if "fpf" in args.algo and args.b and args.r and args.s:
    print("You can't set b_param, r_param, and n_est at the same time")

# INIT SETS
X, y = load_dataset(args.dataset_folder, args.dataset)
X = normalize(X)

X_train, X_valid_test, y_train, y_valid_test = train_test_split(X, y, train_size=0.6, random_state=SEED, shuffle=True, stratify=y)
X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test, train_size=0.5, random_state=SEED, shuffle=True, stratify=y_valid_test)

clf = None
if args.algo == "ffpf":
    clf = FeaturePartitionedForest(b=args.b, r=args.r, n_est=args.n_est, max_leaf_nodes=args.ml, random_state=SEED)

elif args.algo == "ffpf":
    clf = HierarchicalFeaturePartitionedForest(b=args.b, r=args.r, n_est=args.n_est, max_leaf_nodes=args.ml, random_state=SEED)

elif args.algo == "rsm":
    clf = RandomSubspaceMethod(p=args.p, n_trees=args.n_est, max_leaf_nodes=args.ml, random_state=SEED)

elif args.algo == "rf":
    clf = RandomForestClassifier(n_estimators=args.n_est, max_leaf_nodes=args.ml, random_state=SEED)

if clf is not None:
    clf.fit(X_train,y_train)

    acc_train = clf.score(X_train, y_train)
    acc_valid = clf.score(X_valid, y_valid)
    acc_test = clf.score(X_test, y_test)

    print(f"TRAIN acc: {acc_train}, VALID acc: {acc_valid}, TEST acc: {acc_test}")

    if args.output:
        dump(clf, args.output + ".pkl")
else:
    print("Error occorred in the creation of the model!")