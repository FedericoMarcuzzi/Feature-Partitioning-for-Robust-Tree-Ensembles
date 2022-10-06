'''
test.py

Created by 

name: Federico Marcuzzi
e-mail: federico.marcuzzi@unive.it

date 20/02/2020
'''


import argparse
from joblib import load
from sklearn.model_selection import train_test_split
from fpf_lib.models import *
from fpf_lib.misc import SEED, load_dataset, normalize
from fpf_lib.certificates import fast_lower_bound, fast_lower_bound_hierarchical, accurate_lower_bound, accurate_lower_bound_hierarchical

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", dest = "dataset", help="Dataset name") # bc, sb, wn
parser.add_argument("-df", "--dataset_folder", dest = "dataset_folder", help="Dataset folder")
parser.add_argument("-m", "--model_path", dest = "model_path", help="Model path")
parser.add_argument("-a", "--algo", dest = "algo", help="Learning algorithm") # ffpf, hfpf, rf, rsm
parser.add_argument("-k", "--k_param", dest = "k", help="Attacker budget", type=int, default=0)
parser.add_argument("-j", "--n_threads", dest = "n_ths", help="Attacker budget", type=int, default=0)

args = parser.parse_args()

# INIT SETS
X, y = load_dataset(args.dataset_folder, args.dataset)
X = normalize(X)

X_train, X_valid_test, y_train, y_valid_test = train_test_split(X, y, train_size=0.6, random_state=SEED, shuffle=True, stratify=y)
X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test, train_size=0.5, random_state=SEED, shuffle=True, stratify=y_valid_test)

# DATASET ATTRIBUTES
n_ist = y.shape[0]
maj_clss = Counter(y_train).most_common()[0][0]

# LOAD MODEL
clf = load(args.model_path)

# COMPUTE ACCURACIES
fast_lb_algo = fast_lower_bound_hierarchical if args.algo == "hfpf" else fast_lower_bound
accurate_lb_algo = accurate_lower_bound_hierarchical if args.algo == "hfpf" else accurate_lower_bound

fast_broken = fast_lb_algo(clf, X, y, 0, args.k, maj_clss)
accurate_broken = fast_lb_algo(clf, X, y, 0, args.k, maj_clss)

print("Fast-Lower-Bound accuracies: ", end="")
for val in fast_broken.values():
    print(1 - val.shape[0] / n_ist, end=" ")

print()

print("Accurate-Lower-Bound accuracies: ", end="")
for val in accurate_broken.values():
    print(1 - val.shape[0] / n_ist, end=" ")

print()