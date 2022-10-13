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
from fpf_lib.certificates import exhaustive_lower_bound, exhaustive_lower_bound_hierarchical
from fpf_lib.attacks_generator import brute_force

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

clf = load(args.model_path)

lb_algo = exhaustive_lower_bound_hierarchical if args.algo == "hfpf" else exhaustive_lower_bound
acc_undr_atk = brute_force(clf, X_test, y_test, args.k, n_th=args.n_ths)

print("Model robustness: ", acc_undr_atk)