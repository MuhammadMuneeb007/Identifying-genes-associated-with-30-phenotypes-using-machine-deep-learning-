from __future__ import absolute_import, division, print_function
import pandas as pd
import numpy as np
import os
import re
import sys
import warnings
import sklearn.base
warnings.filterwarnings("ignore")
np.seterr(all="ignore")

from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import (confusion_matrix, roc_auc_score,
                             matthews_corrcoef, f1_score)
from sklearn.utils.class_weight import compute_sample_weight

# ── Model imports ─────────────────────────────────────────────────────────────
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                               ExtraTreesClassifier, GradientBoostingClassifier,
                               RandomForestClassifier, HistGradientBoostingClassifier)
from sklearn.linear_model  import (PassiveAggressiveClassifier, RidgeClassifier,
                                   SGDClassifier, LogisticRegression)
from sklearn.naive_bayes   import BernoulliNB, GaussianNB
from sklearn.tree          import DecisionTreeClassifier
from sklearn.svm           import SVC
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier

# ── Model registry ────────────────────────────────────────────────────────────
# Matches exactly the models used in the original paper:
# 16 sklearn models + 1 SVC linear + 9 XGBoost variants = 26 total
MODEL_REGISTRY = {}

# ── Sklearn models (same as original allmodels list) ─────────────────────────
sklearn_models = [
    ("AdaBoostClassifier",             AdaBoostClassifier()),
    ("BaggingClassifier",              BaggingClassifier()),
    ("ExtraTreesClassifier",           ExtraTreesClassifier()),
    ("GradientBoostingClassifier",     GradientBoostingClassifier()),
    ("RandomForestClassifier",         RandomForestClassifier()),
    ("HistGradientBoostingClassifier", HistGradientBoostingClassifier()),
    ("PassiveAggressiveClassifier",    PassiveAggressiveClassifier()),
    ("RidgeClassifier",                RidgeClassifier()),
    ("SGDClassifier",                  SGDClassifier()),
    ("BernoulliNB",                    BernoulliNB()),
    ("GaussianNB",                     GaussianNB()),
    ("KNeighborsClassifier",           KNeighborsClassifier()),
    ("LogisticRegression",             LogisticRegression(max_iter=1000)),
    ("MLPClassifier",                  MLPClassifier(max_iter=500)),
    ("DecisionTreeClassifier",         DecisionTreeClassifier()),
]
for name, obj in sklearn_models:
    MODEL_REGISTRY[name] = obj

# ── SVC — 4 kernels (matches original paper) ─────────────────────────────────
for kern in ['linear', 'poly', 'rbf', 'sigmoid']:
    MODEL_REGISTRY[f"SVC_{kern}"] = SVC(gamma="auto", kernel=kern)

# ── XGBoost — 9 variants (3 boosters × 3 loss functions) ─────────────────────
for booster in ['gblinear', 'gbtree', 'dart']:
    for loss in ['binary:hinge', 'binary:logistic', 'binary:logitraw']:
        name = "Xgboost-Booster-{}-Lossfunction-{}".format(
            booster, loss.replace(":", "-"))
        MODEL_REGISTRY[name] = XGBClassifier(
            booster=booster,
            objective=loss,
            eval_metric="auc",
            verbosity=0
        )

# ── Metric helpers ────────────────────────────────────────────────────────────
def get_preds(model, X):
    return [np.round(v) for v in model.predict(X)]

def traintestAUC(model, x_train, y_train, x_test, y_test):
    a = int(roc_auc_score(y_train, get_preds(model, x_train)) * 100)
    c = int(roc_auc_score(y_test,  get_preds(model, x_test))  * 100)
    b = confusion_matrix(y_train,  get_preds(model, x_train))
    d = confusion_matrix(y_test,   get_preds(model, x_test))
    return a, b, c, d

def traintestMCC(model, x_train, y_train, x_test, y_test):
    a = int(matthews_corrcoef(y_train, get_preds(model, x_train)) * 100)
    c = int(matthews_corrcoef(y_test,  get_preds(model, x_test))  * 100)
    b = confusion_matrix(y_train,      get_preds(model, x_train))
    d = confusion_matrix(y_test,       get_preds(model, x_test))
    return a, b, c, d

def traintestF1(model, x_train, y_train, x_test, y_test):
    a = int(f1_score(y_train, get_preds(model, x_train)) * 100)
    c = int(f1_score(y_test,  get_preds(model, x_test))  * 100)
    b = confusion_matrix(y_train, get_preds(model, x_train))
    d = confusion_matrix(y_test,  get_preds(model, x_test))
    return a, b, c, d

# ── Feature importance savers ─────────────────────────────────────────────────
def save_importance(mod, name, out_path):
    def method1(m): return m.feature_importances_
    def method2(m): return np.mean(
        [t.feature_importances_ for t in m.estimators_], axis=0)
    def method3(m): return m.best_estimator_.feature_importances_
    def method4(m): return m.coef_[0]
    def method5(m): return m.ranking_
    def method6(m): return np.abs(
        m.feature_log_prob_[1] - m.feature_log_prob_[0])
    def method7(m): return np.zeros(m.n_features_in_)

    for i, method in enumerate([method1, method2, method3,
                                 method4, method5, method6, method7], 1):
        try:
            importance = method(mod)
            pd.DataFrame({"Features_importance": importance}).to_csv(out_path)
            print(f"    {name} importance method {i}")
            return
        except Exception:
            continue
    print(f"    {name} - all importance methods failed")

# ── Fit helper ────────────────────────────────────────────────────────────────
def fit_model(model, x_train, y_train, sample_weights):
    try:
        return model.fit(x_train, y_train, sample_weight=sample_weights)
    except TypeError:
        return model.fit(x_train, y_train)

# ── Sorted nicely ─────────────────────────────────────────────────────────────
def sorted_nicely(l):
    convert      = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)

# ── Main ──────────────────────────────────────────────────────────────────────
pheno     = sys.argv[1]
iteration = sys.argv[2]
pvalues   = os.listdir(pheno + os.sep + str(iteration))

results_auc = {}
results_mcc = {}
results_f1  = {}

for pvalue in sorted(pvalues):
    if "pv_" not in pvalue:
        continue

    print(f"\n{'='*60}")
    print(f"  Phenotype: {pheno} | Fold: {iteration} | PValue: {pvalue}")
    print(f"{'='*60}")

    iterationdirec = pheno + os.sep + iteration
    datadirec      = pheno + os.sep + iteration + os.sep + pvalue
    pv_out         = pheno + os.sep + iteration + os.sep + pvalue + os.sep

    # ── Load genotype data ────────────────────────────────────────────────────
    x_train = pd.read_csv("./" + datadirec + os.sep + "ptrain.raw", sep=r"\s+")
    x_test  = pd.read_csv("./" + datadirec + os.sep + "ptest.raw",  sep=r"\s+")

    x_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    x_test.replace( [np.inf, -np.inf], np.nan, inplace=True)
    x_train = x_train.fillna(0).iloc[:, 6:].values
    x_test  = x_test.fillna(0).iloc[:,  6:].values

    key = "SNPs:" + str(x_train.shape[1])
    results_auc[key] = []
    results_mcc[key] = []
    results_f1[key]  = []

    # ── Standardize ───────────────────────────────────────────────────────────
    std_scale = preprocessing.StandardScaler().fit(x_train)
    x_train   = std_scale.transform(x_train)
    x_test    = std_scale.transform(x_test)

    # ── Load labels ───────────────────────────────────────────────────────────
    y_train = pd.read_csv(
        iterationdirec + os.sep + "train/train.fam",
        sep=r"\s+", header=None, names=list("abcdef"))
    y_test  = pd.read_csv(
        iterationdirec + os.sep + "test/test.fam",
        sep=r"\s+", header=None, names=list("abcdef"))
 
    
    # ── REPLACE WITH this ─────────────────────────────────────────────────────────
    y_train = y_train["f"].map({1: 0, 2: 1}).values
    y_test  = y_test["f"].map({1: 0, 2: 1}).values
 

    # ── Save fold info ────────────────────────────────────────────────────────
    pd.DataFrame([{
        "fold":           iteration,
        "pvalue":         pvalue,
        "n_train":        len(y_train),
        "n_test":         len(y_test),
        "train_cases":    int(sum(y_train)),
        "train_controls": int(len(y_train) - sum(y_train)),
        "test_cases":     int(sum(y_test)),
        "test_controls":  int(len(y_test) - sum(y_test)),
        "n_snps":         x_train.shape[1]
    }]).to_csv(pv_out + "fold_info.csv", index=False)

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    # ── Run all models ────────────────────────────────────────────────────────
    for model_name, model_obj in MODEL_REGISTRY.items():
        print(f"  Running: {model_name} | SNPs: {x_train.shape[1]}")

        # Fresh instance — no state leakage between pvalue folders
        mod = sklearn.base.clone(model_obj)
        mod = fit_model(mod, x_train, y_train, sample_weights)

        # Save feature importance
        save_importance(mod, model_name, pv_out + model_name + ".csv")

        # ── Evaluate — all three metrics use the SAME model ──────────────────
        a, b, c, d = traintestAUC(mod, x_train, y_train, x_test, y_test)
        results_auc[key].append(str(a) + "/" + str(c))

        a, b, c, d = traintestMCC(mod, x_train, y_train, x_test, y_test)
        results_mcc[key].append(str(a) + "/" + str(c))

        a, b, c, d = traintestF1(mod, x_train, y_train, x_test, y_test)
        results_f1[key].append(str(a) + "/" + str(c))

    # ── Save results after each pvalue folder ─────────────────────────────────
    pd.DataFrame(results_auc).to_csv(
        pheno + os.sep + iteration + os.sep + "Results_MachineLearning_AUC.csv",
        index=False, sep="\t")
    pd.DataFrame(results_mcc).to_csv(
        pheno + os.sep + iteration + os.sep + "Results_MachineLearning_MCC.csv",
        index=False, sep="\t")
    pd.DataFrame(results_f1).to_csv(
        pheno + os.sep + iteration + os.sep + "Results_MachineLearning_f1score.csv",
        index=False, sep="\t")

    print(pd.DataFrame(results_auc).to_markdown())
