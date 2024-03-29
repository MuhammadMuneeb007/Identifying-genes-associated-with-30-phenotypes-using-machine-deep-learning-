from __future__ import absolute_import, division, print_function
import pandas as pd
import numpy as np
import os
import shutil
import re 
import glob
import sys
from sklearn.model_selection import train_test_split
import sys
import argparse 
import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from numpy import genfromtxt
from sklearn import svm 
from numpy import genfromtxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')
 
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
scaler = StandardScaler()
from sklearn.metrics import precision_score, recall_score, accuracy_score 
from sklearn import metrics
from sklearn import preprocessing 
import seaborn as sn
import matplotlib.pyplot as plt
from pylab import rcParams
import sys
import os
from sklearn import tree, ensemble
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import os
import re
import pandas as pd
import numpy as np
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#from sklearn.inspection import DecisionBoundaryDisplay
np.seterr(all="ignore")
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import sys
import re
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from yellowbrick.model_selection import feature_importances
import inspect
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import *
from sklearn import *
from sklearn.linear_model import *
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import *
from sklearn.neighbors import *
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def sorted_nicely( l ):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)

def traintestf1score(model,x_train, y_train,x_test,y_test):
 a = int( (f1_score(y_train, [np.round(value) for value in model.predict(x_train)]) * 100.0))
 b = confusion_matrix(y_train, model.predict(x_train))
 c = int((f1_score(y_test, [np.round(value) for value in model.predict(x_test)]) * 100.0))
 d = confusion_matrix(y_test, model.predict(x_test))
 return a,b,c,d

 a = int( (recall_score(y_train, [np.round(value) for value in model.predict(x_train)]) * 100.0))
 b = confusion_matrix(y_train, model.predict(x_train))
 c = int((recall_score(y_test, [np.round(value) for value in model.predict(x_test)]) * 100.0))
 d = confusion_matrix(y_test, model.predict(x_test))

 return a,b,c,d
def traintestAUC(model,x_train, y_train,x_test,y_test):
 a = int( (roc_auc_score(y_train, [np.round(value) for value in model.predict(x_train)]) * 100.0))
 b = confusion_matrix(y_train, model.predict(x_train))
 c = int((roc_auc_score(y_test, [np.round(value) for value in model.predict(x_test)]) * 100.0))
 d = confusion_matrix(y_test, model.predict(x_test))
 return a,b,c,d
def traintestMCC(model,x_train, y_train,x_test,y_test):
 a = int( (matthews_corrcoef(y_train, [np.round(value) for value in model.predict(x_train)]) * 100.0))
 b = confusion_matrix(y_train, model.predict(x_train))
 c = int((matthews_corrcoef(y_test, [np.round(value) for value in model.predict(x_test)]) * 100.0))
 d = confusion_matrix(y_test, model.predict(x_test))
 return a,b,c,d


def getmefeatureimportance(mod,name,pvalue):
    try:    
        saveimportance_1(mod,name,pvalue)
    except:
        try:
            saveimportance_2(mod,name,pvalue)
        except:
            try:
                saveimportance_3(mod,name,pvalue)
            except:
                try:
                    saveimportance_4(mod,name,pvalue)
                except:
                    saveimportance_5(mod,name,pvalue)

def saveimportance_1(mod,name,pvalue):
    importance = mod.feature_importances_
    temp = pd.DataFrame()
    temp['Features_importance'] =  importance
    print(name,"1")
    temp.to_csv(sys.argv[1]+os.sep+sys.argv[2]+os.sep+pvalue+os.sep+name+".csv")

def saveimportance_2(mod,name,pvalue):
    feature_importances = np.mean([tree.feature_importances_ for tree in mod.estimators_], axis=0)
    temp = pd.DataFrame()
    temp['Features_importance'] =  feature_importances
    print(name,"2")
    
    temp.to_csv(sys.argv[1]+os.sep+sys.argv[2]+os.sep+pvalue+os.sep+name+".csv")    
 
def saveimportance_3(mod,name,pvalue):
    importance = mod.best_estimator_.feature_importances_
    temp = pd.DataFrame()
    temp['Features_importance'] =  importance
    print(name,"3")
    
    temp.to_csv(sys.argv[1]+os.sep+sys.argv[2]+os.sep+pvalue+os.sep+name+".csv")

def saveimportance_4(mod,name,pvalue):
    importance = mod.coef_[0]
    temp = pd.DataFrame()
    temp['Features_importance'] =  importance
    print(name,"4")
    
    temp.to_csv(sys.argv[1]+os.sep+sys.argv[2]+os.sep+pvalue+os.sep+name+".csv")

def saveimportance_5(mod,name,pvalue):
    importance = mod.ranking_
    temp = pd.DataFrame()
    temp['Features_importance'] =  importance
    print(name,"5")
    
    temp.to_csv(sys.argv[1]+os.sep+sys.argv[2]+os.sep+pvalue+os.sep+name+".csv")


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np, sequentia as seq

allmodels = ['AdaBoostClassifier','BaggingClassifier','ExtraTreesClassifier','GradientBoostingClassifier',
'RandomForestClassifier','PassiveAggressiveClassifier'
,'RidgeClassifier','SGDClassifier','BernoulliNB','LogisticRegression','DecisionTreeClassifier']



pheno = sys.argv[1]
iteration = sys.argv[2]
pvalues = os.listdir(sys.argv[1]+os.sep+str(iteration))

results = {}
results2 = {}
results3 = {}
results4 = {}
results5 = {}

count=1
for pvalue in pvalues:
    if "pv_" in pvalue:
        iterationdirec = pheno+os.sep+iteration
        datadirec = pheno +os.sep + iteration + os.sep + pvalue 
        
        x_train = pd.read_csv("./"+datadirec+os.sep+'ptrain.raw', sep="\s+")
        x_test = pd.read_csv("./"+datadirec+os.sep+'ptest.raw', sep="\s+")
        x_train.replace([np.inf, -np.inf], np.nan, inplace=True)
        x_test.replace([np.inf, -np.inf], np.nan, inplace=True)

        x_train = x_train.fillna(0)
        x_test = x_test.fillna(0)

        x_train =x_train.iloc[:,6:].values
        x_test  =x_test.iloc[:,6:].values

        results["SNPs:"+str(x_train.shape[1])] = []
        results2["SNPs:"+str(x_train.shape[1])] = []
        results3["SNPs:"+str(x_train.shape[1])] = []
        results4["SNPs:"+str(x_train.shape[1])] = []
        results5["SNPs:"+str(x_train.shape[1])] = []
        
        scaler = StandardScaler()
        std_scale = preprocessing.StandardScaler().fit(x_train)
        x_train = std_scale.transform(x_train)
        x_test = std_scale.transform(x_test)
        y_train  = pd.read_csv(iterationdirec+os.sep+'train/train.fam', sep="\s+",header=None,names=["a","b","c","d","e","f"])
        y_test= pd.read_csv(iterationdirec+os.sep+'test/test.fam', sep="\s+",header=None,names=["a","b","c","d","e","f"])
        
        y_train.f[y_train['f']==1]=0
        y_train.f[y_train['f']==2]=1
        y_test.f[y_test['f']==1]=0
        y_test.f[y_test['f']==2]=1

        y_train = y_train['f'].values
        y_test = y_test['f'].values

        sample_weights = compute_sample_weight(
        class_weight='balanced',
        y=y_train
        )

        for model in allmodels:
            try:
                mod = eval(model)().fit(x_train, y_train, sample_weight=sample_weights)
                getmefeatureimportance(mod,model,pvalue)

                a,b,c,d =traintestAUC(mod,x_train, y_train,x_test,y_test) 
                results["SNPs:"+str(x_train.shape[1])].append(str(a)+"/"+str(c))

                a,b,c,d =traintestMCC(mod,x_train, y_train,x_test,y_test) 
                results2["SNPs:"+str(x_train.shape[1])].append(str(a)+"/"+str(c))

                a,b,c,d =traintestf1score(mod,x_train, y_train,x_test,y_test) 
                results3["SNPs:"+str(x_train.shape[1])].append(str(a)+"/"+str(c))   
                             
            except:
                mod = eval(model)().fit(x_train, y_train)
                getmefeatureimportance(mod,model,pvalue)


                a,b,c,d =traintestAUC(mod,x_train, y_train,x_test,y_test) 
                results["SNPs:"+str(x_train.shape[1])].append(str(a)+"/"+str(c))

                a,b,c,d =traintestMCC(mod,x_train, y_train,x_test,y_test) 
                results2["SNPs:"+str(x_train.shape[1])].append(str(a)+"/"+str(c))

                a,b,c,d =traintestf1score(mod,x_train, y_train,x_test,y_test) 
                results3["SNPs:"+str(x_train.shape[1])].append(str(a)+"/"+str(c))


        for kern in ['linear']:
            mod = SVC(gamma='auto',kernel=kern).fit(x_train, y_train, sample_weight=sample_weights)
            getmefeatureimportance(mod,"SGDClassifier",pvalue)
            a,b,c,d =traintestAUC(mod,x_train, y_train,x_test,y_test) 
            results["SNPs:"+str(x_train.shape[1])].append(str(a)+"/"+str(c))

            a,b,c,d =traintestMCC(mod,x_train, y_train,x_test,y_test) 
            results2["SNPs:"+str(x_train.shape[1])].append(str(a)+"/"+str(c))
            
            a,b,c,d =traintestf1score(mod,x_train, y_train,x_test,y_test) 
            results3["SNPs:"+str(x_train.shape[1])].append(str(a)+"/"+str(c))



                        
        booster = ['gblinear','gbtree','dart']
        lossfunction = ['binary:hinge','binary:logistic','binary:logitraw']


        for x1 in booster:
            for x2 in lossfunction:
                xgb = XGBClassifier(booster=x1,objective=x2,eval_metric="auc").fit(x_train, y_train)
                getmefeatureimportance(xgb,"Xgboost-Booster-"+x1+"-Lossfunction-"+x2.replace(":","-"),pvalue)
                a,b,c,d =traintestAUC(xgb,x_train, y_train,x_test,y_test) 
                results["SNPs:"+str(x_train.shape[1])].append(str(a)+"/"+str(c))

                a,b,c,d =traintestMCC(mod,x_train, y_train,x_test,y_test) 
                results2["SNPs:"+str(x_train.shape[1])].append(str(a)+"/"+str(c))       

                a,b,c,d =traintestf1score(mod,x_train, y_train,x_test,y_test) 
                results3["SNPs:"+str(x_train.shape[1])].append(str(a)+"/"+str(c))

        L = pd.DataFrame(results)
        L.to_csv(pheno+os.sep+iteration+os.sep+"Results_MachineLearning_AUC.csv",index=False, sep="\t")

        L = pd.DataFrame(results2)
        L.to_csv(pheno+os.sep+iteration+os.sep+"Results_MachineLearning_MCC.csv",index=False, sep="\t")

        L = pd.DataFrame(results3)
        L.to_csv(pheno+os.sep+iteration+os.sep+"Results_MachineLearning_f1score.csv",index=False, sep="\t")
        
        
        print(pd.DataFrame(results).to_markdown()) 
