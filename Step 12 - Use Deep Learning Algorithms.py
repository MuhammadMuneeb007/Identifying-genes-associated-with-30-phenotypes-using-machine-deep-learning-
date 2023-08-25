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
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, Bidirectional, GRU
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split as split
from sklearn.metrics import median_absolute_error
from tensorflow.keras.optimizers import RMSprop, SGD, Adam , Adadelta
from pandas import concat
from numpy import concatenate
from pandas import DataFrame
from math import e 
import math
import keras
import tensorflow as tf
import math
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


def traintestAUC(model,x_train,x_test, y_train,y_test):
 #print(y_test.shape)
 #print(model.predict(x_test).shape)
 #print(model.predict(x_test))
 
 a = int( (roc_auc_score(y_train, [np.round(value) for value in model.predict(x_train)]) * 100.0))
 c = int((roc_auc_score(y_test, [np.round(value) for value in model.predict(x_test)]) * 100.0))
 #del model,x_train,x_test, y_train,y_test
 #print(a,c)
 return a,c 
 #b = confusion_matrix(y_train, model.predict(x_train))
 #d = confusion_matrix(y_test, model.predict(x_test))
 #return a,b,c,d



def traintestMCC(model,x_train,x_test, y_train,y_test):
 a = int( (matthews_corrcoef(y_train, [np.round(value) for value in model.predict(x_train)]) * 100.0))
 #b = confusion_matrix(y_train, model.predict(x_train))
 
 c = int((matthews_corrcoef(y_test, [np.round(value) for value in model.predict(x_test)]) * 100.0))
 #d = confusion_matrix(y_test, model.predict(x_test))
 #del model,x_train,x_test, y_train,y_test
 return a,c




from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
allmodels = ['AdaBoostClassifier','BaggingClassifier','ExtraTreesClassifier','GradientBoostingClassifier',
'RandomForestClassifier','HistGradientBoostingClassifier','GaussianProcessClassifier','PassiveAggressiveClassifier'
,'RidgeClassifier','SGDClassifier','BernoulliNB','GaussianNB','KNeighborsClassifier','LogisticRegression','MLPClassifier','DecisionTreeClassifier']
import numpy as np, sequentia as seq
from sklearn.utils.class_weight import compute_class_weight
from keras.optimizers import *

results = pd.DataFrame(columns = ['Training AUC','Test AUC'])
results2 = pd.DataFrame(columns = ['Training MCC','Test MCC'])

pheno = sys.argv[1]
iteration = sys.argv[2]

#pvalues = os.listdir(sys.argv[1]+os.sep+iteration)

ddropout = [0.2,0.5]
optimizer = ["Adam"]
batchsize = [1,5]
epochsnumber = [50,200]


import os


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_feature_importance_permutation1(model, x, y, metric,name):

    baseline_score = int( roc_auc_score(y, [np.round(value) for value in model.predict(x)]) * 100.0)
    importance_scores = np.zeros(x.shape[1])

    for feature in range(x.shape[1]):
        x_permuted = x.copy()
        #np.random.shuffle(x_permuted[:, feature])
        x_permuted[:,feature] = 0
        permuted_score = int( (roc_auc_score(y, [np.round(value) for value in model.predict(x_permuted,verbose=0)]) * 100.0))
        importance_scores[feature] = baseline_score - permuted_score
        del x_permuted
        import gc
        gc.collect()
    
    del x
    import gc
    gc.collect()
    importance_scores /= importance_scores.sum()
    print(importance_scores)

    temp = pd.DataFrame()
    temp['Features_importance'] =  importance_scores
    temp.to_csv(sys.argv[1]+os.sep+sys.argv[2]+os.sep+sys.argv[3]+os.sep+name+".csv",index=False,header=False)
    del temp
    gc.collect()

def get_feature_importance_permutation2(model, x, y, metric,name):
    baseline_score = roc_auc_score(y, [np.round(value) for value in model.predict(x)]) 
    print(baseline_score)
    importance_scores = np.zeros(x.shape[2])
    for feature in range(x.shape[2]):
        x_permuted = x.copy()
        #x_permuted = x_permuted.reshape(x.shape[0],x.shape[2])
        #np.random.shuffle(x_permuted[:, feature])
        #x_permuted = x_permuted.reshape(x.shape[0],1,x.shape[2])
        x_permuted[:,:,feature] = 0
        permuted_score = roc_auc_score(y, [np.round(value) for value in model.predict(x_permuted,verbose=0)]) 
        #print(permuted_score)
        importance_scores[feature] = baseline_score - permuted_score
        del x_permuted
        import gc
        gc.collect()

    del x
    import gc
    gc.collect()
    print(importance_scores)
    importance_scores /= importance_scores.sum()
    temp = pd.DataFrame()
    temp['Features_importance'] =  importance_scores
    temp.to_csv(sys.argv[1]+os.sep+sys.argv[2]+os.sep+sys.argv[3]+os.sep+name+".csv",index=False,header=False)
    del temp
    gc.collect()

@profile
def lstmmodel(X_train, X_test, y_train, y_test):
 for drop in ddropout:
  for opt in optimizer:
   for bbb in batchsize:
    for e in epochsnumber:
     try:
      model = keras.models.Sequential()
      model.add(LSTM(128+int(X_train.shape[2]**(1/2)), input_shape=(X_train.shape[1], X_train.shape[2]),return_sequences=True))
      model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
      model.add(LSTM(64+int(X_train.shape[2]**(1/4)), return_sequences=True))
      model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
      model.add(tf.keras.layers.Dropout(drop))
      model.add(LSTM(32+int(X_train.shape[2]**(1/8)), return_sequences=True))
      model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
      model.add(tf.keras.layers.Dropout(drop))
      model.add((LSTM(16+int(X_train.shape[2]**(1/16)),return_sequences=False)))
      model.add(tf.keras.layers.Dropout(drop))
      
      model.add(Dense(1, activation='sigmoid'))
      #optimizer = keras.optimizers.Adam()
      model.compile(loss='binary_crossentropy', metrics=['accuracy'],optimizer=opt)
      w=  compute_class_weight(class_weight = "balanced", classes= np.unique(y_train), y= y_train)
      w = dict(zip(np.unique(y_train), w))
      model.fit(X_train, y_train,epochs=e,batch_size=bbb,validation_split=0.1,shuffle=False,class_weight=w,verbose=0)
      #get_feature_importance_permutation2(model, X_test,y_test,"","LSTM_"+str(int(drop*10))+"_"+str(opt)+"_"+str(bbb)+"_"+str(e)+".csv")
      a,b = traintestAUC(model,X_train, X_test,y_train,y_test)
      results["SNPs:"+str(x_train.shape[2])].append(str(a)+"-"+str(b))
      a,b = traintestMCC(model,X_train, X_test,y_train,y_test)
      results2["SNPs:"+str(x_train.shape[2])].append(str(a)+"-"+str(b))
      a,b = traintestf1score(model,X_train, X_test,y_train,y_test)
      results3["SNPs:"+str(x_train.shape[2])].append(str(a)+"-"+str(b)) 

     except:
      results["SNPs:"+str(x_train.shape[2])].append(str(0)+"-"+str(0))
      results2["SNPs:"+str(x_train.shape[2])].append(str(0)+"-"+str(0)) 
      results3["SNPs:"+str(x_train.shape[2])].append(str(0)+"-"+str(0))

     import psutil
     process = psutil.Process(os.getpid())
     memory_usage = process.memory_info().rss
     print(memory_usage) 
     del model
     import gc
     gc.collect()   

 del X_train
 del X_test
 import gc
 gc.collect()
# Define deep learning models.
@profile
def Stackedgrulstmbilstm(model1,model2,X_train, X_test, y_train, y_test):
 for drop in ddropout:
  for opt in optimizer:
   for bbb in batchsize:
    for e in epochsnumber:
     try: 
      model = keras.models.Sequential()
      
      if model1=="LSTM":  
        model.add(LSTM(128+int(X_train.shape[2]**(1/2)), input_shape=(X_train.shape[1], X_train.shape[2]),return_sequences=True))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
        model.add(LSTM(64+int(X_train.shape[2]**(1/4)), return_sequences=True))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
        
        #model.add(Dense(1, activation='sigmoid'))
      elif model1=="GRU":
        model.add(GRU(128+int(X_train.shape[2]**(1/2)), input_shape=(X_train.shape[1], X_train.shape[2]),return_sequences=True))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
        model.add(GRU(64+int(X_train.shape[2]**(1/4)), return_sequences=True))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
        
        #model.add(Dense(1, activation='sigmoid'))
      elif model1=="BILSTM":
        model.add(Bidirectional(LSTM(128+int(X_train.shape[2]**(1/2)), input_shape=(X_train.shape[1], X_train.shape[2]),return_sequences=True)))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
        model.add(Bidirectional(LSTM(64+int(X_train.shape[2]**(1/4)), return_sequences=True)))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
        
        #model.add(Dense(1, activation='sigmoid'))
    
      if model2=="LSTM":  
        model.add(LSTM(32+int(X_train.shape[2]**(1/8)), return_sequences=True))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
        model.add(tf.keras.layers.Dropout(drop))
        model.add((LSTM(16+int(X_train.shape[2]**(1/16)),return_sequences=False)))
        model.add(tf.keras.layers.Dropout(drop))
        #model.add(Dense(1, activation='sigmoid'))
      elif model2=="GRU":
        model.add(GRU(32+int(X_train.shape[2]**(1/8)), return_sequences=True))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
        model.add(tf.keras.layers.Dropout(drop))
        model.add((GRU(16+int(X_train.shape[2]**(1/16)),return_sequences=False)))
        model.add(tf.keras.layers.Dropout(drop))
        #model.add(Dense(1, activation='sigmoid'))
      elif model2=="BILSTM":
        model.add(Bidirectional(LSTM(32+int(X_train.shape[2]**(1/8)), return_sequences=True)))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
        model.add(tf.keras.layers.Dropout(drop))
        model.add(Bidirectional((LSTM(16+int(X_train.shape[2]**(1/16)),return_sequences=False))))
        model.add(tf.keras.layers.Dropout(drop))
        #model.add(Dense(1, activation='sigmoid'))

      model.add(Dense(1, activation='sigmoid'))

      model.compile(loss='binary_crossentropy', metrics=['accuracy'],optimizer=opt)
      w=  compute_class_weight(class_weight = "balanced", classes= np.unique(y_train), y= y_train)
      w = dict(zip(np.unique(y_train), w))
      model.fit(X_train, y_train,epochs=e,batch_size=bbb,validation_split=0.1,shuffle=False,class_weight=w,verbose=0)
      #get_feature_importance_permutation2(model, X_test,y_test,"",model1+"_"+model2+"_"+str(int(drop*10))+"_"+str(opt)+"_"+str(bbb)+"_"+str(e)+".csv")
      a,b = traintestAUC(model,X_train, X_test,y_train,y_test)
      results["SNPs:"+str(x_train.shape[2])].append(str(a)+"-"+str(b))
      a,b = traintestMCC(model,X_train, X_test,y_train,y_test)
      results2["SNPs:"+str(x_train.shape[2])].append(str(a)+"-"+str(b))
      a,b = traintestf1score(model,X_train, X_test,y_train,y_test)
      results3["SNPs:"+str(x_train.shape[2])].append(str(a)+"-"+str(b)) 

     except:
      results["SNPs:"+str(x_train.shape[2])].append(str(0)+"-"+str(0))
      results2["SNPs:"+str(x_train.shape[2])].append(str(0)+"-"+str(0))
      results3["SNPs:"+str(x_train.shape[2])].append(str(0)+"-"+str(0))   

     import psutil
     process = psutil.Process(os.getpid())
     memory_usage = process.memory_info().rss
     print(memory_usage)
     del model
     import gc
     gc.collect()
 
 del X_train
 del X_test
 import gc
 gc.collect()

@profile
def grumodel(X_train, X_test, y_train, y_test):
 for drop in ddropout:
  for opt in optimizer:
   for bbb in batchsize:
    for e in epochsnumber:
     try:
      model = keras.models.Sequential()
      model.add(GRU(128+int(X_train.shape[2]**(1/2)), input_shape=(X_train.shape[1], X_train.shape[2]),return_sequences=True))
      model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
      model.add(GRU(64+int(X_train.shape[2]**(1/4)), return_sequences=True))
      model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
      model.add(tf.keras.layers.Dropout(drop))
      model.add(GRU(32+int(X_train.shape[2]**(1/8)), return_sequences=True))
      model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
      model.add(tf.keras.layers.Dropout(drop))
      model.add(GRU(16+int(X_train.shape[2]**(1/16)), return_sequences=True))
      model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
      model.add(tf.keras.layers.Dropout(drop))
      model.add((GRU(8,return_sequences=False)))
      model.add(tf.keras.layers.Dropout(drop))
      model.add(Dense(1, activation='sigmoid'))
      model.compile(loss='binary_crossentropy', metrics=['accuracy'],optimizer=opt)
      w=  compute_class_weight(class_weight = "balanced", classes= np.unique(y_train), y= y_train)
      w = dict(zip(np.unique(y_train), w))
      model.fit(X_train, y_train,epochs=e,batch_size=bbb,validation_split=0.1,shuffle=False,class_weight=w,verbose=0)
      #get_feature_importance_permutation2(model, X_test,y_test,"","GRU_"+str(int(drop*10))+"_"+str(opt)+"_"+str(bbb)+"_"+str(e)+".csv")
      import psutil
      process = psutil.Process(os.getpid())
      memory_usage = process.memory_info().rss
      print(memory_usage)

      a,b = traintestAUC(model,X_train, X_test,y_train,y_test)
      results["SNPs:"+str(x_train.shape[2])].append(str(a)+"-"+str(b))
      a,b = traintestMCC(model,X_train, X_test,y_train,y_test)
      results2["SNPs:"+str(x_train.shape[2])].append(str(a)+"-"+str(b))
      a,b = traintestf1score(model,X_train, X_test,y_train,y_test)
      results3["SNPs:"+str(x_train.shape[2])].append(str(a)+"-"+str(b)) 

     except:
      results["SNPs:"+str(x_train.shape[2])].append(str(0)+"-"+str(0))
      results2["SNPs:"+str(x_train.shape[2])].append(str(0)+"-"+str(0))
      results3["SNPs:"+str(x_train.shape[2])].append(str(0)+"-"+str(0))

     del model
     import gc
     gc.collect()
 
 del X_train
 del X_test
 import gc
 gc.collect()

@profile
def annmodel(X_train, X_test, y_train, y_test):
 for drop in ddropout:
  for opt in optimizer:
   for bbb in batchsize:
    for e in epochsnumber:
     try:

      model = keras.models.Sequential()
      model.add(Dense(128+int(X_train.shape[1]**(1/2))))
      model.add(tf.keras.layers.Dropout(drop))
      model.add(Dense(64+int(X_train.shape[1]**(1/4))))
      model.add(tf.keras.layers.Dropout(drop))
      model.add(Dense(32+int(X_train.shape[1]**(1/8))))
      model.add(tf.keras.layers.Dropout(drop))
      model.add(Dense(16+int(X_train.shape[1]**(1/16))))
      model.add(tf.keras.layers.Dropout(drop))
      model.add((Dense(8)))
      model.add(tf.keras.layers.Dropout(drop))
      model.add(Dense(1, activation='sigmoid'))
      model.compile(loss='binary_crossentropy', metrics=['accuracy'],optimizer=opt)
      
      w=  compute_class_weight(class_weight = "balanced", classes= np.unique(y_train), y= y_train)
      w = dict(zip(np.unique(y_train), w))                                               
      model.fit(X_train, y_train,epochs=e,batch_size=bbb,validation_split=0.1,shuffle=False,class_weight=w,verbose=0)
      
      
      
      
      #get_feature_importance_permutation1(model, X_test,y_test,"","ANN_"+str(int(drop*10))+"_"+str(opt)+"_"+str(bbb)+"_"+str(e)+".csv")
      a,b = traintestAUC(model,X_train, X_test,y_train,y_test)
      results["SNPs:"+str(x_train.shape[1])].append(str(a)+"-"+str(b))
      a,b = traintestMCC(model,X_train, X_test,y_train,y_test)
      results2["SNPs:"+str(x_train.shape[1])].append(str(a)+"-"+str(b))
      a,b = traintestf1score(model,X_train, X_test,y_train,y_test)
      results3["SNPs:"+str(x_train.shape[1])].append(str(a)+"-"+str(b)) 

     except:
      results["SNPs:"+str(x_train.shape[1])].append(str(0)+"-"+str(0))
      results2["SNPs:"+str(x_train.shape[1])].append(str(0)+"-"+str(0))
      results3["SNPs:"+str(x_train.shape[1])].append(str(0)+"-"+str(0))

     import psutil
     process = psutil.Process(os.getpid())
     memory_usage = process.memory_info().rss
     print(memory_usage)
     del model
     import gc
     gc.collect()

 del X_test
 del X_train
 import gc
 gc.collect()

@profile
def bilstmmodel(X_train, X_test, y_train, y_test):
 for drop in ddropout:
  for opt in optimizer:
   for bbb in batchsize:
    for e in epochsnumber:
     try:  
      model = keras.models.Sequential()
      model.add(Bidirectional(LSTM(128+int(X_train.shape[2]**(1/2)), input_shape=(X_train.shape[1], X_train.shape[2]),return_sequences=True)))
      model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
      model.add(Bidirectional(LSTM(64+int(X_train.shape[2]**(1/4)), return_sequences=True)))
      model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
      model.add(tf.keras.layers.Dropout(drop))
      model.add(Bidirectional(LSTM(32+int(X_train.shape[2]**(1/8)), return_sequences=True)))
      model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
      model.add(tf.keras.layers.Dropout(drop))
      model.add(Bidirectional(LSTM(16+int(X_train.shape[2]**(1/16)),return_sequences=False)))
      model.add(tf.keras.layers.Dropout(drop))
      model.add(Dense(1, activation='sigmoid'))
      model.compile(loss='binary_crossentropy', metrics=['accuracy'],optimizer=opt)
      w=  compute_class_weight(class_weight = "balanced", classes= np.unique(y_train), y= y_train)
      w = dict(zip(np.unique(y_train), w))                                               
      model.fit(X_train, y_train,epochs=e,batch_size=bbb,validation_split=0.1,shuffle=False,class_weight=w,verbose=0)
      #get_feature_importance_permutation2(model, X_test,y_test,"","BILSTM_"+str(int(drop*10))+"_"+str(opt)+"_"+str(bbb)+"_"+str(e)+".csv")
      import psutil
      process = psutil.Process(os.getpid())
      memory_usage = process.memory_info().rss
      print(memory_usage)
      a,b = traintestAUC(model,X_train, X_test,y_train,y_test)
      results["SNPs:"+str(x_train.shape[2])].append(str(a)+"-"+str(b))
      a,b = traintestMCC(model,X_train, X_test,y_train,y_test)
      results2["SNPs:"+str(x_train.shape[2])].append(str(a)+"-"+str(b))  
      a,b = traintestf1score(model,X_train, X_test,y_train,y_test)
      results3["SNPs:"+str(x_train.shape[2])].append(str(a)+"-"+str(b))  
          
     except:
      results["SNPs:"+str(x_train.shape[2])].append(str(0)+"-"+str(0))
      results2["SNPs:"+str(x_train.shape[2])].append(str(0)+"-"+str(0))
      results3["SNPs:"+str(x_train.shape[2])].append(str(0)+"-"+str(0))

     import psutil
     process = psutil.Process(os.getpid())
     memory_usage = process.memory_info().rss
     print(memory_usage)
     del model
     import gc
     gc.collect()


 del X_train
 del X_test
 import gc
 gc.collect()


results = {}
results2 = {}
results3 = {}

count=1

pvalues = [sys.argv[3]]

import tracemalloc
tracemalloc.start()

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
        
        annmodel(x_train, x_test, y_train, y_test)
        del(annmodel)
        x_train =  x_train.reshape(x_train.shape[0],1,x_train.shape[1])
        x_test =  x_test.reshape(x_test.shape[0],1,x_test.shape[1])
        
        grumodel(x_train, x_test, y_train, y_test)
        del(grumodel)

        lstmmodel(x_train, x_test, y_train, y_test)
        del(lstmmodel)

        bilstmmodel(x_train, x_test, y_train, y_test)
        del(bilstmmodel)

        Stackedgrulstmbilstm("LSTM","GRU",x_train, x_test, y_train, y_test)
        #del(Stackedgrulstmbilstm)
        Stackedgrulstmbilstm("LSTM","BILSTM",x_train, x_test, y_train, y_test)
        #del(Stackedgrulstmbilstm)
        Stackedgrulstmbilstm("BILSTM","LSTM",x_train, x_test, y_train, y_test)
        #del(Stackedgrulstmbilstm)
        Stackedgrulstmbilstm("BILSTM","GRU",x_train, x_test, y_train, y_test)
        #del(Stackedgrulstmbilstm)
        Stackedgrulstmbilstm("GRU","LSTM",x_train, x_test, y_train, y_test)
        #del(Stackedgrulstmbilstm)
        Stackedgrulstmbilstm("GRU","BILSTM",x_train, x_test, y_train, y_test)
        del(Stackedgrulstmbilstm)

        L = pd.DataFrame(results)
        L.to_csv(pheno+os.sep+iteration+os.sep+pvalue+os.sep+"Results_DeepLearning_AUC.csv",index=False, sep="\t")
        L = pd.DataFrame(results2)
        L.to_csv(pheno+os.sep+iteration+os.sep+pvalue+os.sep+"Results_DeepLearning_MCC.csv",index=False, sep="\t")
        L = pd.DataFrame(results3)
        L.to_csv(pheno+os.sep+iteration+os.sep+pvalue+os.sep+"Results_DeepLearning_f1score.csv",index=False, sep="\t")
        
print(tracemalloc.get_traced_memory())

tracemalloc.stop()
exit(0)
