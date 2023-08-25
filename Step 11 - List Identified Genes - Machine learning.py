import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import re
import squarify
#plt.style.use('seaborn')
from natsort import natsort_keygen
import pdfkit
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
import seaborn as sns

import pandas as pd
import numpy as np
import sys
import os


phenotypes1 =   pd.read_csv("DeeplearningbasedbechmarkingAUC.csv")["Phenotype"].values
phenotypes1 = ['ADHD', 'Allergicrhinitis', 'Asthma', 'Bipolar disorder', 'Cholesterol', 'Craves sugar', 'Dental decay', 'Depression', 'Diagnosed vitamin D deficiency', 'Diagnosed with sleep apnea', 'Dyslexia', 'Earlobe free or attached', 'Hair type', 'Hypertension', 'Hypertriglyceridemia', 'Irritable bowel syndrome', 'Mental disease', 'Migraine', 'Motion sickness', 'Panic disorder', 'Photic sneeze reflex photoptarmis', 'PTSD', 'Scoliosis', 'Sensitivity to Mosquito bites', 'Sleep disorders', 'Strabismus', 'Thyroid issues cancer', 'TypeIIDiabetes', 'Eczema', 'Restless leg syndrome']

diseaseornot = []


###Machine learning processing
def machinelearningprocessor(metric):
    def machinelearning(x, y=0):
        indexs = pd.read_csv("MachineLearningAlgorithms.txt",sep="\t")
        indexs = indexs[indexs["Algorithm Index for Reference"]==x]
        print(indexs)
        print(len(indexs),x)
        x = indexs["Algorithm Name"].values[0]
        return x
    data = pd.read_csv("Machinelearningbasedbechmarking"+metric+".csv")


    data['Machine learning algorithm index'] = data['Machine learning algorithm index'].apply(machinelearning)
    data['Number of SNPs'] = np.where((data['Number of SNPs'] >= 0) & (data['Number of SNPs'] <=70), 50, data['Number of SNPs'])
    data['Number of SNPs'] = np.where((data['Number of SNPs'] >= 70) & (data['Number of SNPs'] <=150), 100, data['Number of SNPs'])
    data['Number of SNPs'] = np.where((data['Number of SNPs'] >= 180) & (data['Number of SNPs'] <=220), 200, data['Number of SNPs'])
    data['Number of SNPs'] = np.where((data['Number of SNPs'] >= 220) & (data['Number of SNPs'] <=600), 500, data['Number of SNPs'])
    data['Number of SNPs'] = np.where((data['Number of SNPs'] >= 600) & (data['Number of SNPs'] <=1500), 1000, data['Number of SNPs'])
    data['Number of SNPs'] = np.where((data['Number of SNPs'] >= 1500) & (data['Number of SNPs'] <=2500), 2000, data['Number of SNPs'])
    data['Number of SNPs'] = np.where((data['Number of SNPs'] >= 2500) & (data['Number of SNPs'] <=5500), 5000, data['Number of SNPs'])
    data['Number of SNPs'] = np.where((data['Number of SNPs'] >= 9000) & (data['Number of SNPs'] <=15000), 10000, data['Number of SNPs'])

    #data['Phenotype'] = phenotypes1
    data = data.round(2)

    data.rename(columns = {'Test AUC 5 Iterations Average':'AUC'}, inplace = True)
    data.rename(columns = {'Test f1score 5 Iterations Average':'F1 Score'}, inplace = True)
    data.rename(columns = {'Test MCC 5 Iterations Average':'MCC'}, inplace = True)
    
    data.rename(columns = {'Standard Deviation':'SD'}, inplace = True)
    data.rename(columns = {'Standard Deviation':'SD'}, inplace = True)
    data.sort_values(by=['Phenotype'], inplace=True)
    data['Phenotype'] = phenotypes1

    #print(list(data['Phenotype'].values))
    #exit(0)

    data.to_csv("Machinelearning_Results_"+metric+".csv",sep="\t",index=None)
    data.to_html("Machinelearning_Results_"+metric+".html",index=None)
    ### Print value counts
    print(data['Machine learning algorithm index'].value_counts())
    #pdfkit.from_file('Machinelearning_Results.html', 'Machinelearning_Results.pdf', configuration=config)
    return data

def machinelearningprocessor1(metric):
    def machinelearning(x, y=0):
        indexs = pd.read_csv("MachineLearningAlgorithms.txt",sep="\t")
        indexs = indexs[indexs["Algorithm Index for Reference"]==x]
        print(indexs)
        print(len(indexs),x)
        x = indexs["Algorithm Name"].values[0]
        return x
    data = pd.read_csv("Machinelearningbasedbechmarking"+metric+".csv")


    data['Machine learning algorithm index'] = data['Machine learning algorithm index'].apply(machinelearning)
    data['Number of SNPs'] = np.where((data['Number of SNPs'] >= 0) & (data['Number of SNPs'] <=70), 50, data['Number of SNPs'])
    data['Number of SNPs'] = np.where((data['Number of SNPs'] >= 70) & (data['Number of SNPs'] <=150), 100, data['Number of SNPs'])
    data['Number of SNPs'] = np.where((data['Number of SNPs'] >= 180) & (data['Number of SNPs'] <=220), 200, data['Number of SNPs'])
    data['Number of SNPs'] = np.where((data['Number of SNPs'] >= 220) & (data['Number of SNPs'] <=600), 500, data['Number of SNPs'])
    data['Number of SNPs'] = np.where((data['Number of SNPs'] >= 600) & (data['Number of SNPs'] <=1500), 1000, data['Number of SNPs'])
    data['Number of SNPs'] = np.where((data['Number of SNPs'] >= 1500) & (data['Number of SNPs'] <=2500), 2000, data['Number of SNPs'])
    data['Number of SNPs'] = np.where((data['Number of SNPs'] >= 2500) & (data['Number of SNPs'] <=5500), 5000, data['Number of SNPs'])
    data['Number of SNPs'] = np.where((data['Number of SNPs'] >= 9000) & (data['Number of SNPs'] <=15000), 10000, data['Number of SNPs'])

    #data['Phenotype'] = phenotypes1
    data = data.round(2)

    data.rename(columns = {'Test AUC 5 Iterations Average':'AUC'}, inplace = True)
    data.rename(columns = {'Test f1score 5 Iterations Average':'F1 Score'}, inplace = True)
    data.rename(columns = {'Test MCC 5 Iterations Average':'MCC'}, inplace = True)
    
    data.rename(columns = {'Standard Deviation':'SD'}, inplace = True)
    data.sort_values(by=['Phenotype'], inplace=True)

    data.to_csv("Machinelearning_Results_"+metric+".csv",sep="\t",index=None)
    data.to_html("Machinelearning_Results_"+metric+".html",index=None)
    ### Print value counts
    print(data['Machine learning algorithm index'].value_counts())
    #pdfkit.from_file('Machinelearning_Results.html', 'Machinelearning_Results.pdf', configuration=config)
    return data


machinelearningprocessor("MCC")
machinelearningprocessor("f1score")
machinelearningprocessor("AUC")


dataauc = machinelearningprocessor1("MCC")
dataauc.sort_values(by=['Phenotype'], inplace=True)

dataf1score = machinelearningprocessor1("f1score")
dataf1score.sort_values(by=['Phenotype'], inplace=True)

datamcc = machinelearningprocessor1("AUC")
datamcc.sort_values(by=['Phenotype'], inplace=True)

yuui = []
def getmesnpsandfile1(path,snp,actual):
  allfiles = os.listdir(path+os.sep+"pv_"+snp)
  temp = ""
  temp2 = ""
  for all in allfiles:
    temp = all.replace("-","").replace(":","").replace(" ","").replace(",","").replace(".csv","")
    #print(temp,actual)
    if temp==actual:
      temp2=all
      break
  #print(path+os.sep+"pv_"+snp+os.sep+snp+".txt",temp2)
  snpsfile = pd.read_csv(path+os.sep+"pv_"+snp+os.sep+snp+".txt",header=None)[0].values
  weights =  pd.read_csv(path+os.sep+"pv_"+snp+os.sep+temp2)["Features_importance"].values

  global tempdic

  for loop in range(0,len(snpsfile)):
    if snpsfile[loop] in tempdic:
      tempdic[snpsfile[loop]].append(weights[loop])
    else:
      tempdic[snpsfile[loop]]= [weights[loop]]


def getmeaphenotype1(path,snps,algo):
  files = os.listdir(path)
  pvaluesfiles = []
  count=0

  for l in files:
    if "pv_" in l:
      pvaluesfiles.append(l.replace("pv_",""))

  pvaluesfiles = np.sort(pvaluesfiles)

  if snps==50:
    pvaluesfiles = pvaluesfiles[0]
  if snps==100:
    pvaluesfiles = pvaluesfiles[1]
  if snps==200:
    pvaluesfiles = pvaluesfiles[2]
  if snps==500:
    pvaluesfiles = pvaluesfiles[3]
  if snps==1000:
    pvaluesfiles = pvaluesfiles[4]
  if snps==5000:
    pvaluesfiles = pvaluesfiles[5]
  if snps==10000:
    pvaluesfiles = pvaluesfiles[6]

  getmesnpsandfile1(path,str(pvaluesfiles),algo.replace("-","").replace(":","").replace(" ","").replace(",",""))

  pass


tempdic={}
identifiedbymachine = []
commonmachineandgwas =[]
commonbetweenusandgwas = []

def machinelearningmatching():
  machinedata = dataauc
  identifiednumber1 = []
  identifiednumber2 = []
  identifiednumber3 = []
  identified1 = []
  identified2 = []
  identified3 = []

  machinedata.rename(columns={'Machine learning algorithm index': 'Parameters'}, inplace=True)
  for index, row in machinedata.iterrows():
    global tempdic
    tempdic={}
    for loop2 in range(1,6):
      getmeaphenotype1(row['Phenotype']+os.sep+str(loop2),row['Number of SNPs'],row['Parameters'])
      print(len(tempdic))
    for k in tempdic:
      tempdic[k] = np.array(tempdic[k]).sum()
    tempdata = pd.DataFrame(tempdic.items(), columns=['Snps', 'Importance'])

    print(len(tempdata))

    tempdata = tempdata[tempdata['Importance']!=0.0]
    #tempdata.to_csv(row['Phenotype']+os.sep+"FeatureImportance.csv",index=None)

    commons = pd.read_csv("C:\\Users\\kl\\Desktop\\The University of Queensland\\Identify genes using machine learning\\IdentifyGenes"+os.sep+row['Phenotype']+os.sep+"CommonSNPs.csv")
    commons = commons["Common"].values
    #print(len(commons),len(tempdata["Snps"].values))

    identified1.append(list(set(commons).intersection(tempdata["Snps"].values)))

    identifiednumber1.append(len(list(set(commons).intersection(tempdata["Snps"].values))))

  machinedata = dataf1score
  machinedata.rename(columns={'Machine learning algorithm index': 'Parameters'}, inplace=True)
  for index, row in machinedata.iterrows():
    tempdic={}
    for loop2 in range(1,6):
      getmeaphenotype1(row['Phenotype']+os.sep+str(loop2),row['Number of SNPs'],row['Parameters'])

    for k in tempdic:
      tempdic[k] = np.array(tempdic[k]).sum()
    tempdata = pd.DataFrame(tempdic.items(), columns=['Snps', 'Importance'])

    #print(len(tempdata))
    tempdata = tempdata[tempdata['Importance']!=0.0]

    tempdata.to_csv(row['Phenotype']+os.sep+"FeatureImportance.csv",index=None)

    commons = pd.read_csv("C:\\Users\\kl\\Desktop\\The University of Queensland\\Identify genes using machine learning\\IdentifyGenes"+os.sep+row['Phenotype']+os.sep+"CommonSNPs.csv")
    commons = commons["Common"].values
    identified2.append(list(set(commons).intersection(tempdata["Snps"].values)))
    identifiednumber2.append(len(list(set(commons).intersection(tempdata["Snps"].values))))

  machinedata = datamcc
  machinedata.rename(columns={'Machine learning algorithm index': 'Parameters'}, inplace=True)
  for index, row in machinedata.iterrows():
    tempdic={}
    for loop2 in range(1,6):
      getmeaphenotype1(row['Phenotype']+os.sep+str(loop2),row['Number of SNPs'],row['Parameters'])

    for k in tempdic:
      tempdic[k] = np.array(tempdic[k]).sum()
    tempdata = pd.DataFrame(tempdic.items(), columns=['Snps', 'Importance'])

    #print(len(tempdata))
    tempdata = tempdata[tempdata['Importance']!=0.0]

    tempdata.to_csv(row['Phenotype']+os.sep+"FeatureImportance.csv",index=None)

    commons = pd.read_csv("C:\\Users\\kl\\Desktop\\The University of Queensland\\Identify genes using machine learning\\IdentifyGenes"+os.sep+row['Phenotype']+os.sep+"CommonSNPs.csv")
    commonbetweenusandgwas.append(len(commons))
    commons = commons["Common"].values
    identified3.append(list(set(commons).intersection(tempdata["Snps"].values)))
    identifiednumber3.append(len(list(set(commons).intersection(tempdata["Snps"].values))))


  temp1 = []
  temp2 = []
  mappedgenes = []
  locationgenes = []
  for loop in range(0,len(identified1)):
    temp = set(identified1[loop]).union(identified2[loop])
    temp = set(identified3[loop]).union(temp)
    temp1.append(list(temp))
    temp2.append(len(temp))
    phenotypes1 =   pd.read_csv("Final_MachineLearning_Results.csv")["Phenotype"].values
    print(phenotypes1[loop])
    alltempfiles = os.listdir("C:\\Users\\kl\\Desktop\\The University of Queensland\\Identify genes using machine learning\\IdentifyGenes"+os.sep+phenotypes1[loop]+os.sep)
    ori = ""
    for loop2 in alltempfiles:
      if "associations" in loop2:
        ori = loop2
    

    snpsingwas = pd.read_csv("C:\\Users\\kl\\Desktop\\The University of Queensland\\Identify genes using machine learning\\IdentifyGenes"+os.sep+phenotypes1[loop]+os.sep+ori)
    try:
      snpsingwas[['Snp', 'risk']] = snpsingwas["Variant and risk allele"].str.split("-",expand=True)
    except:
      try:
        snpsingwas[['Snp', 'risk','s']] = snpsingwas["Variant and risk allele"].str.split("-",expand=True)
      except:
        try:
          snpsingwas[['Snp', 'risk','s','t']] = snpsingwas["Variant and risk allele"].str.split("-",expand=True)
        except:
          try:
            snpsingwas[['Snp', 'risk','s','r','t']] = snpsingwas["Variant and risk allele"].str.split("-",expand=True)
          except:
            try:
              snpsingwas[['Snp', 'risk','s','r','t','o']] = snpsingwas["Variant and risk allele"].str.split("-",expand=True)
            except:
              try:
                snpsingwas[['Snp', 'risk','s','r','t','o','y']] = snpsingwas["Variant and risk allele"].str.split("-",expand=True)
              except:
                print(loop)
              #continue
    print(len(snpsingwas))
    print(list(temp))
    print(temp)
    snpsingwas = snpsingwas[snpsingwas["Snp"].isin(list(temp))]
    #print(snpsingwas)
    snpsingwas = snpsingwas.drop_duplicates(subset=["Snp","Mapped gene"], keep='first')

    mappedgenes.append(list(snpsingwas["Mapped gene"].values))
    try:
      locationgenes.append(list(snpsingwas["Location"].values))
    except:
      locationgenes.append("")
    print(mappedgenes)
    print(len(snpsingwas))
  #print(snpsingwas.head())

  #snpsingwas = snpsingwas[snpsingwas['Snp'].isin(temp1)]
  #print(len(snpsingwas))
  #print(len(snpsingwas.head()))


  tempdataframe = pd.DataFrame()
  tempdataframe["Phenotype"] = dataauc["Phenotype"].values
  tempdataframe["CommonBetweenUsandGWAS"] = commonbetweenusandgwas
  tempdataframe["Identified"] = temp2 
  tempdataframe["AUC"] = identifiednumber1
  tempdataframe["F1score"] = identifiednumber2
  tempdataframe["MCC"] = identifiednumber3
  tempdataframe["ActualSNPs"] = temp1
  tempdataframe["MappedGene"] = mappedgenes
  tempdataframe["Location"] = locationgenes
  
  
  tempdataframe.to_csv("Final_MachineLearning_Results.csv")
  print(tempdataframe.to_markdown())

  print("AUC",identified1,"F1Score",identified2,"MCC",identified3)

  #identifiedbymachine.append(len(tempdata))
  #
  #commonmachineandgwas.append(identified)

  #print(row['Phenotype'],identified)

  pass


print("Machine learning")
machinelearningmatching()


exit(0)
