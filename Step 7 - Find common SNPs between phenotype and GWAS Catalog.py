import pandas as pd
import numpy as np
import sys
import os
import pandas as pd
import numpy as np
import os


countdata = pd.DataFrame()

pheno = []
count1 = []
count2 = []
common = []

phenotypes = pd.read_csv("allphenotypesname2.txt",header=None)
path1 = "PATH to Genotype data in Plink format"
path2 = "PATH to SNPs associated with Phenotypes"

countdata = pd.DataFrame()

pheno = []
count1 = []
count2 = []
common = []

for loop in phenotypes[0].values:
  print(loop)
  snpsinorginaldata =pd.read_csv(path1+os.sep+loop+os.sep+"final.QC.bim",header=None,sep="\t")[1].values
  listcsv = os.listdir(path2+os.sep+loop)

  snpsingwas = pd.read_csv(path2+os.sep+loop+os.sep+listcsv[0])
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
          continue 


  pheno.append(loop)
  count1.append(len(snpsingwas['Snp'].values))
  count2.append(len(snpsinorginaldata))
  common.append(len(list(set(snpsingwas['Snp'].values).intersection(snpsinorginaldata))))
  tempp = list(set(snpsingwas['Snp'].values).intersection(snpsinorginaldata))
  p =  pd.DataFrame()
  p["Common"] = tempp
  p.to_csv(loop+os.sep+"CommonSNPs.csv")

countdata["Phenotype"] = pheno
countdata["SNPs in GWAS Catalogue"] = count1
countdata["SNPs in our data"] = count2
countdata["Common SNPs"] = common
countdata.to_csv("SNPscount.csv")

