import pandas as pd
import numpy as np
import os
from os.path import exists
import re
import sys

def sorted_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

metric = sys.argv[1]

hu = pd.DataFrame()
disease = []
auc = []
SNP = []
STD = []
ALGO = []

for loop in pd.read_csv("allphenotypesname2.txt", header=None)[0].values:
    count = 0
    for loop2 in range(1, 6):
        if exists("./"+loop+os.sep+str(loop2)+os.sep+"Results_MachineLearning_"+metric+".csv"):
            count = count + 1
    print(loop, count)

    if count == 5:
        # ── Load all folds and align columns ────────────────────────────
        fold_data = []
        all_columns = set()
    
        for loop2 in range(1, 6):
            data = pd.read_csv("./"+loop+os.sep+str(loop2)+os.sep+"Results_MachineLearning_"+metric+".csv", sep="\t")
            data.columns = data.columns.str.replace(r"SNPs:", "", regex=True)
            x = sorted_nicely(data.columns)
            data = data[list(x)]
    
            for col in data.columns:
                data[["Train"+col, "Test"+col]] = data[col].str.split("/", expand=True)
                data["Test"+col] = pd.to_numeric(data["Test"+col], errors="coerce")
                del data[col]
                del data["Train"+col]
    
            all_columns.update(data.columns)
            fold_data.append(data)
    
        # Align all folds to the same columns, filling missing with NaN
        all_columns = sorted_nicely(list(all_columns))
        fold_data = [df.reindex(columns=all_columns) for df in fold_data]
    
        # Stack and compute mean and std across folds, ignoring NaN
        stacked   = np.stack([df.values for df in fold_data], axis=0)  # shape (5, rows, cols)
        average   = np.nanmean(stacked, axis=0)
        std_array = np.nanstd(stacked, axis=0, ddof=1)
    
        # Find indices where average is maximum
        result = np.where(average == np.nanmax(average))
        row = result[0]
        col = result[1]
    
        # Among all max positions, find the one with minimum std
        aa     = 1000
        minrow = 0
        mincol = 0
        for xx in range(0, len(row)):
            if aa > std_array[row[xx]][col[xx]]:
                aa     = std_array[row[xx]][col[xx]]
                minrow = row[xx]
                mincol = col[xx]
    
        maximum = average[minrow][mincol]
        std     = std_array[minrow][mincol]
    
        x = np.array(all_columns)
    
        disease.append(loop)
        auc.append(maximum)
        STD.append(std)
        SNP.append(x[mincol])
        ALGO.append("ML_"+str(minrow+1))

print(len(disease), len(auc), len(STD), len(ALGO), len(SNP))

hu['Phenotype'] = disease
hu['Test '+metric+' 5 Iterations Average'] = auc
hu['Standard Deviation'] = STD
hu['Machine learning algorithm index'] = ALGO
hu['Number of SNPs'] = SNP

hu.to_html("Machinelearningbasedbechmarking"+metric+".html")
hu.to_csv("Machinelearningbasedbechmarking"+metric+".csv", index=False, sep=",")
print(hu.to_markdown())
