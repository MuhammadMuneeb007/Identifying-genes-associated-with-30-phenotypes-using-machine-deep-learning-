import pandas as pd
import numpy as np
import os
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
import numpy as np
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
import seaborn as sns
import pandas as pd
import numpy as np
#phenotypes1 = ['ADHD', 'Allergicrhinitis', 'Asthma', 'Bipolar disorder', 'Cholesterol', 'Craves sugar', 'Dental decay', 'Depression', 'Diagnosed vitamin D deficiency', 'Diagnosed with sleep apnea', 'Dyslexia', 'Earlobe free or attached', 'Hair type', 'Hypertension', 'Hypertriglyceridemia', 'Irritable bowel syndrome', 'Mental disease', 'Migraine', 'Motion sickness', 'Panic disorder', 'Photic sneeze reflex photoptarmis', 'PTSD', 'Scoliosis', 'Sensitivity to Mosquito bites', 'Sleep disorders', 'Strabismus', 'Thyroid issues cancer', 'TypeIIDiabetes', 'Eczema', 'Restless leg syndrome']



def changeme(x):
    #print(x)
    if type(x) is float:
        return str(x)
    return x.strip('][').replace("'","").split(', ')

deep = pd.read_csv("Final_DeepLearning_Results.csv")
machine = pd.read_csv("Final_MachineLearning_Results.csv")

deep['ActualSNPs'] = deep['ActualSNPs'].apply(changeme)
deep['MappedGene'] = deep['MappedGene'].apply(changeme)
deep['Location'] = deep['Location'].apply(changeme)

machine['ActualSNPs'] = machine['ActualSNPs'].apply(changeme)
machine['MappedGene'] = machine['MappedGene'].apply(changeme)
machine['Location'] = machine['Location'].apply(changeme)


a  = deep['ActualSNPs'].values
b  = machine['ActualSNPs'].values

c  = deep['MappedGene'].values
d  = machine['MappedGene'].values

#deep['Location'] = deep['Location'].astype(str)
#machine['Location'] = machine['Location'].astype(str)

e  = deep['Location'].values
f  = machine['Location'].values
x = []
y = []
z = []
zz = []
tttt  =[]
for loop in range(0,len(a)):
    
    t = []
    tt = []
    ttt = []


    for ll in a[loop]:
        t.append(ll)
    for ll in b[loop]:
        t.append(ll)

    for ll in c[loop]:
        tt.append(ll)
    for ll in d[loop]:
        tt.append(ll)
    
    for ll in e[loop]:        
        ttt.append(ll)
    for ll in f[loop]:
        ttt.append(ll)
    
    t = set(t)
    tt = set(tt)
    ttt = set(ttt)
    print(t)
    while("" in t):
        t.remove("")

    while("" in tt):
        tt.remove("")

    while("" in ttt):
        ttt.remove("")

    x.append(list(t))
    y.append(list(tt))
    z.append(list(ttt))
    

    tttt.append(len(t))
    print(tttt)


newframe = pd.DataFrame()
newframe["Phenotype"] = machine["Phenotype"].values
newframe["Common Between Genotype Data and GWAS catalog"] = machine["CommonBetweenUsandGWAS"].values
newframe["Common Beween Identified by Machine/Deep Learning and GWAS catalog"] = tttt
newframe["ActualSNPs"] = x
newframe["MappedGene"] = y
newframe["Location"] = z
config = pdfkit.configuration(wkhtmltopdf='wkhtmltopdf.exe')
newframe.to_csv("Final_Results.csv",index=None)
#newframe.sort_values(by=['Phenotype'], inplace=True)
#data['Phenotype'] = phenotypes1

newframe.to_html("Final_Results.html",index=None)
pdfkit.from_file('Final_Results.html', 'Final_Results.pdf', configuration=config)
    

for loop in range(0,len(x)):
    print("\subsection{"+machine["Phenotype"].values[loop]+"}")
    print("\n")
    print(str(y[loop]).replace("'","").replace('[', '').replace(']', ''))




#newframe.sort_values('Phenotype', inplace=True)
newframe.to_csv("Final_Results.csv",index=False)


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    if len(lst3):
        print("ascascasc",lst3)
    return len(lst3)

twoddata = []


x = newframe["ActualSNPs"].values

for loop in range(0,len(x)):
    for loop2 in range(0,len(x)):
        if loop==loop2:
            twoddata.append(0)
        else:
            t = intersection(x[loop], x[loop2])
            twoddata.append(t)
            if t:
                print(t)


twoddata = np.array(twoddata)
twoddata = twoddata.reshape(len(machine["Phenotype"].values),len(machine["Phenotype"].values))

df = pd.DataFrame(twoddata, machine["Phenotype"].values)

g = sns.heatmap(df, annot=True, annot_kws={"size": 7},square=True, xticklabels=machine["Phenotype"].values, yticklabels=machine["Phenotype"].values,linewidths=0.3,cmap='viridis_r')
g.set_xticklabels(g.get_xmajorticklabels(), fontsize = 7)
g.set_yticklabels(g.get_ymajorticklabels(), fontsize = 7)
plt.tight_layout() 
plt.savefig('plot1.png',dpi=1000)
plt.show()

twoddata = []
y = newframe["MappedGene"].values

for loop in range(0,len(y)):
    for loop2 in range(0,len(y)):
        if loop==loop2:
            twoddata.append(0)
        else:
            t = intersection(y[loop], y[loop2])
            twoddata.append(t)
            if t:
                print(t)
twoddata = np.array(twoddata)
twoddata = twoddata.reshape(len(machine["Phenotype"].values),len(machine["Phenotype"].values))

df = pd.DataFrame(twoddata, machine["Phenotype"].values)

g = sns.heatmap(df, annot=True, annot_kws={"size": 7},square=True, xticklabels=machine["Phenotype"].values, yticklabels=machine["Phenotype"].values,linewidths=0.3,cmap='viridis_r')
g.set_xticklabels(g.get_xmajorticklabels(), fontsize = 7)
g.set_yticklabels(g.get_ymajorticklabels(), fontsize = 7)
plt.tight_layout() 
plt.savefig('plot2.png',dpi=1000)
plt.show()

print(newframe)
#newframe.to_csv("Final_Results.csv")







