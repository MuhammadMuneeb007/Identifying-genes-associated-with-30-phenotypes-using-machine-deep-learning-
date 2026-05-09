import pandas as pd
import numpy as np
import os
from os.path import exists
import re
import sys

def sorted_nicely(l):
    convert      = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def resolve_algo_name(index_str):
    try:
        indexs = pd.read_csv("MachineLearningAlgorithms.txt", sep="\t")
        match  = indexs[indexs["Algorithm Index for Reference"] == index_str]
        if len(match) > 0:
            return match["Algorithm Name"].values[0]
        return index_str
    except Exception:
        return index_str

# ── Target SNP counts — always 7 thresholds ───────────────────────────────────
TARGET_SNPS = [50, 100, 200, 500, 1000, 5000, 10000]

def match_to_targets(data, target_snps):
    """
    Given a dataframe whose columns are SNP counts (integers),
    return a new dataframe with exactly len(target_snps) columns,
    each matched to the closest SNP count in the data.
    Column names become the target SNP counts.
    """
    available = np.array([int(c) for c in data.columns])
    matched   = pd.DataFrame()
    for target in target_snps:
        # Find closest available SNP count to this target
        closest_idx = np.argmin(np.abs(available - target))
        closest_col = str(data.columns[closest_idx])
        matched[str(target)] = data[closest_col].values
    return matched

metric = sys.argv[1]

hu        = pd.DataFrame()
disease   = []
auc_list  = []
SNP       = []
STD       = []
ALGO_IDX  = []
ALGO_NAME = []

for loop in pd.read_csv("allphenotypesname2.txt", header=None)[0].values:
    count = 0
    for loop2 in range(1, 6):
        if exists("./" + loop + os.sep + str(loop2) + os.sep +
                  "Results_MachineLearning_" + metric + ".csv"):
            count += 1
    print(loop, count)

    if count == 5:
        n_targets = len(TARGET_SNPS)
        shape_rows = None

        # ── First pass: get number of rows from fold 1 ───────────────────────
        data_check = pd.read_csv(
            "./" + loop + os.sep + "1" + os.sep +
            "Results_MachineLearning_" + metric + ".csv", sep="\t")
        shape_rows = data_check.shape[0]

        average      = np.zeros((shape_rows, n_targets))
        fold_results = []

        for loop2 in range(1, 6):
            data = pd.read_csv(
                "./" + loop + os.sep + str(loop2) + os.sep +
                "Results_MachineLearning_" + metric + ".csv", sep="\t")

            # Strip SNPs: prefix and sort columns
            data.columns = data.columns.str.replace(r"SNPs:", "", regex=True)
            x    = sorted_nicely(data.columns)
            data = data[list(x)]

            # Split train/test keep only test
            for col in data.columns:
                data[["Train"+col, "Test"+col]] = data[col].str.split(
                    "/", expand=True)
                data["Test"+col] = pd.to_numeric(
                    data["Test"+col], errors="coerce")
                del data[col]
                del data["Train"+col]

            # Rename columns to just the SNP count number
            data.columns = [c.replace("Test", "").strip() for c in data.columns]

            # ── Match to 7 target SNP counts by closest value ─────────────────
            data_matched = match_to_targets(data, TARGET_SNPS)

            fold_results.append(data_matched.values)
            average = average + data_matched.values

        # ── Compute mean and std across 5 folds ──────────────────────────────
        std_array = np.std(fold_results, axis=0, ddof=1)
        average   = average / 5

        # ── Find max positions ────────────────────────────────────────────────
        result = np.where(average == np.amax(average))
        row    = result[0]
        col    = result[1]

        # ── Among max positions pick minimum std ──────────────────────────────
        aa     = 1000
        minrow = row[0]
        mincol = col[0]
        for xx in range(len(row)):
            std_val = std_array[row[xx]][col[xx]]
            if not np.isnan(std_val) and aa > std_val:
                aa     = std_val
                minrow = row[xx]
                mincol = col[xx]

        maximum  = average[minrow][mincol]
        std      = std_array[minrow][mincol]

        algo_idx  = "ML_" + str(minrow + 1)
        algo_name = resolve_algo_name(algo_idx)
        snp_int   = TARGET_SNPS[mincol]

        disease.append(loop)
        auc_list.append(round(maximum, 2))
        STD.append(round(std, 2) if not np.isnan(std) else np.nan)
        SNP.append(snp_int)
        ALGO_IDX.append(algo_idx)
        ALGO_NAME.append(algo_name)

        print(f"  Best: {algo_idx} | {algo_name} | SNPs={snp_int} | "
              f"Avg={round(maximum,2)} | STD={round(std,2) if not np.isnan(std) else 'NaN'}")

print(len(disease), len(auc_list), len(STD), len(ALGO_IDX), len(ALGO_NAME), len(SNP))

hu["Phenotype"]                                = disease
hu["Test " + metric + " 5 Iterations Average"] = auc_list
hu["Standard Deviation"]                       = STD
hu["Machine learning algorithm index"]         = ALGO_IDX
hu["Algorithm Name"]                           = ALGO_NAME
hu["Number of SNPs"]                           = SNP

hu.to_html("Machinelearningbasedbechmarking" + metric + ".html", index=False)
hu.to_csv( "Machinelearningbasedbechmarking" + metric + ".csv",  index=False, sep=",")
print(hu.to_markdown(index=False))
