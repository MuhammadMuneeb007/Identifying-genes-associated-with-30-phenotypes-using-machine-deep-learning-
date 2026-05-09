#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import ast
import json
import argparse
from typing import Dict, List, Set, Tuple, Optional

import pandas as pd


# ============================================================
# IDENTIFY 7.0 — FINAL GENE IDENTIFICATION USING NEW GWAS FILES
# ============================================================
# Purpose:
#   Final gene-identification step using:
#     1. Best ML models selected by AUC, F1, and MCC.
#     2. Fold-wise feature-importance files.
#     3. New GWAS Catalog selected association files from:
#          GWASCatalogDownloaded/<Phenotype>/
#
# Expected ML files:
#   MachinelearningbasedbechmarkingAUC.csv
#   Machinelearningbasedbechmarkingf1score.csv
#   MachinelearningbasedbechmarkingMCC.csv
#   MachineLearningAlgorithms.txt
#
# Expected phenotype folders:
#   <Phenotype>/1/pv_*/<pv>.txt
#   <Phenotype>/1/pv_*/*.csv
#   ...
#   <Phenotype>/5/pv_*/...
#
# Expected new GWAS files:
#   GWASCatalogDownloaded/<Phenotype>/GWAS_Selected_Common_Variants.csv
#   GWASCatalogDownloaded/<Phenotype>/GWAS_Selected_Annotated_Associations.csv
#   GWASCatalogDownloaded/<Phenotype>/GWAS_Summary.csv
#
# Main outputs:
#   GeneIdentification_Final/Final_Gene_Identification_Results.csv
#   GeneIdentification_Final/Final_Gene_Identification_Results.html
#   GeneIdentification_Final/Final_Gene_Identification_Results.md
#   GeneIdentification_Final/Final_Gene_Identification_Table.tex
#   GeneIdentification_Final/<Phenotype>/*
# ============================================================


# ============================================================
# SETTINGS
# ============================================================
GWAS_ROOT = "GWASCatalogDownloaded"
OUTPUT_ROOT = "GeneIdentification_Final"

ML_ALGO_FILE = "MachineLearningAlgorithms.txt"

ML_RESULT_FILES = {
    "AUC": "MachinelearningbasedbechmarkingAUC.csv",
    "F1": "Machinelearningbasedbechmarkingf1score.csv",
    "MCC": "MachinelearningbasedbechmarkingMCC.csv",
}

N_FOLDS = 5


# ============================================================
# PHENOTYPE DISPLAY NAMES
# ============================================================
DISPLAY_NAMES = {
    "ADHD": "Attention Deficit Hyperactivity Disorder (ADHD)",
    "Allergicrhinitis": "Allergic rhinitis",
    "Amblyopia": "Amblyopia",
    "Asthma": "Asthma",
    "Astigmatism": "Astigmatism",
    "Bipolardisorder": "Bipolar Disorder",
    "Cholesterol": "Cholesterol",
    "clusterheadache": "Cluster headache",
    "Cravessugar": "Craves sugar",
    "Dentaldecay": "Dental decay",
    "Depression": "Depression",
    "DiagnosedVitaminDdeficiency": "Diagnosed Vitamin D deficiency",
    "DiagnosedwithSleepApnea": "Diagnosed with Sleep Apnea",
    "Dyslexia": "Dyslexia",
    "EarlobeFreeorattached": "Earlobe Free or attached",
    "eczema": "Eczema",
    "generalizedanxietydisorder": "Generalized Anxiety Disorder",
    "HairType": "Hair Type",
    "HaveMECFS": "Have Myalgic Encephalomyelitis/Chronic Fatigue Syndrome (MECFS)",
    "Hypertension": "Hypertension",
    "Hypertriglyceridemia": "Hypertriglyceridemia",
    "IrritableBowelSyndrome": "Irritable Bowel Syndrome",
    "MentalDisease": "Mental Disease",
    "Migraine": "Migraine",
    "Motionsickness": "Motion Sickness",
    "PanicDisorder": "Panic Disorder",
    "PhoticSneezeReflexPhotoptarmis": "Photic sneeze reflex (photoptarmic reflex)",
    "Plantarfasciitis": "Plantar fasciitis",
    "PosttraumaticStressDisorderorPTSD": "Post-Traumatic Stress Disorder (PTSD)",
    "restlesslegsyndrome": "Restless leg syndrome",
    "Scoliosis": "Scoliosis",
    "SeborrhoeicDermatitis": "Seborrhoeic Dermatitis",
    "SensitivitytoMosquitoBites": "Sensitivity to Mosquito Bites",
    "SleepDisorders": "Sleep Disorders",
    "Strabismus": "Strabismus",
    "ThyroidIssuesCancer": "Thyroid Issues Cancer",
    "TypeIIDiabetes": "Type II Diabetes",
}


# ============================================================
# BASIC HELPERS
# ============================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def display_name(phenotype: str) -> str:
    return DISPLAY_NAMES.get(phenotype, phenotype)


def safe_filename(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def sorted_nicely(items: List[str]) -> List[str]:
    def convert(text):
        return int(text) if text.isdigit() else text

    def alphanum_key(key):
        return [convert(c) for c in re.split(r"([0-9]+)", key)]

    return sorted(items, key=alphanum_key)


def read_csv_safely(path: str, sep: Optional[str] = None) -> pd.DataFrame:
    attempts = [
        {"encoding": "utf-8"},
        {"encoding": "utf-8", "encoding_errors": "replace"},
        {"encoding": "latin1"},
    ]

    last_error = None

    for kwargs in attempts:
        try:
            if sep is None:
                return pd.read_csv(path, dtype=str, low_memory=False, **kwargs)
            return pd.read_csv(path, sep=sep, dtype=str, low_memory=False, **kwargs)
        except Exception as e:
            last_error = e

    raise RuntimeError(f"Could not read {path}. Last error: {last_error}")


def normalize_rsid(value) -> str:
    """
    Normalize SNP ID for matching.
    Handles:
      rs123
      rs123-A
      rs123_A
      123
      123-A
    """
    if value is None:
        return ""

    value = str(value).strip()

    if not value or value.lower() in {"nan", "none", "null", "na", "-", "[]"}:
        return ""

    # Remove allele suffixes commonly present in GWAS/PLINK-like names
    value = re.split(r"[-_]", value)[0].strip()

    m = re.search(r"\brs\d+\b", value, flags=re.IGNORECASE)
    if m:
        return m.group(0).lower()

    if re.fullmatch(r"\d+", value):
        return f"rs{value}".lower()

    return value.lower()


def split_possible_snps(value) -> List[str]:
    if value is None:
        return []

    value = str(value).strip()

    if not value or value.lower() in {"nan", "none", "null", "na", "-", "[]"}:
        return []

    parts = re.split(r"[,;/\s]+", value)
    snps = []

    for part in parts:
        snp = normalize_rsid(part)
        if snp:
            snps.append(snp)

    return sorted(set(snps))


def clean_gene(gene) -> str:
    if gene is None:
        return ""

    gene = str(gene).strip()

    if not gene or gene.lower() in {"nan", "none", "null", "na", "-", "nr", "[]"}:
        return ""

    return gene


def latex_escape(text: str) -> str:
    if text is None:
        return ""

    text = str(text)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


# ============================================================
# LOAD ML BENCHMARK RESULTS
# ============================================================
def load_algorithm_index() -> pd.DataFrame:
    if not os.path.exists(ML_ALGO_FILE):
        print(f"[WARNING] {ML_ALGO_FILE} not found. Algorithm names will not be resolved.")
        return pd.DataFrame(columns=["Algorithm Index for Reference", "Algorithm Name"])

    return pd.read_csv(ML_ALGO_FILE, sep="\t", dtype=str)


def resolve_algorithm_name(algo_value: str, algo_index: pd.DataFrame) -> str:
    if algo_value is None:
        return ""

    algo_value = str(algo_value).strip()

    if algo_index.empty:
        return algo_value

    if "Algorithm Index for Reference" not in algo_index.columns:
        return algo_value

    if "Algorithm Name" not in algo_index.columns:
        return algo_value

    match = algo_index[algo_index["Algorithm Index for Reference"].astype(str) == algo_value]

    if match.empty:
        return algo_value

    return str(match["Algorithm Name"].values[0])


def load_ml_results(metric: str, algo_index: pd.DataFrame) -> pd.DataFrame:
    path = ML_RESULT_FILES[metric]

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing ML result file: {path}")

    df = pd.read_csv(path, dtype=str)

    if "Phenotype" not in df.columns:
        raise RuntimeError(f"'Phenotype' column missing in {path}")

    if "Number of SNPs" not in df.columns:
        raise RuntimeError(f"'Number of SNPs' column missing in {path}")

    if "Machine learning algorithm index" not in df.columns:
        raise RuntimeError(f"'Machine learning algorithm index' column missing in {path}")

    df["Number of SNPs"] = pd.to_numeric(df["Number of SNPs"], errors="coerce").fillna(0).astype(int)
    df["Resolved Algorithm Name"] = df["Machine learning algorithm index"].apply(
        lambda x: resolve_algorithm_name(x, algo_index)
    )

    return df


def get_metric_row(ml_df: pd.DataFrame, phenotype: str) -> Optional[pd.Series]:
    row = ml_df[ml_df["Phenotype"].astype(str) == phenotype]

    if row.empty:
        return None

    return row.iloc[0]


def get_metric_score(row: pd.Series, metric: str) -> float:
    possible_cols = {
        "AUC": [
            "AUC",
            "Test AUC 5 Iterations Average",
            "Test AUC 5 Iterations Average ",
        ],
        "F1": [
            "F1 Score",
            "Test f1score 5 Iterations Average",
            "Test f1score 5 Iterations Average ",
        ],
        "MCC": [
            "MCC",
            "Test MCC 5 Iterations Average",
            "Test MCC 5 Iterations Average ",
        ],
    }

    for col in possible_cols[metric]:
        if col in row.index:
            val = pd.to_numeric(row[col], errors="coerce")
            if pd.notna(val):
                return float(val)

    return 0.0


# ============================================================
# FIND PV FOLDER AND LOAD IMPORTANCES
# ============================================================
def get_pv_folder(fold_path: str, target_n_snps: int) -> Optional[str]:
    if not os.path.isdir(fold_path):
        print(f"  WARNING: fold path missing: {fold_path}")
        return None

    pv_folders = [
        f for f in os.listdir(fold_path)
        if os.path.isdir(os.path.join(fold_path, f)) and f.startswith("pv_")
    ]

    if not pv_folders:
        print(f"  WARNING: no pv folders found in {fold_path}")
        return None

    # Sort by numeric pv value where possible
    def pv_key(name):
        try:
            return float(name.replace("pv_", ""))
        except Exception:
            return 999999.0

    pv_folders = sorted(pv_folders, key=pv_key)

    best_folder = None
    best_diff = float("inf")
    best_count = 0

    for pv_folder in pv_folders:
        pv_value = pv_folder.replace("pv_", "")
        snp_file = os.path.join(fold_path, pv_folder, f"{pv_value}.txt")

        if not os.path.exists(snp_file):
            continue

        with open(snp_file, "r", encoding="utf-8", errors="replace") as f:
            count = sum(1 for _ in f)

        diff = abs(count - int(target_n_snps))

        if diff < best_diff:
            best_diff = diff
            best_folder = pv_folder
            best_count = count

    if best_folder is None:
        print(f"  WARNING: could not match pv folder in {fold_path}")
        return None

    print(
        f"  Matched pv folder: {best_folder} "
        f"(target={target_n_snps}, observed={best_count}, diff={best_diff})"
    )

    return best_folder


def algorithm_filename_match(fname_without_ext: str, algorithm_name: str) -> bool:
    def clean(x):
        return re.sub(r"[-: ,_.()\[\]/\\]+", "", str(x)).lower()

    return clean(fname_without_ext) == clean(algorithm_name)


def load_fold_importances(fold_path: str, target_n_snps: int, algorithm_name: str) -> pd.DataFrame:
    pv_folder = get_pv_folder(fold_path, target_n_snps)

    if pv_folder is None:
        return pd.DataFrame(columns=["SNP", "SNP_Normalized", "Importance", "AbsImportance"])

    pv_dir = os.path.join(fold_path, pv_folder)
    pv_value = pv_folder.replace("pv_", "")
    snp_file = os.path.join(pv_dir, f"{pv_value}.txt")

    if not os.path.exists(snp_file):
        print(f"  WARNING: SNP list file not found: {snp_file}")
        return pd.DataFrame(columns=["SNP", "SNP_Normalized", "Importance", "AbsImportance"])

    snp_names = pd.read_csv(snp_file, header=None, dtype=str)[0].astype(str).tolist()

    target_file = None

    for fname in os.listdir(pv_dir):
        if not fname.endswith(".csv"):
            continue

        fname_without_ext = fname.replace(".csv", "")

        if algorithm_filename_match(fname_without_ext, algorithm_name):
            target_file = fname
            break

    if target_file is None:
        print(f"  WARNING: no importance file for algorithm '{algorithm_name}' in {pv_dir}")
        return pd.DataFrame(columns=["SNP", "SNP_Normalized", "Importance", "AbsImportance"])

    importance_path = os.path.join(pv_dir, target_file)

    try:
        imp_df = pd.read_csv(importance_path)
    except Exception as e:
        print(f"  WARNING: could not read importance file {importance_path}: {e}")
        return pd.DataFrame(columns=["SNP", "SNP_Normalized", "Importance", "AbsImportance"])

    if "Features_importance" not in imp_df.columns:
        print(f"  WARNING: 'Features_importance' missing in {importance_path}")
        return pd.DataFrame(columns=["SNP", "SNP_Normalized", "Importance", "AbsImportance"])

    weights = pd.to_numeric(imp_df["Features_importance"], errors="coerce").fillna(0).tolist()

    if len(snp_names) != len(weights):
        print(
            f"  WARNING: SNP count {len(snp_names)} != weight count {len(weights)} "
            f"in {importance_path}; truncating to minimum length."
        )
        min_len = min(len(snp_names), len(weights))
        snp_names = snp_names[:min_len]
        weights = weights[:min_len]

    out = pd.DataFrame({
        "SNP": snp_names,
        "SNP_Normalized": [normalize_rsid(x) for x in snp_names],
        "Importance": weights,
        "AbsImportance": [abs(float(x)) for x in weights],
    })

    out = out[out["SNP_Normalized"] != ""].reset_index(drop=True)

    return out


def aggregate_importance(phenotype: str, target_n_snps: int, algorithm_name: str) -> pd.DataFrame:
    print(f"  Aggregating feature importance: {algorithm_name} | SNPs={target_n_snps}")

    rows = []

    for fold in range(1, N_FOLDS + 1):
        fold_path = os.path.join(phenotype, str(fold))

        fold_df = load_fold_importances(
            fold_path=fold_path,
            target_n_snps=target_n_snps,
            algorithm_name=algorithm_name,
        )

        if fold_df.empty:
            continue

        fold_df["Fold"] = fold
        rows.append(fold_df)

    if not rows:
        print(f"  WARNING: no feature importances found for {phenotype} | {algorithm_name}")
        return pd.DataFrame(columns=[
            "SNP", "SNP_Normalized", "Importance", "AbsImportance", "Folds", "MeanAbsImportance"
        ])

    all_df = pd.concat(rows, ignore_index=True)

    # Non-zero in at least one fold
    all_df["NonZero"] = all_df["Importance"].astype(float) != 0.0

    grouped = all_df.groupby("SNP_Normalized", as_index=False).agg(
        SNP=("SNP", "first"),
        Importance=("Importance", lambda x: float(pd.to_numeric(x, errors="coerce").fillna(0).sum())),
        AbsImportance=("AbsImportance", lambda x: float(pd.to_numeric(x, errors="coerce").fillna(0).sum())),
        MeanAbsImportance=("AbsImportance", lambda x: float(pd.to_numeric(x, errors="coerce").fillna(0).mean())),
        Folds=("NonZero", lambda x: int(x.sum())),
    )

    grouped = grouped[grouped["Folds"] > 0].copy()
    grouped = grouped.sort_values(["AbsImportance", "Folds"], ascending=False).reset_index(drop=True)

    print(f"  Non-zero SNPs in any fold: {len(grouped)}")

    return grouped


# ============================================================
# LOAD NEW GWAS SELECTED FILES
# ============================================================
def load_gwas_summary(phenotype: str) -> dict:
    summary_path = os.path.join(GWAS_ROOT, phenotype, "GWAS_Summary.csv")

    if not os.path.exists(summary_path):
        print(f"  WARNING: GWAS summary missing: {summary_path}")
        return {}

    df = read_csv_safely(summary_path)

    if df.empty:
        return {}

    return df.iloc[0].to_dict()


def load_selected_common_variants(phenotype: str) -> pd.DataFrame:
    path = os.path.join(GWAS_ROOT, phenotype, "GWAS_Selected_Common_Variants.csv")

    if not os.path.exists(path):
        print(f"  WARNING: selected common variants file missing: {path}")
        return pd.DataFrame()

    df = read_csv_safely(path)

    if "COMMON_SNP" not in df.columns:
        print(f"  WARNING: COMMON_SNP column missing in {path}")
        return pd.DataFrame()

    df["COMMON_SNP_Normalized"] = df["COMMON_SNP"].apply(normalize_rsid)

    return df


def load_selected_annotated_gwas(phenotype: str) -> pd.DataFrame:
    path = os.path.join(GWAS_ROOT, phenotype, "GWAS_Selected_Annotated_Associations.csv")

    if not os.path.exists(path):
        print(f"  WARNING: selected annotated GWAS file missing: {path}")
        return pd.DataFrame()

    return read_csv_safely(path)


def extract_genes_from_gwas_rows(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return []

    gene_cols = [
        "MAPPED_GENE",
        "REPORTED GENE(S)",
        "UPSTREAM_GENE_ID",
        "DOWNSTREAM_GENE_ID",
        "SNP_GENE_IDS",
    ]

    genes = set()

    for col in gene_cols:
        if col not in df.columns:
            continue

        for value in df[col].dropna().astype(str):
            for part in re.split(r"[,;/|]", value):
                g = clean_gene(part)
                if g:
                    genes.add(g)

    return sorted(genes)


def build_snp_gene_map_from_common_rows(common_rows: pd.DataFrame) -> pd.DataFrame:
    if common_rows.empty:
        return pd.DataFrame(columns=["SNP", "Gene", "SourceColumn"])

    records = []

    gene_cols = [
        "MAPPED_GENE",
        "REPORTED GENE(S)",
        "UPSTREAM_GENE_ID",
        "DOWNSTREAM_GENE_ID",
        "SNP_GENE_IDS",
    ]

    for _, row in common_rows.iterrows():
        snp = normalize_rsid(row.get("COMMON_SNP", ""))

        if not snp:
            continue

        for col in gene_cols:
            if col not in row.index:
                continue

            value = row.get(col, "")

            for part in re.split(r"[,;/|]", str(value)):
                gene = clean_gene(part)
                if gene:
                    records.append({
                        "SNP": snp,
                        "Gene": gene,
                        "SourceColumn": col,
                    })

    if not records:
        return pd.DataFrame(columns=["SNP", "Gene", "SourceColumn"])

    return pd.DataFrame(records).drop_duplicates().reset_index(drop=True)


# ============================================================
# IDENTIFY SNPs AND GENES
# ============================================================
def identify_by_metric(
    phenotype: str,
    metric: str,
    ml_row: pd.Series,
    common_snp_set: Set[str],
    output_dir: str,
) -> Tuple[pd.DataFrame, Set[str]]:
    algorithm_name = str(ml_row["Resolved Algorithm Name"])
    algorithm_index = str(ml_row["Machine learning algorithm index"])
    target_n_snps = int(ml_row["Number of SNPs"])
    metric_score = get_metric_score(ml_row, metric)

    print(
        f"  Best {metric}: {algorithm_index} | {algorithm_name} | "
        f"SNPs={target_n_snps} | score={metric_score}"
    )

    imp_df = aggregate_importance(
        phenotype=phenotype,
        target_n_snps=target_n_snps,
        algorithm_name=algorithm_name,
    )

    imp_out = os.path.join(output_dir, f"FeatureImportance_{metric}.csv")
    imp_df.to_csv(imp_out, index=False)

    if imp_df.empty:
        identified = set()
    else:
        imp_snps = set(imp_df["SNP_Normalized"].dropna().astype(str))
        identified = imp_snps.intersection(common_snp_set)

    identified_df = pd.DataFrame({
        "SNP_Normalized": sorted(identified),
        "Metric": metric,
        "Algorithm Index": algorithm_index,
        "Algorithm Name": algorithm_name,
        "Selected SNP Count": target_n_snps,
        "Metric Score": metric_score,
    })

    if not imp_df.empty and identified:
        identified_df = identified_df.merge(
            imp_df,
            on="SNP_Normalized",
            how="left"
        )

    identified_out = os.path.join(output_dir, f"IdentifiedSNPs_{metric}.csv")
    identified_df.to_csv(identified_out, index=False)

    print(f"  Identified common SNPs by {metric}: {len(identified)}")

    return identified_df, identified


def map_identified_snps_to_genes(
    identified_snps: Set[str],
    snp_gene_map: pd.DataFrame,
) -> pd.DataFrame:
    if not identified_snps or snp_gene_map.empty:
        return pd.DataFrame(columns=["SNP", "Gene", "SourceColumn"])

    sub = snp_gene_map[snp_gene_map["SNP"].isin(identified_snps)].copy()
    sub = sub.drop_duplicates().sort_values(["Gene", "SNP"]).reset_index(drop=True)

    return sub


# ============================================================
# PROCESS ONE PHENOTYPE
# ============================================================
def process_phenotype(
    phenotype: str,
    ml_results: Dict[str, pd.DataFrame],
) -> dict:
    print("\n" + "=" * 100)
    print(f"Processing phenotype: {phenotype}")
    print(f"Display name        : {display_name(phenotype)}")

    pheno_out = os.path.join(OUTPUT_ROOT, phenotype)
    ensure_dir(pheno_out)

    gwas_summary = load_gwas_summary(phenotype)
    common_rows = load_selected_common_variants(phenotype)
    annotated_gwas = load_selected_annotated_gwas(phenotype)

    if common_rows.empty:
        print("  WARNING: no selected common GWAS rows found. Phenotype may have zero common SNPs.")

    common_snp_set = set(common_rows["COMMON_SNP_Normalized"].dropna().astype(str)) if not common_rows.empty else set()

    snp_gene_map = build_snp_gene_map_from_common_rows(common_rows)
    snp_gene_map_file = os.path.join(pheno_out, "SelectedGWAS_SNP_Gene_Map.csv")
    snp_gene_map.to_csv(snp_gene_map_file, index=False)

    # Also save selected GWAS files into final folder for traceability
    common_rows.to_csv(os.path.join(pheno_out, "SelectedGWAS_CommonRows.csv"), index=False)
    annotated_gwas.to_csv(os.path.join(pheno_out, "SelectedGWAS_AnnotatedRows.csv"), index=False)

    print(f"  Common SNPs in selected GWAS/data overlap: {len(common_snp_set)}")
    print(f"  SNP-gene mappings from selected common rows: {len(snp_gene_map)}")

    metric_identified_snps = {}
    metric_gene_maps = {}
    metric_summaries = {}

    for metric in ["AUC", "F1", "MCC"]:
        row = get_metric_row(ml_results[metric], phenotype)

        if row is None:
            print(f"  WARNING: no ML benchmark result for metric {metric}")
            metric_identified_snps[metric] = set()
            metric_gene_maps[metric] = pd.DataFrame(columns=["SNP", "Gene", "SourceColumn"])
            metric_summaries[metric] = {
                "algorithm_index": "",
                "algorithm_name": "",
                "selected_snp_count": 0,
                "score": 0.0,
            }
            continue

        identified_df, identified_set = identify_by_metric(
            phenotype=phenotype,
            metric=metric,
            ml_row=row,
            common_snp_set=common_snp_set,
            output_dir=pheno_out,
        )

        mapped = map_identified_snps_to_genes(identified_set, snp_gene_map)
        mapped_file = os.path.join(pheno_out, f"IdentifiedGenes_{metric}.csv")
        mapped.to_csv(mapped_file, index=False)

        metric_identified_snps[metric] = identified_set
        metric_gene_maps[metric] = mapped
        metric_summaries[metric] = {
            "algorithm_index": str(row["Machine learning algorithm index"]),
            "algorithm_name": str(row["Resolved Algorithm Name"]),
            "selected_snp_count": int(row["Number of SNPs"]),
            "score": get_metric_score(row, metric),
        }

    # Union across AUC, F1, MCC
    union_snps = (
        metric_identified_snps["AUC"]
        | metric_identified_snps["F1"]
        | metric_identified_snps["MCC"]
    )

    union_gene_map = map_identified_snps_to_genes(union_snps, snp_gene_map)
    union_genes = sorted(set(union_gene_map["Gene"].dropna().astype(str))) if not union_gene_map.empty else []

    union_snp_file = os.path.join(pheno_out, "IdentifiedSNPs_Union.csv")
    union_gene_file = os.path.join(pheno_out, "IdentifiedGenes_Union.csv")
    union_gene_list_file = os.path.join(pheno_out, "IdentifiedGeneList_Union.txt")

    pd.DataFrame({"SNP": sorted(union_snps)}).to_csv(union_snp_file, index=False)
    union_gene_map.to_csv(union_gene_file, index=False)

    with open(union_gene_list_file, "w", encoding="utf-8") as f:
        for gene in union_genes:
            f.write(gene + "\n")

    total_gwas_genes = int(gwas_summary.get("Unique genes", 0) or 0)

    if total_gwas_genes == 0:
        total_gwas_genes = len(extract_genes_from_gwas_rows(common_rows))

    gir = round(len(union_genes) / total_gwas_genes, 4) if total_gwas_genes > 0 else 0.0

    result = {
        "Phenotype": phenotype,
        "Phenotype Display": display_name(phenotype),

        "GWAS Catalog ID": gwas_summary.get("GWAS Catalog ID", ""),
        "GWAS Catalog Trait": gwas_summary.get("GWAS Catalog Trait", ""),
        "GWAS SNPs": int(float(gwas_summary.get("SNPs in GWAS Catalogue", 0) or 0)),
        "Dataset SNPs": int(float(gwas_summary.get("SNPs in our data", 0) or 0)),
        "Common SNPs GWAS vs Dataset": int(float(gwas_summary.get("Common SNPs", len(common_snp_set)) or 0)),
        "GWAS Genes from Common Rows": total_gwas_genes,

        "AUC Algorithm": metric_summaries["AUC"]["algorithm_name"],
        "AUC SNP Count": metric_summaries["AUC"]["selected_snp_count"],
        "AUC Score": metric_summaries["AUC"]["score"],
        "Identified SNPs AUC": len(metric_identified_snps["AUC"]),
        "Identified Genes AUC": len(set(metric_gene_maps["AUC"]["Gene"].dropna().astype(str))) if not metric_gene_maps["AUC"].empty else 0,

        "F1 Algorithm": metric_summaries["F1"]["algorithm_name"],
        "F1 SNP Count": metric_summaries["F1"]["selected_snp_count"],
        "F1 Score": metric_summaries["F1"]["score"],
        "Identified SNPs F1": len(metric_identified_snps["F1"]),
        "Identified Genes F1": len(set(metric_gene_maps["F1"]["Gene"].dropna().astype(str))) if not metric_gene_maps["F1"].empty else 0,

        "MCC Algorithm": metric_summaries["MCC"]["algorithm_name"],
        "MCC SNP Count": metric_summaries["MCC"]["selected_snp_count"],
        "MCC Score": metric_summaries["MCC"]["score"],
        "Identified SNPs MCC": len(metric_identified_snps["MCC"]),
        "Identified Genes MCC": len(set(metric_gene_maps["MCC"]["Gene"].dropna().astype(str))) if not metric_gene_maps["MCC"].empty else 0,

        "Identified SNPs Union": len(union_snps),
        "Identified Genes Union": len(union_genes),
        "GIR": gir,

        "Identified SNP List": ";".join(sorted(union_snps)),
        "Identified Gene List": ";".join(union_genes),

        "SNP Gene Map File": snp_gene_map_file,
        "Union SNP File": union_snp_file,
        "Union Gene File": union_gene_file,
        "Status": "OK",
    }

    result_file = os.path.join(pheno_out, "GeneIdentification_Summary.json")
    result_csv = os.path.join(pheno_out, "GeneIdentification_Summary.csv")

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    pd.DataFrame([result]).to_csv(result_csv, index=False)

    print("\n  Final phenotype summary:")
    print(f"    GWAS ID              : {result['GWAS Catalog ID']}")
    print(f"    GWAS trait           : {result['GWAS Catalog Trait']}")
    print(f"    Common SNPs          : {result['Common SNPs GWAS vs Dataset']}")
    print(f"    Identified SNP union : {result['Identified SNPs Union']}")
    print(f"    Identified genes     : {result['Identified Genes Union']}")
    print(f"    Total GWAS genes     : {result['GWAS Genes from Common Rows']}")
    print(f"    GIR                  : {result['GIR']}")

    return result


# ============================================================
# OUTPUT TABLES
# ============================================================
def write_markdown(df: pd.DataFrame, out_file: str) -> None:
    cols = [
        "Phenotype Display",
        "GWAS Catalog ID",
        "GWAS Catalog Trait",
        "Common SNPs GWAS vs Dataset",
        "Identified SNPs Union",
        "Identified Genes Union",
        "GWAS Genes from Common Rows",
        "GIR",
    ]

    available = [c for c in cols if c in df.columns]

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(df[available].to_markdown(index=False))
        f.write("\n")


def write_latex(df: pd.DataFrame, out_file: str) -> None:
    lines = []

    lines.append(r"\begin{table*}[!ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{|l|l|l|r|r|r|r|r|}")
    lines.append(r"\hline")
    lines.append(
        r"\textbf{Phenotype} & \textbf{GWAS ID} & \textbf{GWAS trait} & "
        r"\textbf{GWAS--data SNPs} & \textbf{Identified SNPs} & "
        r"\textbf{Identified genes} & \textbf{GWAS genes} & \textbf{GIR} \\ \hline"
    )

    for _, row in df.iterrows():
        phenotype = latex_escape(row.get("Phenotype Display", ""))
        gwas_id = latex_escape(row.get("GWAS Catalog ID", ""))
        gwas_trait = latex_escape(row.get("GWAS Catalog Trait", ""))
        common = int(float(row.get("Common SNPs GWAS vs Dataset", 0) or 0))
        identified_snps = int(float(row.get("Identified SNPs Union", 0) or 0))
        identified_genes = int(float(row.get("Identified Genes Union", 0) or 0))
        gwas_genes = int(float(row.get("GWAS Genes from Common Rows", 0) or 0))
        gir = float(row.get("GIR", 0) or 0)

        lines.append(
            f"{phenotype} & {gwas_id} & {gwas_trait} & "
            f"{common} & {identified_snps} & {identified_genes} & {gwas_genes} & {gir:.3f} \\\\ \\hline"
        )

    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(
        r"\caption{\textbf{Final gene-identification results using selected GWAS Catalog association sets.} "
        r"For each phenotype, SNPs prioritised by the best-performing machine-learning models according to AUC, F1 score, "
        r"and MCC were intersected with the SNPs shared between the selected GWAS Catalog association set and the processed genotype dataset. "
        r"Identified genes were obtained by mapping the union of identified SNPs across the three metrics to genes from the selected GWAS rows. "
        r"GIR denotes the gene-identification ratio, calculated as the number of identified genes divided by the number of GWAS-linked genes "
        r"available from overlapping GWAS rows.}"
    )
    lines.append(r"\label{tab:final_gene_identification_results}")
    lines.append(r"\end{table*}")

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Identify genes using ML feature importance and new GWASCatalogDownloaded selected GWAS files."
    )

    parser.add_argument(
        "--phenotype",
        default=None,
        help="Run one phenotype only, e.g. Depression.",
    )

    args = parser.parse_args()

    ensure_dir(OUTPUT_ROOT)

    algo_index = load_algorithm_index()

    print("Loading ML benchmark results...")
    ml_results = {
        metric: load_ml_results(metric, algo_index)
        for metric in ["AUC", "F1", "MCC"]
    }

    if args.phenotype:
        phenotypes = [args.phenotype]
    else:
        phenotypes = sorted(ml_results["AUC"]["Phenotype"].dropna().astype(str).unique().tolist())

    print(f"Phenotypes to process: {len(phenotypes)}")

    results = []

    for phenotype in phenotypes:
        try:
            result = process_phenotype(
                phenotype=phenotype,
                ml_results=ml_results,
            )
            results.append(result)

        except Exception as e:
            print(f"[ERROR] Failed phenotype {phenotype}: {e}")

            results.append({
                "Phenotype": phenotype,
                "Phenotype Display": display_name(phenotype),
                "GWAS Catalog ID": "",
                "GWAS Catalog Trait": "",
                "GWAS SNPs": 0,
                "Dataset SNPs": 0,
                "Common SNPs GWAS vs Dataset": 0,
                "GWAS Genes from Common Rows": 0,
                "Identified SNPs Union": 0,
                "Identified Genes Union": 0,
                "GIR": 0,
                "Status": f"FAILED: {e}",
            })

    final_df = pd.DataFrame(results)

    final_csv = os.path.join(OUTPUT_ROOT, "Final_Gene_Identification_Results.csv")
    final_html = os.path.join(OUTPUT_ROOT, "Final_Gene_Identification_Results.html")
    final_md = os.path.join(OUTPUT_ROOT, "Final_Gene_Identification_Results.md")
    final_tex = os.path.join(OUTPUT_ROOT, "Final_Gene_Identification_Table.tex")
    final_json = os.path.join(OUTPUT_ROOT, "Final_Gene_Identification_Results.json")

    final_df.to_csv(final_csv, index=False)
    final_df.to_html(final_html, index=False)

    with open(final_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    write_markdown(final_df, final_md)
    write_latex(final_df, final_tex)

    print("\n" + "=" * 100)
    print("FINAL GENE IDENTIFICATION SUMMARY")
    print("=" * 100)

    show_cols = [
        "Phenotype Display",
        "GWAS Catalog ID",
        "Common SNPs GWAS vs Dataset",
        "Identified SNPs Union",
        "Identified Genes Union",
        "GWAS Genes from Common Rows",
        "GIR",
        "Status",
    ]

    show_cols = [c for c in show_cols if c in final_df.columns]
    print(final_df[show_cols].to_string(index=False))

    print("\nFiles written:")
    print(f"  {final_csv}")
    print(f"  {final_html}")
    print(f"  {final_md}")
    print(f"  {final_tex}")
    print(f"  {final_json}")
    print("=" * 100)


if __name__ == "__main__":
    main()
