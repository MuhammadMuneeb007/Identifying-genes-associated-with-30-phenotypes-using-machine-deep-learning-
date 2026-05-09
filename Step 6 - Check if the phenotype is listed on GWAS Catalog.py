#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import zipfile
import argparse
import shutil
from typing import Dict, List

import requests
import pandas as pd


# ============================================================
# STEP 6.0 — FIND AND SAVE GWAS CATALOG ASSOCIATION FILES
# ============================================================
# This script:
#   1. Downloads full GWAS Catalog association dump if needed.
#   2. Unzips it automatically.
#   3. Searches phenotype-relevant GWAS trait columns.
#   4. Lists candidate GWAS Catalog trait IDs per phenotype.
#   5. Saves all candidate association files.
#   6. Saves old-pipeline-compatible outputs under:
#        GWASCatalogDownloaded/<Phenotype>/
# ============================================================


# ============================================================
# PHENOTYPES
# ============================================================
PHENOTYPES = [
    "ADHD",
    "Allergicrhinitis",
    "Amblyopia",
    "Asthma",
    "Astigmatism",
    "Bipolardisorder",
    "Cholesterol",
    "clusterheadache",
    "Cravessugar",
    "Dentaldecay",
    "Depression",
    "DiagnosedVitaminDdeficiency",
    "DiagnosedwithSleepApnea",
    "Dyslexia",
    "EarlobeFreeorattached",
    "eczema",
    "generalizedanxietydisorder",
    "HairType",
    "HaveMECFS",
    "Hypertension",
    "Hypertriglyceridemia",
    "IrritableBowelSyndrome",
    "MentalDisease",
    "Migraine",
    "Motionsickness",
    "PanicDisorder",
    "PhoticSneezeReflexPhotoptarmis",
    "Plantarfasciitis",
    "PosttraumaticStressDisorderorPTSD",
    "restlesslegsyndrome",
    "Scoliosis",
    "SeborrhoeicDermatitis",
    "SensitivitytoMosquitoBites",
    "SleepDisorders",
    "Strabismus",
    "ThyroidIssuesCancer",
    "TypeIIDiabetes",
]


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


SEARCH_TERMS: Dict[str, List[str]] = {
    "ADHD": ["ADHD", "attention deficit hyperactivity disorder", "attention deficit disorder"],
    "Allergicrhinitis": ["allergic rhinitis", "hay fever", "rhinitis"],
    "Amblyopia": ["amblyopia", "lazy eye"],
    "Asthma": ["asthma"],
    "Astigmatism": ["astigmatism"],
    "Bipolardisorder": ["bipolar disorder", "bipolar", "manic depression"],
    "Cholesterol": ["cholesterol", "total cholesterol", "LDL cholesterol", "HDL cholesterol"],
    "clusterheadache": ["cluster headache"],
    "Cravessugar": ["sugar craving", "sweet taste", "sweet preference", "sugar intake"],
    "Dentaldecay": ["dental decay", "dental caries", "tooth decay", "caries"],
    "Depression": ["depression", "major depressive disorder", "depressive symptoms", "unipolar depression", "depressive disorder"],
    "DiagnosedVitaminDdeficiency": ["vitamin D deficiency", "vitamin D", "25 hydroxyvitamin D", "25-hydroxyvitamin D", "25(OH)D"],
    "DiagnosedwithSleepApnea": ["sleep apnea", "sleep apnoea", "obstructive sleep apnea", "obstructive sleep apnoea"],
    "Dyslexia": ["dyslexia", "reading disability", "reading disorder"],
    "EarlobeFreeorattached": ["earlobe", "ear lobe", "attached earlobe", "free earlobe"],
    "eczema": ["eczema", "atopic dermatitis"],
    "generalizedanxietydisorder": ["generalized anxiety disorder", "generalised anxiety disorder", "anxiety disorder"],
    "HairType": ["hair type", "hair morphology", "hair shape", "hair curl", "hair texture"],
    "HaveMECFS": ["chronic fatigue syndrome", "myalgic encephalomyelitis", "ME/CFS"],
    "Hypertension": ["hypertension", "blood pressure", "systolic blood pressure", "diastolic blood pressure"],
    "Hypertriglyceridemia": ["hypertriglyceridemia", "triglycerides", "triglyceride"],
    "IrritableBowelSyndrome": ["irritable bowel syndrome", "IBS"],
    "MentalDisease": ["mental disorder", "mental disease", "psychiatric disorder", "psychiatric disease"],
    "Migraine": ["migraine"],
    "Motionsickness": ["motion sickness"],
    "PanicDisorder": ["panic disorder", "panic attack"],
    "PhoticSneezeReflexPhotoptarmis": ["photic sneeze reflex", "sneeze reflex", "ACHOO", "photoptarmic"],
    "Plantarfasciitis": ["plantar fasciitis"],
    "PosttraumaticStressDisorderorPTSD": ["post-traumatic stress disorder", "posttraumatic stress disorder", "PTSD"],
    "restlesslegsyndrome": ["restless leg syndrome", "restless legs syndrome"],
    "Scoliosis": ["scoliosis"],
    "SeborrhoeicDermatitis": ["seborrhoeic dermatitis", "seborrheic dermatitis"],
    "SensitivitytoMosquitoBites": ["mosquito bite", "mosquito bites", "insect bite reaction", "mosquito bite reaction"],
    "SleepDisorders": ["sleep disorder", "sleep disorders", "sleep duration", "insomnia", "sleep quality"],
    "Strabismus": ["strabismus", "squint"],
    "ThyroidIssuesCancer": ["thyroid cancer", "thyroid disease", "thyroid disorder", "thyroid"],
    "TypeIIDiabetes": ["type 2 diabetes", "type II diabetes", "T2D", "diabetes mellitus type 2", "type 2 diabetes mellitus"],
}


# ============================================================
# PATHS
# ============================================================
SEARCH_OUTDIR = "Identify6.0_GWAS_Search"
COMPAT_OUTDIR = "GWASCatalogDownloaded"

ZIP_FILE = os.path.join(SEARCH_OUTDIR, "gwas_catalog_associations.zip")
EXTRACTED_DIR = os.path.join(SEARCH_OUTDIR, "Extracted")
CATALOG_FILE = os.path.join(SEARCH_OUTDIR, "gwas_catalog_associations_full.tsv")

BEST_OUTPUT = os.path.join(SEARCH_OUTDIR, "Identify6.0_find_GWAS_best_candidates.tsv")
ALL_CANDIDATES_OUTPUT = os.path.join(SEARCH_OUTDIR, "Identify6.0_all_GWAS_candidates.tsv")
SUMMARY_JSON = os.path.join(SEARCH_OUTDIR, "Identify6.0_find_GWAS_summary.json")

GWAS_DOWNLOAD_URLS = [
    "https://www.ebi.ac.uk/gwas/api/search/downloads/associations/v1.0.2?split=false",
    "https://www.ebi.ac.uk/gwas/api/search/downloads/full",
    "https://www.ebi.ac.uk/gwas/api/search/downloads/alternative",
]


# ============================================================
# HELPERS
# ============================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_filename(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def display_name(phenotype: str) -> str:
    return DISPLAY_NAMES.get(phenotype, phenotype)


def normalize_text(text) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_trait_ids_from_uri(uri: str) -> List[str]:
    if pd.isna(uri):
        return []
    return re.findall(r"(EFO_\d+|MONDO_\d+|HP_\d+)", str(uri))


def extract_first_trait_id(uri: str) -> str:
    ids = extract_trait_ids_from_uri(uri)
    return ids[0] if ids else ""


def count_unique_snps(df: pd.DataFrame) -> int:
    for col in ["SNPS", "SNP_ID_CURRENT", "STRONGEST SNP-RISK ALLELE"]:
        if col in df.columns:
            vals = df[col].dropna().astype(str).str.strip()
            vals = vals[vals != ""]
            return int(vals.nunique())
    return int(len(df))


def extract_gene_set(df: pd.DataFrame) -> List[str]:
    gene_cols = ["MAPPED_GENE", "REPORTED GENE(S)", "UPSTREAM_GENE_ID", "DOWNSTREAM_GENE_ID", "SNP_GENE_IDS"]
    genes = set()

    for col in gene_cols:
        if col not in df.columns:
            continue

        for value in df[col].dropna().astype(str):
            for part in re.split(r"[,;/|]", value):
                g = part.strip()
                if g and g.lower() not in {"na", "nan", "none", "null", "-", "nr"}:
                    genes.add(g)

    return sorted(genes)


def trait_page(candidate_id: str) -> str:
    return f"https://www.ebi.ac.uk/gwas/efotraits/{candidate_id}" if candidate_id else ""


# ============================================================
# DOWNLOAD / UNZIP
# ============================================================
def download_file(url: str, outfile: str) -> bool:
    print("\nTrying download:")
    print(url)

    try:
        with requests.get(url, stream=True, timeout=600) as r:
            print(f"HTTP status: {r.status_code}")
            r.raise_for_status()

            tmp = outfile + ".tmp"
            total = 0

            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        total += len(chunk)

            if total < 1000:
                print(f"Downloaded file too small: {total} bytes")
                return False

            os.replace(tmp, outfile)
            print(f"Saved: {outfile}")
            print(f"Size: {total / (1024 * 1024):.2f} MB")
            return True

    except Exception as e:
        print(f"Download failed: {e}")
        return False


def find_largest_table_file(folder: str) -> str:
    files = []
    for root, _, names in os.walk(folder):
        for name in names:
            if name.lower().endswith((".tsv", ".txt", ".csv")):
                path = os.path.join(root, name)
                files.append((os.path.getsize(path), path))

    if not files:
        raise RuntimeError(f"No TSV/TXT/CSV file found in {folder}")

    files.sort(reverse=True)
    return files[0][1]


def prepare_catalog(force_download: bool = False) -> None:
    ensure_dir(SEARCH_OUTDIR)
    ensure_dir(EXTRACTED_DIR)

    if os.path.exists(CATALOG_FILE) and not force_download:
        print(f"Using existing prepared GWAS Catalog file: {CATALOG_FILE}")
        return

    if force_download:
        for p in [ZIP_FILE, CATALOG_FILE]:
            if os.path.exists(p):
                os.remove(p)
        if os.path.exists(EXTRACTED_DIR):
            shutil.rmtree(EXTRACTED_DIR)
        ensure_dir(EXTRACTED_DIR)

    if not os.path.exists(ZIP_FILE):
        ok = False
        for url in GWAS_DOWNLOAD_URLS:
            if download_file(url, ZIP_FILE):
                ok = True
                break
        if not ok:
            raise RuntimeError("Could not download GWAS Catalog associations.")

    if zipfile.is_zipfile(ZIP_FILE):
        print(f"\nUnzipping: {ZIP_FILE}")
        with zipfile.ZipFile(ZIP_FILE, "r") as z:
            for info in z.infolist():
                print(f"  {info.filename} | {info.file_size / (1024 * 1024):.2f} MB")
            z.extractall(EXTRACTED_DIR)

        extracted = find_largest_table_file(EXTRACTED_DIR)
    else:
        extracted = ZIP_FILE

    print(f"\nUsing extracted table: {extracted}")

    with open(extracted, "rb") as src, open(CATALOG_FILE, "wb") as dst:
        shutil.copyfileobj(src, dst)

    print(f"Prepared catalog file: {CATALOG_FILE}")
    print(f"Size: {os.path.getsize(CATALOG_FILE) / (1024 * 1024):.2f} MB")


# ============================================================
# LOAD / STANDARDIZE
# ============================================================
def load_catalog() -> pd.DataFrame:
    print(f"\nLoading GWAS Catalog table: {CATALOG_FILE}")

    attempts = [
        ("utf-8", "strict"),
        ("utf-8", "replace"),
        ("latin1", "strict"),
    ]

    last_error = None

    for encoding, errors in attempts:
        try:
            print(f"Trying encoding={encoding}, errors={errors}")
            df = pd.read_csv(
                CATALOG_FILE,
                sep="\t",
                dtype=str,
                low_memory=False,
                encoding=encoding,
                encoding_errors=errors,
            )
            break
        except Exception as e:
            last_error = e
            print(f"Failed: {e}")
    else:
        raise RuntimeError(f"Could not load catalog file. Last error: {last_error}")

    df.columns = [str(c).strip() for c in df.columns]

    if "MAPPED_TRAIT_URI" in df.columns:
        df["CANDIDATE_TRAIT_ID"] = df["MAPPED_TRAIT_URI"].apply(extract_first_trait_id)
    else:
        df["CANDIDATE_TRAIT_ID"] = ""

    for col in ["DISEASE/TRAIT", "MAPPED_TRAIT", "MAPPED_TRAIT_URI"]:
        if col not in df.columns:
            df[col] = ""

    print(f"Rows loaded: {len(df):,}")
    print(f"Columns loaded: {len(df.columns):,}")
    print("Important columns:")
    for col in ["DISEASE/TRAIT", "MAPPED_TRAIT", "MAPPED_TRAIT_URI", "SNPS", "MAPPED_GENE", "STUDY ACCESSION"]:
        print(f"  {'FOUND' if col in df.columns else 'MISSING'}: {col}")

    return df


# ============================================================
# MATCHING
# ============================================================
def build_trait_search_blob(df: pd.DataFrame) -> pd.Series:
    """
    IMPORTANT:
    Do not use STUDY here. It creates false positives.
    Example: Asthma matched breastfeeding/gut microbiome because asthma appeared in study text.
    """
    blob = (
        df["DISEASE/TRAIT"].fillna("").astype(str)
        + " "
        + df["MAPPED_TRAIT"].fillna("").astype(str)
        + " "
        + df["MAPPED_TRAIT_URI"].fillna("").astype(str)
    )
    return blob.apply(normalize_text)


def match_rows(df: pd.DataFrame, search_blob: pd.Series, phenotype: str) -> pd.DataFrame:
    terms = SEARCH_TERMS.get(phenotype, [phenotype])
    terms_norm = [normalize_text(t) for t in terms if normalize_text(t)]

    mask = pd.Series(False, index=df.index)

    for term in terms_norm:
        mask = mask | search_blob.str.contains(re.escape(term), na=False)

    matched = df.loc[mask].copy()
    matched["INPUT_PHENOTYPE"] = phenotype
    matched["INPUT_PHENOTYPE_DISPLAY"] = display_name(phenotype)
    matched["SEARCH_TERMS_USED"] = "; ".join(terms)

    return matched


def score_candidate(candidate_trait: str, phenotype: str) -> int:
    """
    Higher score = more phenotype-specific.
    This prevents broad unrelated terms from winning only by row count.
    """
    trait_norm = normalize_text(candidate_trait)
    terms_norm = [normalize_text(t) for t in SEARCH_TERMS.get(phenotype, [phenotype])]

    score = 0

    for term in terms_norm:
        if not term:
            continue
        if trait_norm == term:
            score += 100
        elif term in trait_norm:
            score += 50
        elif trait_norm in term:
            score += 30

    return score


def summarize_candidates(matched: pd.DataFrame, phenotype: str) -> pd.DataFrame:
    if matched.empty:
        return pd.DataFrame([{
            "Input_Phenotype": phenotype,
            "Input_Phenotype_Display": display_name(phenotype),
            "Candidate_Rank": 1,
            "Candidate_ID": "",
            "Candidate_Trait": "",
            "Trait_Match_Score": 0,
            "N_Association_Rows": 0,
            "N_Unique_SNPs": 0,
            "N_Unique_Studies": 0,
            "N_Unique_PubMed": 0,
            "N_Unique_Genes": 0,
            "Example_Reported_Traits": "",
            "Search_Terms_Used": "; ".join(SEARCH_TERMS.get(phenotype, [phenotype])),
            "Trait_Page": "",
        }])

    rows = []

    for (candidate_id, candidate_trait), g in matched.groupby(["CANDIDATE_TRAIT_ID", "MAPPED_TRAIT"], dropna=False):
        candidate_id = "" if pd.isna(candidate_id) else str(candidate_id)
        candidate_trait = "" if pd.isna(candidate_trait) else str(candidate_trait)

        reported = ""
        if "DISEASE/TRAIT" in g.columns:
            reported = "; ".join(sorted(set(g["DISEASE/TRAIT"].dropna().astype(str).str.strip())))[:1500]

        n_studies = int(g["STUDY ACCESSION"].nunique()) if "STUDY ACCESSION" in g.columns else 0
        n_pubmed = int(g["PUBMEDID"].nunique()) if "PUBMEDID" in g.columns else 0
        genes = extract_gene_set(g)

        rows.append({
            "Input_Phenotype": phenotype,
            "Input_Phenotype_Display": display_name(phenotype),
            "Candidate_ID": candidate_id,
            "Candidate_Trait": candidate_trait,
            "Trait_Match_Score": score_candidate(candidate_trait, phenotype),
            "N_Association_Rows": int(len(g)),
            "N_Unique_SNPs": count_unique_snps(g),
            "N_Unique_Studies": n_studies,
            "N_Unique_PubMed": n_pubmed,
            "N_Unique_Genes": len(genes),
            "Example_Reported_Traits": reported,
            "Search_Terms_Used": "; ".join(SEARCH_TERMS.get(phenotype, [phenotype])),
            "Trait_Page": trait_page(candidate_id),
        })

    out = pd.DataFrame(rows)

    out = out.sort_values(
        by=[
            "Trait_Match_Score",
            "N_Association_Rows",
            "N_Unique_SNPs",
            "N_Unique_Studies",
            "N_Unique_Genes",
        ],
        ascending=False,
    ).reset_index(drop=True)

    out.insert(2, "Candidate_Rank", range(1, len(out) + 1))

    return out


# ============================================================
# SAVE COMPATIBLE OUTPUTS
# ============================================================
def save_phenotype_outputs(matched: pd.DataFrame, candidates: pd.DataFrame, phenotype: str) -> dict:
    pheno_dir = os.path.join(COMPAT_OUTDIR, phenotype)
    candidate_dir = os.path.join(pheno_dir, "Candidates")

    ensure_dir(pheno_dir)
    ensure_dir(candidate_dir)

    # Save all matched associations
    all_matched_csv = os.path.join(pheno_dir, "GWAS_AllMatched_Associations.csv")
    matched.to_csv(all_matched_csv, index=False)

    # Save candidate index
    candidate_index_csv = os.path.join(pheno_dir, "GWAS_Candidate_Index.csv")
    candidates.to_csv(candidate_index_csv, index=False)

    # Save each candidate association file
    saved_candidate_files = []

    if not matched.empty:
        for (candidate_id, candidate_trait), g in matched.groupby(["CANDIDATE_TRAIT_ID", "MAPPED_TRAIT"], dropna=False):
            candidate_id = "" if pd.isna(candidate_id) else str(candidate_id)
            candidate_trait = "" if pd.isna(candidate_trait) else str(candidate_trait)

            if not candidate_id:
                candidate_id = "NO_TRAIT_ID"

            candidate_file = os.path.join(
                candidate_dir,
                f"{safe_filename(phenotype)}__{safe_filename(candidate_id)}__associations.csv"
            )

            g.to_csv(candidate_file, index=False)

            genes = extract_gene_set(g)
            gene_file = os.path.join(
                candidate_dir,
                f"{safe_filename(phenotype)}__{safe_filename(candidate_id)}__genes.csv"
            )
            pd.DataFrame({"gene": genes}).to_csv(gene_file, index=False)

            saved_candidate_files.append({
                "candidate_id": candidate_id,
                "candidate_trait": candidate_trait,
                "associations_file": candidate_file,
                "genes_file": gene_file,
                "n_rows": len(g),
                "n_unique_snps": count_unique_snps(g),
                "n_unique_genes": len(genes),
            })

    # Save best candidate files in predictable names
    best = candidates.iloc[0].to_dict()
    best_id = str(best.get("Candidate_ID", ""))

    if best_id and not matched.empty:
        best_rows = matched[matched["CANDIDATE_TRAIT_ID"] == best_id].copy()
    else:
        best_rows = pd.DataFrame(columns=matched.columns)

    best_assoc_csv = os.path.join(pheno_dir, "GWAS_Best_Associations.csv")
    best_rows.to_csv(best_assoc_csv, index=False)

    best_genes = extract_gene_set(best_rows)
    best_genes_csv = os.path.join(pheno_dir, "GWAS_Best_Genes.csv")
    best_genes_txt = os.path.join(pheno_dir, "GWAS_Best_Genes.txt")

    pd.DataFrame({"gene": best_genes}).to_csv(best_genes_csv, index=False)
    with open(best_genes_txt, "w", encoding="utf-8") as f:
        for gene in best_genes:
            f.write(gene + "\n")

    # Save old-style summary
    summary = {
        "Phenotype": phenotype,
        "Phenotype_Display": display_name(phenotype),
        "Best_GWAS_Catalog_ID": best_id,
        "Best_GWAS_Trait": best.get("Candidate_Trait", ""),
        "Best_Trait_Page": best.get("Trait_Page", ""),
        "SNPs in GWAS Catalogue": int(best.get("N_Unique_SNPs", 0)),
        "Association rows": int(best.get("N_Association_Rows", 0)),
        "Unique genes": int(best.get("N_Unique_Genes", 0)),
        "Number of candidate files": len(saved_candidate_files),
        "Search terms": "; ".join(SEARCH_TERMS.get(phenotype, [phenotype])),
        "Files": {
            "all_matched_associations": all_matched_csv,
            "candidate_index": candidate_index_csv,
            "best_associations": best_assoc_csv,
            "best_genes_csv": best_genes_csv,
            "best_genes_txt": best_genes_txt,
            "candidate_folder": candidate_dir,
        },
    }

    summary_json = os.path.join(pheno_dir, "GWAS_Best_Summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # CSV version used by your table code
    summary_csv = os.path.join(pheno_dir, "GWAS_Summary.csv")
    pd.DataFrame([{
        "Phenotype": phenotype,
        "GWAS Catalog ID": best_id,
        "GWAS Catalog Trait": best.get("Candidate_Trait", ""),
        "SNPs in GWAS Catalogue": int(best.get("N_Unique_SNPs", 0)),
        "Association rows": int(best.get("N_Association_Rows", 0)),
        "Unique genes": int(best.get("N_Unique_Genes", 0)),
        "SNPs in our data": 0,
        "Common SNPs": 0,
    }]).to_csv(summary_csv, index=False)

    return summary


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Step 6.0: Find GWAS Catalog candidate association files and save old-pipeline-compatible outputs."
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    ensure_dir(SEARCH_OUTDIR)
    ensure_dir(COMPAT_OUTDIR)

    prepare_catalog(force_download=args.force_download)

    df = load_catalog()
    search_blob = build_trait_search_blob(df)

    all_candidates = []
    best_rows = []
    final_summary = {}

    for phenotype in PHENOTYPES:
        print("\n" + "=" * 100)
        print(f"Searching phenotype: {phenotype}")
        print(f"Display name       : {display_name(phenotype)}")
        print(f"Search terms       : {SEARCH_TERMS.get(phenotype, [phenotype])}")

        matched = match_rows(df, search_blob, phenotype)
        print(f"Matched association rows: {len(matched):,}")

        candidates = summarize_candidates(matched, phenotype)
        top_candidates = candidates.head(args.top_n).copy()

        all_candidates.append(top_candidates)
        best_rows.append(candidates.iloc[0].to_dict())

        summary = save_phenotype_outputs(matched, candidates, phenotype)
        final_summary[phenotype] = summary

        best = candidates.iloc[0].to_dict()
        print("\nSelected best candidate:")
        print(f"  ID          : {best.get('Candidate_ID', '')}")
        print(f"  Trait       : {best.get('Candidate_Trait', '')}")
        print(f"  Match score : {best.get('Trait_Match_Score', 0)}")
        print(f"  Rows        : {best.get('N_Association_Rows', 0)}")
        print(f"  Unique SNPs : {best.get('N_Unique_SNPs', 0)}")
        print(f"  Studies     : {best.get('N_Unique_Studies', 0)}")
        print(f"  Genes       : {best.get('N_Unique_Genes', 0)}")
        print(f"  Page        : {best.get('Trait_Page', '')}")

    best_df = pd.DataFrame(best_rows)
    all_df = pd.concat(all_candidates, ignore_index=True) if all_candidates else pd.DataFrame()

    best_df.to_csv(BEST_OUTPUT, sep="\t", index=False)
    all_df.to_csv(ALL_CANDIDATES_OUTPUT, sep="\t", index=False)

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    print("\n" + "=" * 100)
    print("DONE")
    print(f"Global best-candidate summary:")
    print(f"  {BEST_OUTPUT}")
    print(f"Global all-candidate summary:")
    print(f"  {ALL_CANDIDATES_OUTPUT}")
    print(f"Compatibility phenotype folders:")
    print(f"  {COMPAT_OUTDIR}/<Phenotype>/")
    print("\nFor each phenotype, use:")
    print("  GWASCatalogDownloaded/<Phenotype>/GWAS_Candidate_Index.csv")
    print("  GWASCatalogDownloaded/<Phenotype>/GWAS_Best_Associations.csv")
    print("  GWASCatalogDownloaded/<Phenotype>/Candidates/*.csv")


if __name__ == "__main__":
    main()
