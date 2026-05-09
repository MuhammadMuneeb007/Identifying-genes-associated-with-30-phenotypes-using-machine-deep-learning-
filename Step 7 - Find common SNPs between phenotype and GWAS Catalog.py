#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
import json
import time
import argparse
import xml.etree.ElementTree as ET
from typing import Dict, List, Set, Tuple, Optional

import requests
import pandas as pd


# ============================================================
# STEP 6.1 — COMMON VARIANTS + FINAL SELECTED CITATIONS
# ============================================================
# Purpose:
#   For each phenotype:
#     1. Read all candidate GWAS association files from Step 6.0.
#     2. Compare each candidate GWAS file with the phenotype dataset SNPs.
#     3. Select ONE final GWAS association set per phenotype using:
#          a) closest trait match score,
#          b) highest number of common SNPs,
#          c) highest GWAS SNP count,
#          d) highest association rows.
#     4. Generate BibTeX citations ONLY for the selected final candidate.
#     5. Save one merged BibTeX file across all selected phenotypes.
#     6. Save manuscript-ready CSV, Markdown, and LaTeX tables.
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
GWAS_ROOT = "GWASCatalogDownloaded"
DATA_ROOT = "."

MASTER_SUMMARY_FILE = os.path.join(GWAS_ROOT, "GWAS_Common_Variants_Master_Summary.csv")
MASTER_BIB_FILE = os.path.join(GWAS_ROOT, "GWAS_Common_Variants_Master_Citations.bib")
MASTER_MD_FILE = os.path.join(GWAS_ROOT, "GWAS_Common_Variants_Table_For_Manuscript.md")
MASTER_TEX_FILE = os.path.join(GWAS_ROOT, "GWAS_Common_Variants_Table_For_Manuscript.tex")
MASTER_JSON_FILE = os.path.join(GWAS_ROOT, "GWAS_Common_Variants_Master_Summary.json")


# ============================================================
# BASIC HELPERS
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


def read_csv_safely(path: str) -> pd.DataFrame:
    attempts = [
        {"encoding": "utf-8"},
        {"encoding": "utf-8", "encoding_errors": "replace"},
        {"encoding": "latin1"},
    ]

    last_error = None
    for kwargs in attempts:
        try:
            return pd.read_csv(path, dtype=str, low_memory=False, **kwargs)
        except Exception as e:
            last_error = e

    raise RuntimeError(f"Could not read {path}. Last error: {last_error}")


def normalize_rsid(value: str) -> str:
    if value is None:
        return ""

    value = str(value).strip()
    if not value or value.lower() in {"nan", "none", "null", "na", "-"}:
        return ""

    value = value.split("-")[0].strip()

    m = re.search(r"\brs\d+\b", value, flags=re.IGNORECASE)
    if m:
        return m.group(0).lower()

    if re.fullmatch(r"\d+", value):
        return f"rs{value}".lower()

    return value.lower()


def split_possible_snps(value: str) -> List[str]:
    if value is None:
        return []

    value = str(value).strip()
    if not value or value.lower() in {"nan", "none", "null", "na", "-"}:
        return []

    parts = re.split(r"[,;/\s]+", value)
    out = []

    for part in parts:
        snp = normalize_rsid(part)
        if snp:
            out.append(snp)

    return sorted(set(out))


def latex_escape(text: str) -> str:
    if text is None or pd.isna(text):
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
# DATASET SNP LOADING
# ============================================================
def find_dataset_snp_file(phenotype: str, data_root: str) -> Optional[str]:
    phenotype_dir = os.path.join(data_root, phenotype)

    patterns = []

    if os.path.isdir(phenotype_dir):
        patterns.extend([
            os.path.join(phenotype_dir, "**", "*.bim"),
            os.path.join(phenotype_dir, "**", "*.snplist"),
            os.path.join(phenotype_dir, "**", "*snp*.txt"),
            os.path.join(phenotype_dir, "**", "*SNP*.txt"),
        ])

    patterns.extend([
        os.path.join(data_root, "**", f"*{phenotype}*.bim"),
        os.path.join(data_root, "**", f"*{phenotype}*.snplist"),
        os.path.join(data_root, "**", f"*{phenotype}*snp*.txt"),
        os.path.join(data_root, "**", f"*{phenotype}*SNP*.txt"),
    ])

    candidates = []
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            if not os.path.isfile(path):
                continue
            if GWAS_ROOT in path:
                continue
            size = os.path.getsize(path)
            candidates.append((size, path))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]


def load_dataset_snps(path: str) -> Set[str]:
    snps = set()

    if path.endswith(".bim"):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    snp = normalize_rsid(parts[1])
                    if snp:
                        snps.add(snp)
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                first = line.strip().split()[0]
                snp = normalize_rsid(first)
                if snp:
                    snps.add(snp)

    return snps


# ============================================================
# GWAS SNP / GENE EXTRACTION
# ============================================================
def extract_gwas_snps_from_row(row: pd.Series) -> Set[str]:
    snps = set()

    if "SNPS" in row.index:
        for snp in split_possible_snps(row.get("SNPS", "")):
            snps.add(snp)

    if "SNP_ID_CURRENT" in row.index:
        snp = normalize_rsid(row.get("SNP_ID_CURRENT", ""))
        if snp:
            snps.add(snp)

    if "STRONGEST SNP-RISK ALLELE" in row.index:
        snp = normalize_rsid(row.get("STRONGEST SNP-RISK ALLELE", ""))
        if snp:
            snps.add(snp)

    return snps


def extract_all_gwas_snps(df: pd.DataFrame) -> Set[str]:
    all_snps = set()
    for _, row in df.iterrows():
        all_snps.update(extract_gwas_snps_from_row(row))
    return all_snps


def extract_gene_set(df: pd.DataFrame) -> List[str]:
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
                g = part.strip()
                if g and g.lower() not in {"na", "nan", "none", "null", "-", "nr"}:
                    genes.add(g)

    return sorted(genes)


def add_common_variant_annotations(
    gwas_df: pd.DataFrame,
    dataset_snps: Set[str],
) -> Tuple[pd.DataFrame, Set[str], pd.DataFrame]:
    records = []
    common_snps = set()
    row_common_values = []
    row_gwas_values = []

    for idx, row in gwas_df.iterrows():
        row_snps = extract_gwas_snps_from_row(row)
        row_common = sorted(row_snps.intersection(dataset_snps))

        row_gwas_values.append(";".join(sorted(row_snps)))
        row_common_values.append(";".join(row_common))
        common_snps.update(row_common)

        for snp in row_common:
            rec = row.to_dict()
            rec["COMMON_SNP"] = snp
            rec["GWAS_ROW_INDEX"] = idx
            records.append(rec)

    annotated = gwas_df.copy()
    annotated["GWAS_SNPS_EXTRACTED"] = row_gwas_values
    annotated["COMMON_SNPS_WITH_DATASET"] = row_common_values
    annotated["HAS_COMMON_SNP_WITH_DATASET"] = annotated["COMMON_SNPS_WITH_DATASET"].apply(
        lambda x: "YES" if str(x).strip() else "NO"
    )

    common_rows = pd.DataFrame(records)

    return annotated, common_snps, common_rows


# ============================================================
# CANDIDATE FILES
# ============================================================
def list_candidate_files(phenotype: str) -> List[str]:
    pheno_dir = os.path.join(GWAS_ROOT, phenotype)
    candidate_dir = os.path.join(pheno_dir, "Candidates")

    files = []

    if os.path.isdir(candidate_dir):
        files.extend(sorted(glob.glob(os.path.join(candidate_dir, "*.csv"))))

    best_file = os.path.join(pheno_dir, "GWAS_Best_Associations.csv")
    if os.path.exists(best_file):
        files.append(best_file)

    files = [
        f for f in files
        if not f.lower().endswith("_genes.csv")
        and "gene" not in os.path.basename(f).lower()
    ]

    seen = set()
    unique = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique.append(f)

    return unique


def get_candidate_id_and_trait(df: pd.DataFrame, file_path: str) -> Tuple[str, str]:
    candidate_id = ""
    candidate_trait = ""

    if "CANDIDATE_TRAIT_ID" in df.columns:
        vals = df["CANDIDATE_TRAIT_ID"].dropna().astype(str).str.strip()
        vals = vals[vals != ""]
        if len(vals) > 0:
            candidate_id = vals.iloc[0]

    if "MAPPED_TRAIT" in df.columns:
        vals = df["MAPPED_TRAIT"].dropna().astype(str).str.strip()
        vals = vals[vals != ""]
        if len(vals) > 0:
            candidate_trait = vals.iloc[0]

    if not candidate_id:
        m = re.search(r"(EFO_\d+|MONDO_\d+|HP_\d+)", os.path.basename(file_path))
        if m:
            candidate_id = m.group(1)

    return candidate_id, candidate_trait


def trait_match_score(candidate_trait: str, phenotype: str) -> int:
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


# ============================================================
# PUBMED -> BIBTEX
# ============================================================
def clean_bibtex_value(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def make_cite_key(pmid: str, first_author: str = "", year: str = "") -> str:
    pmid = str(pmid).strip()
    if pmid and pmid.lower() not in {"nan", "none", "null", ""}:
        return f"PMID{pmid}"

    author = re.sub(r"[^A-Za-z0-9]+", "", first_author.split()[0]) if first_author else "GWAS"
    return f"{author}{year}" if year else author


def extract_pubmed_ids(df: pd.DataFrame) -> List[str]:
    if df.empty or "PUBMEDID" not in df.columns:
        return []

    pmids = (
        df["PUBMEDID"]
        .dropna()
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )

    pmids = [p for p in pmids if re.fullmatch(r"\d+", p)]
    return sorted(set(pmids))


def fetch_pubmed_metadata(pmids: List[str], batch_size: int = 100, sleep_sec: float = 0.4) -> Dict[str, dict]:
    metadata = {}

    if not pmids:
        return metadata

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
        }

        try:
            r = requests.get(base_url, params=params, timeout=60)
            r.raise_for_status()

            root = ET.fromstring(r.text)

            for article in root.findall(".//PubmedArticle"):
                pmid_node = article.find(".//PMID")
                if pmid_node is None or not pmid_node.text:
                    continue

                pmid = pmid_node.text.strip()

                title_node = article.find(".//ArticleTitle")
                journal_node = article.find(".//Journal/Title")
                year_node = article.find(".//PubDate/Year")
                volume_node = article.find(".//JournalIssue/Volume")
                issue_node = article.find(".//JournalIssue/Issue")
                pages_node = article.find(".//Pagination/MedlinePgn")

                title = "".join(title_node.itertext()).strip() if title_node is not None else ""
                journal = "".join(journal_node.itertext()).strip() if journal_node is not None else ""
                year = year_node.text.strip() if year_node is not None and year_node.text else ""
                volume = volume_node.text.strip() if volume_node is not None and volume_node.text else ""
                issue = issue_node.text.strip() if issue_node is not None and issue_node.text else ""
                pages = pages_node.text.strip() if pages_node is not None and pages_node.text else ""

                authors = []
                for author in article.findall(".//AuthorList/Author"):
                    last = author.findtext("LastName", default="").strip()
                    initials = author.findtext("Initials", default="").strip()
                    collective = author.findtext("CollectiveName", default="").strip()

                    if last:
                        authors.append(f"{last}, {initials}".strip().strip(","))
                    elif collective:
                        authors.append(collective)

                doi = ""
                for article_id in article.findall(".//ArticleIdList/ArticleId"):
                    if article_id.attrib.get("IdType", "").lower() == "doi":
                        doi = article_id.text.strip() if article_id.text else ""

                metadata[pmid] = {
                    "pmid": pmid,
                    "title": title,
                    "journal": journal,
                    "year": year,
                    "volume": volume,
                    "issue": issue,
                    "pages": pages,
                    "authors": authors,
                    "doi": doi,
                }

            time.sleep(sleep_sec)

        except Exception as e:
            print(f"[WARNING] PubMed fetch failed for batch {batch[:3]}...: {e}")

    return metadata


def fallback_metadata_from_gwas_rows(df: pd.DataFrame) -> Dict[str, dict]:
    out = {}

    if df.empty or "PUBMEDID" not in df.columns:
        return out

    for _, row in df.iterrows():
        pmid = str(row.get("PUBMEDID", "")).replace(".0", "").strip()

        if not re.fullmatch(r"\d+", pmid):
            continue

        if pmid in out:
            continue

        first_author = str(row.get("FIRST AUTHOR", "")).strip()
        title = str(row.get("STUDY", "")).strip()
        journal = str(row.get("JOURNAL", "")).strip()
        date = str(row.get("DATE", "")).strip()
        year = date[:4] if re.match(r"\d{4}", date) else ""

        out[pmid] = {
            "pmid": pmid,
            "title": title,
            "journal": journal,
            "year": year,
            "volume": "",
            "issue": "",
            "pages": "",
            "authors": [first_author] if first_author else [],
            "doi": "",
        }

    return out


def metadata_to_bibtex(meta: dict) -> str:
    pmid = meta.get("pmid", "")
    authors = meta.get("authors", [])
    first_author = authors[0] if authors else ""
    year = meta.get("year", "")

    cite_key = make_cite_key(pmid, first_author, year)
    author_text = " and ".join(authors) if authors else "Unknown"

    fields = {
        "title": clean_bibtex_value(meta.get("title", "")),
        "author": clean_bibtex_value(author_text),
        "journal": clean_bibtex_value(meta.get("journal", "")),
        "year": clean_bibtex_value(year),
        "volume": clean_bibtex_value(meta.get("volume", "")),
        "number": clean_bibtex_value(meta.get("issue", "")),
        "pages": clean_bibtex_value(meta.get("pages", "")),
        "doi": clean_bibtex_value(meta.get("doi", "")),
        "pmid": clean_bibtex_value(pmid),
    }

    lines = [f"@article{{{cite_key},"]
    for k, v in fields.items():
        if v:
            lines.append(f"  {k} = {{{v}}},")
    lines.append("}")

    return "\n".join(lines)


def write_bibtex_for_selected_rows(selected_common_rows: pd.DataFrame, bib_file: str) -> Tuple[List[str], List[str]]:
    """
    Creates BibTeX only for the final selected candidate rows.
    It does not create BibTeX for every candidate.
    """
    ensure_dir(os.path.dirname(bib_file))

    pmids = extract_pubmed_ids(selected_common_rows)

    if not pmids:
        with open(bib_file, "w", encoding="utf-8") as f:
            f.write("% No PMID found for selected common-variant rows.\n")
        return [], []

    pubmed_meta = fetch_pubmed_metadata(pmids)
    fallback_meta = fallback_metadata_from_gwas_rows(selected_common_rows)

    for pmid, fallback in fallback_meta.items():
        if pmid not in pubmed_meta:
            pubmed_meta[pmid] = fallback

    bib_entries = []
    cite_keys = []

    for pmid in pmids:
        meta = pubmed_meta.get(pmid)
        if not meta:
            continue

        bib_entries.append(metadata_to_bibtex(meta))
        cite_keys.append(make_cite_key(
            pmid,
            meta.get("authors", [""])[0] if meta.get("authors") else "",
            meta.get("year", "")
        ))

    with open(bib_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(bib_entries))
        f.write("\n")

    return pmids, cite_keys


# ============================================================
# PHENOTYPE PROCESSING
# ============================================================
def process_phenotype(phenotype: str, data_root: str, skip_citations: bool = False) -> dict:
    print("\n" + "=" * 100)
    print(f"Processing phenotype: {phenotype}")
    print(f"Display name        : {display_name(phenotype)}")

    pheno_dir = os.path.join(GWAS_ROOT, phenotype)
    common_dir = os.path.join(pheno_dir, "CommonVariants")
    citation_dir = os.path.join(pheno_dir, "Citations")

    ensure_dir(common_dir)
    ensure_dir(citation_dir)

    dataset_file = find_dataset_snp_file(phenotype, data_root)

    if not dataset_file:
        print(f"[WARNING] No dataset SNP file found for phenotype: {phenotype}")
        return {
            "Phenotype": phenotype,
            "Phenotype Display": display_name(phenotype),
            "Dataset SNP File": "",
            "SNPs in our data": 0,
            "Selected GWAS Catalog ID": "",
            "Selected GWAS Trait": "",
            "SNPs in GWAS Catalogue": 0,
            "Association rows": 0,
            "Common SNPs": 0,
            "Unique genes": 0,
            "PMIDs": "",
            "Cite keys": "",
            "LaTeX cite": "",
            "BibTeX file": "",
            "Status": "NO_DATASET_SNP_FILE",
        }

    dataset_snps = load_dataset_snps(dataset_file)
    print(f"Dataset SNP file    : {dataset_file}")
    print(f"SNPs in our data    : {len(dataset_snps):,}")

    candidate_files = list_candidate_files(phenotype)

    if not candidate_files:
        print(f"[WARNING] No GWAS candidate files found for phenotype: {phenotype}")
        return {
            "Phenotype": phenotype,
            "Phenotype Display": display_name(phenotype),
            "Dataset SNP File": dataset_file,
            "SNPs in our data": len(dataset_snps),
            "Selected GWAS Catalog ID": "",
            "Selected GWAS Trait": "",
            "SNPs in GWAS Catalogue": 0,
            "Association rows": 0,
            "Common SNPs": 0,
            "Unique genes": 0,
            "PMIDs": "",
            "Cite keys": "",
            "LaTeX cite": "",
            "BibTeX file": "",
            "Status": "NO_GWAS_FILES",
        }

    all_candidate_summaries = []
    selected_summary = None
    selected_common_rows = pd.DataFrame()
    selected_annotated = pd.DataFrame()

    for gwas_file in candidate_files:
        try:
            gwas_df = read_csv_safely(gwas_file)
        except Exception as e:
            print(f"[WARNING] Failed to read GWAS file {gwas_file}: {e}")
            continue

        candidate_id, candidate_trait = get_candidate_id_and_trait(gwas_df, gwas_file)
        match_score = trait_match_score(candidate_trait, phenotype)

        gwas_snps = extract_all_gwas_snps(gwas_df)
        annotated, common_snps, common_rows = add_common_variant_annotations(gwas_df, dataset_snps)
        genes = extract_gene_set(common_rows) if not common_rows.empty else []

        candidate_label = candidate_id if candidate_id else safe_filename(os.path.basename(gwas_file))

        annotated_file = os.path.join(
            common_dir,
            f"{safe_filename(phenotype)}__{safe_filename(candidate_label)}__annotated_gwas.csv"
        )
        common_file = os.path.join(
            common_dir,
            f"{safe_filename(phenotype)}__{safe_filename(candidate_label)}__common_variants.csv"
        )

        annotated.to_csv(annotated_file, index=False)
        common_rows.to_csv(common_file, index=False)

        summary = {
            "Phenotype": phenotype,
            "Phenotype Display": display_name(phenotype),
            "GWAS File": gwas_file,
            "Candidate ID": candidate_id,
            "Candidate Trait": candidate_trait,
            "Trait Match Score": match_score,
            "SNPs in GWAS Catalogue": len(gwas_snps),
            "Association rows": len(gwas_df),
            "SNPs in our data": len(dataset_snps),
            "Common SNPs": len(common_snps),
            "Common SNP List": ";".join(sorted(common_snps)),
            "Unique genes from common rows": len(genes),
            "Common variants file": common_file,
            "Annotated GWAS file": annotated_file,
        }

        all_candidate_summaries.append(summary)

        # Selection rule:
        #   1. closest trait match,
        #   2. highest common SNPs,
        #   3. highest GWAS SNPs,
        #   4. highest association rows.
        current_score = (
            summary["Trait Match Score"],
            summary["Common SNPs"],
            summary["SNPs in GWAS Catalogue"],
            summary["Association rows"],
        )

        if selected_summary is None:
            selected_summary = summary
            selected_common_rows = common_rows
            selected_annotated = annotated
        else:
            best_score = (
                selected_summary["Trait Match Score"],
                selected_summary["Common SNPs"],
                selected_summary["SNPs in GWAS Catalogue"],
                selected_summary["Association rows"],
            )

            if current_score > best_score:
                selected_summary = summary
                selected_common_rows = common_rows
                selected_annotated = annotated

        print(
            f"  Candidate {candidate_id or 'NO_ID'} | "
            f"Trait={candidate_trait[:70]} | "
            f"Match={match_score} | "
            f"GWAS SNPs={len(gwas_snps):,} | "
            f"Common={len(common_snps):,}"
        )

    if not all_candidate_summaries or selected_summary is None:
        return {
            "Phenotype": phenotype,
            "Phenotype Display": display_name(phenotype),
            "Dataset SNP File": dataset_file,
            "SNPs in our data": len(dataset_snps),
            "Selected GWAS Catalog ID": "",
            "Selected GWAS Trait": "",
            "SNPs in GWAS Catalogue": 0,
            "Association rows": 0,
            "Common SNPs": 0,
            "Unique genes": 0,
            "PMIDs": "",
            "Cite keys": "",
            "LaTeX cite": "",
            "BibTeX file": "",
            "Status": "NO_VALID_GWAS_FILES",
        }

    # Save all candidate comparison results
    all_candidates_file = os.path.join(pheno_dir, "GWAS_Common_Variants_AllCandidates.csv")
    pd.DataFrame(all_candidate_summaries).sort_values(
        by=["Trait Match Score", "Common SNPs", "SNPs in GWAS Catalogue", "Association rows"],
        ascending=False,
    ).to_csv(all_candidates_file, index=False)

    # Save selected final common rows
    selected_common_file = os.path.join(pheno_dir, "GWAS_Selected_Common_Variants.csv")
    selected_annotated_file = os.path.join(pheno_dir, "GWAS_Selected_Annotated_Associations.csv")
    selected_common_rows.to_csv(selected_common_file, index=False)
    selected_annotated.to_csv(selected_annotated_file, index=False)

    # Citation only for selected final candidate
    bib_file = os.path.join(
        citation_dir,
        f"{safe_filename(phenotype)}_selected_gwas_citations.bib"
    )

    if skip_citations:
        pmids, cite_keys = [], []
        with open(bib_file, "w", encoding="utf-8") as f:
            f.write("% Citation retrieval skipped.\n")
    else:
        pmids, cite_keys = write_bibtex_for_selected_rows(selected_common_rows, bib_file)

    latex_cite = r"\cite{" + ",".join(cite_keys) + "}" if cite_keys else ""

    selected_genes = extract_gene_set(selected_common_rows) if not selected_common_rows.empty else []

    # Update GWAS_Summary.csv for downstream table
    summary_row = {
        "Phenotype": phenotype,
        "Phenotype Display": display_name(phenotype),
        "GWAS Catalog ID": selected_summary["Candidate ID"],
        "GWAS Catalog Trait": selected_summary["Candidate Trait"],
        "Trait Match Score": selected_summary["Trait Match Score"],
        "SNPs in GWAS Catalogue": selected_summary["SNPs in GWAS Catalogue"],
        "SNPs in our data": selected_summary["SNPs in our data"],
        "Common SNPs": selected_summary["Common SNPs"],
        "Unique genes": len(selected_genes),
        "Association rows": selected_summary["Association rows"],
        "Dataset SNP File": dataset_file,
        "Selected GWAS File": selected_summary["GWAS File"],
        "All candidates comparison file": all_candidates_file,
        "Selected common variants file": selected_common_file,
        "Selected annotated associations file": selected_annotated_file,
        "BibTeX file": bib_file,
        "PMIDs": ";".join(pmids),
        "Cite keys": ";".join(cite_keys),
        "LaTeX cite": latex_cite,
        "Status": "OK",
    }

    summary_csv = os.path.join(pheno_dir, "GWAS_Summary.csv")
    summary_json = os.path.join(pheno_dir, "GWAS_Summary.json")

    pd.DataFrame([summary_row]).to_csv(summary_csv, index=False)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_row, f, indent=2)

    print("\nSelected final GWAS association set:")
    print(f"  ID            : {selected_summary['Candidate ID']}")
    print(f"  Trait         : {selected_summary['Candidate Trait']}")
    print(f"  Match score   : {selected_summary['Trait Match Score']}")
    print(f"  GWAS SNPs     : {selected_summary['SNPs in GWAS Catalogue']:,}")
    print(f"  Dataset SNPs  : {selected_summary['SNPs in our data']:,}")
    print(f"  Common SNPs   : {selected_summary['Common SNPs']:,}")
    print(f"  BibTeX        : {bib_file}")
    print(f"  LaTeX cite    : {latex_cite}")

    return summary_row


# ============================================================
# MASTER MANUSCRIPT TABLES
# ============================================================
def write_master_bib(master_rows: List[dict]) -> None:
    entries = []
    seen_keys = set()

    for row in master_rows:
        bib_file = row.get("BibTeX file", "")
        if not bib_file or not os.path.exists(bib_file):
            continue

        with open(bib_file, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text or text.startswith("%"):
            continue

        # Split by BibTeX entries and deduplicate by cite key
        chunks = re.split(r"\n(?=@article\{)", text)
        for chunk in chunks:
            m = re.search(r"@article\{([^,]+),", chunk)
            if not m:
                continue
            key = m.group(1)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            entries.append(chunk.strip())

    with open(MASTER_BIB_FILE, "w", encoding="utf-8") as f:
        f.write("\n\n".join(entries))
        f.write("\n")


def write_markdown_table(df: pd.DataFrame) -> None:
    cols = [
        "Phenotype Display",
        "GWAS Catalog ID",
        "GWAS Catalog Trait",
        "SNPs in GWAS Catalogue",
        "SNPs in our data",
        "Common SNPs",
        "Unique genes",
        "LaTeX cite",
    ]

    available = [c for c in cols if c in df.columns]
    md = df[available].to_markdown(index=False)

    with open(MASTER_MD_FILE, "w", encoding="utf-8") as f:
        f.write(md)
        f.write("\n")


def write_latex_table(df: pd.DataFrame) -> None:
    lines = []
    lines.append(r"\begin{table*}[!ht]")
    lines.append(r"\centering")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{|l|l|l|r|r|r|r|l|}")
    lines.append(r"\hline")
    lines.append(
        r"\textbf{Phenotype} & \textbf{GWAS ID} & \textbf{GWAS trait} & "
        r"\textbf{GWAS SNPs} & \textbf{Dataset SNPs} & \textbf{Common SNPs} & "
        r"\textbf{Genes} & \textbf{Citation} \\ \hline"
    )

    for _, row in df.iterrows():
        phenotype = latex_escape(row.get("Phenotype Display", ""))
        gwas_id = latex_escape(row.get("GWAS Catalog ID", ""))
        trait = latex_escape(row.get("GWAS Catalog Trait", ""))
        gwas_snps = int(float(row.get("SNPs in GWAS Catalogue", 0) or 0))
        data_snps = int(float(row.get("SNPs in our data", 0) or 0))
        common = int(float(row.get("Common SNPs", 0) or 0))
        genes = int(float(row.get("Unique genes", 0) or 0))
        cite = row.get("LaTeX cite", "")

        lines.append(
            f"{phenotype} & {gwas_id} & {trait} & "
            f"{gwas_snps} & {data_snps} & {common} & {genes} & {cite} \\\\ \\hline"
        )

    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(
        r"\caption{\textbf{GWAS Catalog association sets selected for phenotype-level variant overlap analysis.} "
        r"For each phenotype, candidate GWAS Catalog association sets were compared with the available genotype dataset. "
        r"The selected GWAS association set was chosen using phenotype-trait match score followed by the number of common SNPs, "
        r"GWAS SNP count, and association-row count.}"
    )
    lines.append(r"\label{tab:gwas_selected_association_sets}")
    lines.append(r"\end{table*}")

    with open(MASTER_TEX_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


def print_article_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print("MANUSCRIPT-READY SUMMARY")
    print("=" * 100)

    total_ok = int((df["Status"] == "OK").sum()) if "Status" in df.columns else len(df)
    total_common = int(pd.to_numeric(df["Common SNPs"], errors="coerce").fillna(0).sum()) if "Common SNPs" in df.columns else 0

    print(
        f"\nWe re-queried the GWAS Catalog association data for the analysed phenotypes and "
        f"selected one final GWAS association set per phenotype using a two-stage criterion: "
        f"phenotype/trait match specificity followed by the number of SNPs shared with the genotype dataset. "
        f"Across {total_ok} successfully processed phenotypes, the selected GWAS association sets contained "
        f"a total of {total_common} SNP overlaps with the corresponding genotype datasets. "
        f"For each selected association set, PubMed identifiers from the overlapping GWAS rows were converted "
        f"to BibTeX entries and merged into a single bibliography file."
    )

    print("\nSelected GWAS association sets:")
    show_cols = [
        "Phenotype Display",
        "GWAS Catalog ID",
        "GWAS Catalog Trait",
        "SNPs in GWAS Catalogue",
        "SNPs in our data",
        "Common SNPs",
        "Unique genes",
        "LaTeX cite",
        "Status",
    ]

    available = [c for c in show_cols if c in df.columns]
    print(df[available].to_string(index=False))

    print("\nFiles written:")
    print(f"  Master summary CSV : {MASTER_SUMMARY_FILE}")
    print(f"  Master BibTeX      : {MASTER_BIB_FILE}")
    print(f"  Markdown table     : {MASTER_MD_FILE}")
    print(f"  LaTeX table        : {MASTER_TEX_FILE}")
    print("=" * 100)


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Step 6.1: Select final GWAS association set per phenotype, compute common SNPs, and generate selected BibTeX citations."
    )

    parser.add_argument(
        "--data-root",
        default=DATA_ROOT,
        help="Root folder containing phenotype genotype folders/files. Default: current directory.",
    )

    parser.add_argument(
        "--phenotype",
        default=None,
        help="Run one phenotype only, e.g. Depression.",
    )

    parser.add_argument(
        "--skip-citations",
        action="store_true",
        help="Skip PubMed citation retrieval and create empty BibTeX placeholders.",
    )

    args = parser.parse_args()

    ensure_dir(GWAS_ROOT)

    phenotypes = [args.phenotype] if args.phenotype else PHENOTYPES

    master_rows = []

    for phenotype in phenotypes:
        row = process_phenotype(
            phenotype=phenotype,
            data_root=args.data_root,
            skip_citations=args.skip_citations,
        )
        master_rows.append(row)

    master_df = pd.DataFrame(master_rows)
    master_df.to_csv(MASTER_SUMMARY_FILE, index=False)

    with open(MASTER_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(master_rows, f, indent=2)

    write_master_bib(master_rows)
    write_markdown_table(master_df)
    write_latex_table(master_df)
    print_article_summary(master_df)


if __name__ == "__main__":
    main()
