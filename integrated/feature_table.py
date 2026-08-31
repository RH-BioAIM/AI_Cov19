"""
Builds the candidate clinical-variable and missingness table
(Supplementary Table S1).
"""
import os
import re
import urllib.request
import pandas as pd

from build_integrated_model import ALLDATA_PATH, load_shared_cohort

REVISION_DIR = os.path.dirname(os.path.abspath(__file__))
DICT_URL = "https://www.cancerimagingarchive.net/wp-content/uploads/deidentified_overlap_tcia.csv.cleaned.csv.template_20210806.csv"
DICT_CACHE = os.path.join(REVISION_DIR, "tcia_dictionary_cache.csv")

HIGH_MISSINGNESS_FLAG = 40.0  # percent, threshold to flag on a SELECTED feature


def get_dictionary():
    if not os.path.exists(DICT_CACHE):
        urllib.request.urlretrieve(DICT_URL, DICT_CACHE)
        print(f"Downloaded TCIA dictionary -> {DICT_CACHE}")
    return pd.read_csv(DICT_CACHE)


# ---------------------------------------------------------------------------
# Non-predictor (Step 2 Section A) and dropped (Step 2 Section B, +
# visit_start_datetime) -- names as they appear in the TCIA dictionary.
# ---------------------------------------------------------------------------
NON_PREDICTOR_DICT_NAMES = {"to_patient_id", "length_of_stay", "last.status"}
DROPPED_HOSPITAL_COURSE_DICT_NAMES = {
    "is_icu", "was_ventilated", "invasive_vent_days",
    "Acute.Hepatic.Injury..during.hospitalization.", "Acute.Kidney.Injury..during.hospitalization.",
    "kidney_replacement_therapy", "therapeutic.exnox.Boolean", "therapeutic.heparin.Boolean",
    "Other.anticoagulation.therapy",
}
DROPPED_ADMIN_DICT_NAMES = {"visit_concept_name", "visit_start_datetime"}

TRIAGE_CHART_ABSTRACTED = {
    "htn_v", "dm_v", "cad_v", "hf_ef_v", "ckd_v", "malignancies_v", "copd_v", "other_lung_disease_v",
    "acei_v", "arb_v", "antibiotics_use_v", "nsaid_use_v", "days_prior_sx", "smoking_status_v",
    "cough_v", "dyspnea_admission_v", "nausea_v", "vomiting_v", "diarrhea_v", "abdominal_pain_v", "fever_v",
}
COMORBIDITY_NAMES = {"htn_v", "dm_v", "cad_v", "hf_ef_v", "ckd_v", "malignancies_v", "copd_v",
                      "other_lung_disease_v", "kidney_transplant"}
HOME_MED_HISTORY_NAMES = {"acei_v", "arb_v", "nsaid_use_v", "antibiotics_use_v", "days_prior_sx", "smoking_status_v"}
SYMPTOM_NAMES = {"cough_v", "dyspnea_admission_v", "nausea_v", "vomiting_v", "diarrhea_v",
                  "abdominal_pain_v", "fever_v"}
DEMOGRAPHIC_ADMIN_NAMES = {"covid19_statuses", "age.splits", "gender_concept_name"}

VITAL_KEYWORDS = ["temperature", "pulseOx", "Respiration", "HeartRate", "SBP", "MAP", "BMI",
                  "Oral_temperature", "Oxygen saturation", "Respiratory rate", "Heart rate",
                  "Systolic blood pressure", "Mean blood pressure", "Body mass index"]


def categorize(dict_name):
    if dict_name in NON_PREDICTOR_DICT_NAMES:
        return "Outcome / ID"
    if dict_name in DROPPED_ADMIN_DICT_NAMES:
        return "Demographics / admin"
    if dict_name in DROPPED_HOSPITAL_COURSE_DICT_NAMES:
        return "Hospital course / treatment (dropped)"
    if dict_name in DEMOGRAPHIC_ADMIN_NAMES:
        return "Demographics / admin"
    if dict_name in COMORBIDITY_NAMES:
        return "Comorbidities"
    if dict_name in HOME_MED_HISTORY_NAMES:
        return "Home medications / history"
    if dict_name in SYMPTOM_NAMES:
        return "Presenting symptoms"
    if any(kw in dict_name for kw in VITAL_KEYWORDS):
        return "Admission vitals"
    return "Admission labs"  # everything else in the 118 triage-time set is a lab


def role(dict_name):
    if dict_name in NON_PREDICTOR_DICT_NAMES:
        return "Non-predictor (ID/target/outcome)"
    if dict_name in DROPPED_ADMIN_DICT_NAMES or dict_name in DROPPED_HOSPITAL_COURSE_DICT_NAMES:
        return "Dropped - leakage/consequence"
    return "Selected (restricted feature set)"


# ---------------------------------------------------------------------------
# Human-readable names
# ---------------------------------------------------------------------------
LOINC_READABLE = {
    "8331-1_Oral temperature": "Oral temperature",
    "59408-5_Oxygen saturation in Arterial blood by Pulse oximetry": "Oxygen saturation (pulse ox)",
    "9279-1_Respiratory rate": "Respiratory rate",
    "76282-3_Heart rate.beat-to-beat by EKG": "Heart rate",
    "8480-6_Systolic blood pressure": "Systolic blood pressure",
    "76536-2_Mean blood pressure by Noninvasive": "Mean arterial pressure",
    "33256-9_Leukocytes [#/volume] corrected for nucleated erythrocytes in Blood by Automated count": "White blood cell count",
    "751-8_Neutrophils [#/volume] in Blood by Automated count": "Neutrophil count",
    "731-0_Lymphocytes [#/volume] in Blood by Automated count": "Lymphocyte count",
    "2951-2_Sodium [Moles/volume] in Serum or Plasma": "Sodium, serum",
    "1920-8_Aspartate aminotransferase [Enzymatic activity/volume] in Serum or Plasma": "AST, serum",
    "1744-2_Alanine aminotransferase [Enzymatic activity/volume] in Serum or Plasma by No addition of P-5'-P": "ALT, serum",
    "2157-6_Creatine kinase [Enzymatic activity/volume] in Serum or Plasma": "Creatine kinase, serum",
    "2524-7_Lactate [Moles/volume] in Serum or Plasma": "Lactate, serum",
    "6598-7_Troponin T.cardiac [Mass/volume] in Serum or Plasma": "Troponin T, serum",
    "33762-6_Natriuretic peptide.B prohormone N-Terminal [Mass/volume] in Serum or Plasma": "NT-proBNP",
    "75241-0_Procalcitonin [Mass/volume] in Serum or Plasma by Immunoassay": "Procalcitonin, serum",
    "48058-2_Fibrin D-dimer DDU [Mass/volume] in Platelet poor plasma by Immunoassay": "D-dimer",
    "2276-4_Ferritin [Mass/volume] in Serum or Plasma": "Ferritin, serum",
    "1988-5_C reactive protein [Mass/volume] in Serum or Plasma": "C-reactive protein, serum",
    "4548-4_Hemoglobin A1c/Hemoglobin.total in Blood": "Hemoglobin A1c",
    "39156-5_Body mass index (BMI) [Ratio]": "Body mass index (BMI)",
    "2951-2_Sodium [Moles/volume] in Serum or Plasma.1": "Sodium, serum (duplicate field in source)",
    "2823-3_Potassium [Moles/volume] in Serum or Plasma": "Potassium, serum",
    "2075-0_Chloride [Moles/volume] in Serum or Plasma": "Chloride, serum",
    "1963-8_Bicarbonate [Moles/volume] in Serum or Plasma": "Bicarbonate, serum",
    "3094-0_Urea nitrogen [Mass/volume] in Serum or Plasma": "BUN (blood urea nitrogen), serum",
    "2160-0_Creatinine [Mass/volume] in Serum or Plasma": "Creatinine, serum",
    "62238-1_Glomerular filtration rate/1.73 sq M.predicted [Volume Rate/Area] in Serum, Plasma or Blood by Creatinine-based formula (CKD-EPI)": "eGFR (CKD-EPI)",
    "33254-4_pH of Arterial blood adjusted to patient's actual temperature": "Arterial blood pH",
    "30341-2_Erythrocyte sedimentation rate": "ESR (erythrocyte sedimentation rate)",
    "2345-7_Glucose [Mass/volume] in Serum or Plasma": "Glucose, serum",
    "13457-7_Cholesterol in LDL [Mass/volume] in Serum or Plasma by calculation": "LDL cholesterol",
    "13458-5_Cholesterol in VLDL [Mass/volume] in Serum or Plasma by calculation": "VLDL cholesterol",
    "2571-8_Triglyceride [Mass/volume] in Serum or Plasma": "Triglycerides, serum",
    "2085-9_Cholesterol in HDL [Mass/volume] in Serum or Plasma": "HDL cholesterol",
}

SPECIAL_READABLE = {
    "to_patient_id": "Patient ID",
    "length_of_stay": "Length of stay (target)",
    "last.status": "In-hospital mortality (deceased/discharged)",
    "covid19_statuses": "COVID-19 status",
    "age.splits": "Age (binned)",
    "gender_concept_name": "Gender",
    "visit_start_datetime": "Visit start date/time",
    "visit_concept_name": "Encounter type (ED/inpatient/observation)",
    "is_icu": "ICU admission (any point during stay)",
    "was_ventilated": "Invasive ventilation (any point during stay)",
    "invasive_vent_days": "Days of invasive ventilation",
    "Acute.Hepatic.Injury..during.hospitalization.": "Acute hepatic injury (during hospitalization)",
    "Acute.Kidney.Injury..during.hospitalization.": "Acute kidney injury (during hospitalization)",
    "Urine.protein": "Urine protein",
    "kidney_replacement_therapy": "Renal replacement therapy (during stay)",
    "kidney_transplant": "History of kidney transplant",
    "therapeutic.exnox.Boolean": "Therapeutic-dose enoxaparin administered",
    "therapeutic.heparin.Boolean": "Therapeutic-dose heparin administered",
    "Other.anticoagulation.therapy": "Other anticoagulation therapy administered",
}


def humanize_binned_flag(name):
    """e.g. 'Sodium.above145' -> 'Sodium > 145'; 'eGFR.between30and60' -> 'eGFR 30-60'."""
    s = name
    m = re.match(r"^(.*)\.between(-?\d+\.?\d*)and(-?\d+\.?\d*)$", s)
    if m:
        return f"{m.group(1).replace('_', ' ')} {m.group(2)}-{m.group(3)}"
    m = re.match(r"^(.*)\.above(-?\d+\.?\d*)$", s)
    if m:
        return f"{m.group(1).replace('_', ' ')} > {m.group(2)}"
    m = re.match(r"^(.*)\.below(-?\d+\.?\d*)$", s)
    if m:
        return f"{m.group(1).replace('_', ' ')} < {m.group(2)}"
    m = re.match(r"^(.*)\.over(-?\d+\.?\d*)$", s)
    if m:
        return f"{m.group(1).replace('_', ' ')} > {m.group(2)}"
    m = re.match(r"^(.*)\.under(-?\d+\.?\d*)$", s)
    if m:
        return f"{m.group(1).replace('_', ' ')} < {m.group(2)}"
    m = re.match(r"^(.*)\.(\d+)\.?(\d*)to(\d+)\.?(\d*)$", s)  # e.g. A1C.6.6to7.9
    if m:
        base = m.group(1)
        lo = f"{m.group(2)}.{m.group(3)}" if m.group(3) else m.group(2)
        hi = f"{m.group(4)}.{m.group(5)}" if m.group(5) else m.group(4)
        return f"{base} {lo}-{hi}"
    return s.replace(".", " ").replace("_", " ")


def readable_name(dict_name):
    if dict_name in SPECIAL_READABLE:
        return SPECIAL_READABLE[dict_name]
    if dict_name in LOINC_READABLE:
        return LOINC_READABLE[dict_name]
    if "_v" == dict_name[-2:]:
        return dict_name[:-2].replace("_", " ").capitalize()
    if re.match(r"^[\dA-Za-z]+-\d.*_", dict_name):
        return dict_name  # unmapped LOINC-coded name, shouldn't happen but fail loud, not silent
    return humanize_binned_flag(dict_name)


# genuine one-off inconsistencies in the source cleaning (R janitor-style
# unit-phrase substitution: "[Moles/volume]" etc. became "per volume" for some
# columns and "density" for others, with no space in a few -- verified directly
# against AllData.csv's actual column list, not guessed) plus one real typo
# ("an0.5" vs "and0.5" in the procalcitonin bin column).
MANUAL_NAME_OVERRIDES = {
    "procalcitonin.between0.25and0.5": "procalcitonin_between0_25an0_5",
    "33256-9_Leukocytes [#/volume] corrected for nucleated erythrocytes in Blood by Automated count":
        "Leukocytesper volume corrected for nucleated erythrocytes in Blood by Automated count",
    "751-8_Neutrophils [#/volume] in Blood by Automated count": "Neutrophils per volume in Blood by Automated count",
    "731-0_Lymphocytes [#/volume] in Blood by Automated count": "Lymphocytes per volume in Blood by Automated count",
    "2951-2_Sodium [Moles/volume] in Serum or Plasma": "Sodium per volume in Serum or Plasma",
    "2951-2_Sodium [Moles/volume] in Serum or Plasma.1": "Sodium density in Serum or Plasma",
    "1920-8_Aspartate aminotransferase [Enzymatic activity/volume] in Serum or Plasma":
        "Aspartate aminotransferase activity over volume  in Serum or Plasma",
    "1744-2_Alanine aminotransferase [Enzymatic activity/volume] in Serum or Plasma by No addition of P-5'-P":
        "Alanine aminotransferase activity over volume in Serum or Plasma by No addition of P5primeP",
    "2157-6_Creatine kinase [Enzymatic activity/volume] in Serum or Plasma":
        "Creatine kinase activity per volume in Serum or Plasma",
    "2524-7_Lactate [Moles/volume] in Serum or Plasma": "Lactate moles per volume in Serum or Plasma",
    "6598-7_Troponin T.cardiac [Mass/volume] in Serum or Plasma": "Troponin T cardiac density in Serum or Plasma",
    "33762-6_Natriuretic peptide.B prohormone N-Terminal [Mass/volume] in Serum or Plasma":
        "Natriuretic peptide B prohormone N Terminal density in Serum or Plasma",
    "75241-0_Procalcitonin [Mass/volume] in Serum or Plasma by Immunoassay":
        "Procalcitonin density in Serum or Plasma by Immunoassay",
    "48058-2_Fibrin D-dimer DDU [Mass/volume] in Platelet poor plasma by Immunoassay":
        "Fibrin D-dimer DDU density in Platelet poor plasma by Immunoassay",
    "2276-4_Ferritin [Mass/volume] in Serum or Plasma": "Ferritin density in Serum or Plasma",
    "1988-5_C reactive protein [Mass/volume] in Serum or Plasma": "C reactive protein density in Serum or Plasma",
    "39156-5_Body mass index (BMI) [Ratio]": "Body mass index BMI",
    "2823-3_Potassium [Moles/volume] in Serum or Plasma": "Potassium density in Serum or Plasma",
    "2075-0_Chloride [Moles/volume] in Serum or Plasma": "Chloride density in Serum or Plasma",
    "1963-8_Bicarbonate [Moles/volume] in Serum or Plasma": "Bicarbonate moles per volume in Serum or Plasma",
    "3094-0_Urea nitrogen [Mass/volume] in Serum or Plasma": "Urea nitrogen density in Serum or Plasma",
    "2160-0_Creatinine [Mass/volume] in Serum or Plasma": "Creatininedensity in Serum or Plasma",
    "62238-1_Glomerular filtration rate/1.73 sq M.predicted [Volume Rate/Area] in Serum, Plasma or Blood by Creatinine-based formula (CKD-EPI)":
        "Glomerular filtration rate over 1_73 sq M predicted rate per area in Serum Plasma or Blood by Creatinine-based formula",
    "2345-7_Glucose [Mass/volume] in Serum or Plasma": "Glucosedensity in Serum or Plasma",
    "13457-7_Cholesterol in LDL [Mass/volume] in Serum or Plasma by calculation":
        "Cholesterol in LDL density in Serum or Plasma by calculation",
    "13458-5_Cholesterol in VLDL [Mass/volume] in Serum or Plasma by calculation":
        "Cholesterol in VLDLdensity in Serum or Plasma by calculation",
    "2571-8_Triglyceride [Mass/volume] in Serum or Plasma": "Triglyceride density in Serum or Plasma",
    "2085-9_Cholesterol in HDL [Mass/volume] in Serum or Plasma": "Cholesterol in HDL density in Serum or Plasma",
}


def dict_name_to_alldata_name(dict_name, alldata_cols):
    if dict_name in MANUAL_NAME_OVERRIDES:
        return MANUAL_NAME_OVERRIDES[dict_name]
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    norm_map = {norm(c): c for c in alldata_cols}
    match = norm_map.get(norm(dict_name))
    if match:
        return match
    # LOINC-coded raw columns have their numeric code prefix stripped in
    # AllData.csv (e.g. "8331-1_Oral temperature" -> "Oral_temperature")
    stripped = re.sub(r"^[\dA-Za-z]+-\d+_", "", dict_name)
    return norm_map.get(norm(stripped))


def main():
    tcia_dict = get_dictionary()
    df = load_shared_cohort()  # the matched 1341-patient set, restricted-model's actual cohort
    alldata_cols = pd.read_csv(ALLDATA_PATH, nrows=0).columns.tolist()

    rows = []
    unmapped = []
    for _, r in tcia_dict.iterrows():
        dict_name = r["column_name"]
        alldata_name = dict_name_to_alldata_name(dict_name, alldata_cols)
        if alldata_name is None:
            unmapped.append(dict_name)
            continue

        missing_pct = 100.0 * df[alldata_name].isna().mean()
        rows.append({
            "variable_name": readable_name(dict_name),
            "raw_name": dict_name,
            "category": categorize(dict_name),
            "role": role(dict_name),
            "missingness_pct": round(missing_pct, 2),
            "is_chart_abstracted": bool(r["is_chart_abstracted"]) if pd.notna(r["is_chart_abstracted"]) else False,
        })

    assert not unmapped, f"columns present in TCIA dictionary but not found in AllData.csv: {unmapped}"
    assert len(rows) == 131, f"expected 131 rows, got {len(rows)}"

    out = pd.DataFrame(rows)
    role_order = {"Selected (restricted feature set)": 0, "Dropped - leakage/consequence": 1,
                  "Non-predictor (ID/target/outcome)": 2}
    out["_role_sort"] = out["role"].map(role_order)
    out = out.sort_values(["category", "_role_sort", "variable_name"]).drop(columns="_role_sort")

    path = os.path.join(REVISION_DIR, "feature_table.csv")
    out.to_csv(path, index=False)

    # ---- summary ----
    n_total = len(out)
    n_selected = (out["role"] == "Selected (restricted feature set)").sum()
    n_dropped = (out["role"] == "Dropped - leakage/consequence").sum()
    n_nonpred = (out["role"] == "Non-predictor (ID/target/outcome)").sum()

    print("=" * 70)
    print("STEP 7 SUMMARY")
    print("=" * 70)
    print(f"Total candidate variables: {n_total}")
    print(f"  Selected (restricted feature set): {n_selected}")
    print(f"  Dropped - leakage/consequence:      {n_dropped}")
    print(f"  Non-predictor (ID/target/outcome):  {n_nonpred}")
    print(f"Missingness denominator: n={len(df)} (the matched, modeled cohort -- same patients as the OOF predictions)")

    bins = [(-0.01, 5, "<5%"), (5, 20, "5-20%"), (20, 50, ">20-50%"), (50, 100.01, ">50%")]
    print("\nMissingness distribution (all 131 variables):")
    for lo, hi, label in bins:
        n = ((out["missingness_pct"] > lo) & (out["missingness_pct"] <= hi)).sum()
        print(f"  {label}: {n}")

    high_missing_selected = out[(out["role"] == "Selected (restricted feature set)") &
                                 (out["missingness_pct"] > HIGH_MISSINGNESS_FLAG)]
    print(f"\nSELECTED features with >{HIGH_MISSINGNESS_FLAG}% missingness "
          f"({len(high_missing_selected)} found) -- review before finalizing methods text:")
    if len(high_missing_selected):
        print(high_missing_selected[["variable_name", "raw_name", "missingness_pct"]].to_string(index=False))
    else:
        print("  (none)")

    print(f"\nSaved {path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
