import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer

RANDOM_STATE = 42


# ---------- 1. Load JSON ----------
def load_json_data(path):
    with open(path, "r") as f:
        return json.load(f)


# ---------- 2. BaseID extraction ----------
def extract_baseid(org_id):
    org_id = org_id.upper().strip()
    org_id = re.sub(r"\bDY\d+\b", "", org_id)
    org_id = re.sub(r"\s+", " ", org_id)
    return org_id.strip()


# ---------- 3. Extract features ----------
def extract_features(data):
    records = []
    for org_id, content in data.items():
        BA = content.get("BA", "")
        if not any(b in BA.upper() for b in ["BA1", "BA2"]):
            continue
        day = content.get("dayID", "").strip()
        metabolites = content.get("metabolites", {})

        def safe_val(name):
            return metabolites.get(name, {}).get("concentration_uM", np.nan)

        records.append({
            "OrganoidID": org_id,
            "BA": BA.upper(),
            "Day": day,
            "GlucoseGlo": safe_val("GlucoseGlo"),
            "GlutamateGlo": safe_val("GlutamateGlo"),
            "LactateGlo": safe_val("LactateGlo"),
            "PyruvateGlo": safe_val("PyruvateGlo"),
            "BaseID": extract_baseid(org_id),
        })
    return pd.DataFrame(records)


# ---------- 4. Extract only Dy30 survey labels ----------
def extract_day30_labels(data):
    labels = []
    for org_id, content in data.items():
        if content.get("dayID", "").upper() != "DY30":
            continue
        survey = content.get("survey", {})
        evaluations = survey.get("evaluations", [])
        if not evaluations:
            continue
        num_accept = sum(e["evaluation"] == "Acceptable" for e in evaluations)
        if num_accept >= 4:
            label = "Acceptable"
        elif num_accept <= 1:
            label = "Not Acceptable"
        else:
            continue
        labels.append({"BaseID": extract_baseid(org_id), "Label_Dy30": label})
    return pd.DataFrame(labels).drop_duplicates("BaseID")


# ---------- 5. Train models for each earlier day ----------
def train_models_by_day(features_df, label_df):
    results = []
    all_days = sorted(features_df["Day"].unique(), key=lambda x: int(re.findall(r"\d+", x)[0]))

    print("\n--- DEBUG: Organoid counts per day ---")
    for d in all_days:
        print(f"{d}: {features_df[features_df['Day'] == d]['BaseID'].nunique()} unique organoids")
    print("--------------------------------------\n")

    for day in all_days:
        
        day_df = features_df[features_df["Day"].str.upper() == day.upper()].copy()
        merged = pd.merge(day_df, label_df, on="BaseID", how="inner")

        if merged.empty:
            print(f"⚠️ Skipping {day}: no matching Dy30 labels.")
            continue

        overlap = merged["BaseID"].nunique()
        print(f"✅ {day}: {overlap} organoids with Dy30 labels.")

        X = merged[["GlucoseGlo", "GlutamateGlo", "LactateGlo", "PyruvateGlo"]]
        y = merged["Label_Dy30"]

        # --- Handle missing values safely ---
        if X.isna().all().all():
            print(f"⚠️ {day}: all features missing, skipping.")
            continue

        imputer = SimpleImputer(strategy="mean")
        X_imputed = imputer.fit_transform(X)
        X_imputed = np.nan_to_num(X_imputed, nan=0.0)

        if np.isnan(X_imputed).any():
            print(f"⚠️ {day}: still contains NaN after imputation, skipping.")
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
        )

        models = {
            "LogisticRegression": LogisticRegression(solver="liblinear", random_state=RANDOM_STATE),
            "LogisticRegression_SMOTE": LogisticRegression(class_weight="balanced", solver="liblinear", random_state=RANDOM_STATE),
            "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
            "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE),
        }

        smote = SMOTE(random_state=RANDOM_STATE)

        for name, model in models.items():
            if name == "LogisticRegression_SMOTE":
                try:
                    class_counts = pd.Series(y_train).value_counts()
                    if class_counts.min() < 6:
                        print(f"⚠️ {day} - {name}: Not enough samples for SMOTE (min {class_counts.min()}).")
                        model.fit(X_train, y_train)
                    else:
                        X_res, y_res = smote.fit_resample(X_train, y_train)
                        model.fit(X_res, y_res)
                except Exception as e:
                    print(f"⚠️ {day} - {name}: SMOTE failed ({e}), using plain LR.")
                    model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            results.append({
                "Day": day,
                "Model": name,
                "Accuracy": acc,
                "F1_Acceptable": report["Acceptable"]["f1-score"],
                "F1_NotAcceptable": report["Not Acceptable"]["f1-score"],
            })
            print(f"{day} - {name}: Accuracy={acc:.3f}")

    if not results:
        print("\n⚠️ No day had overlap with Dy30 labels.")
    else:
        days_used = sorted({r["Day"] for r in results}, key=lambda x: int(re.findall(r"\d+", x)[0]))
        print("\n✅ Trained models for days:", ", ".join(days_used))

    return pd.DataFrame(results)


# ---------- 6. Plot per-day accuracy ----------
def plot_results(results, output_folder):
    if results.empty:
        print("\n⚠️ No results to plot.")
        return

    plt.figure(figsize=(9, 6))
    for model in results["Model"].unique():
        subset = results[results["Model"] == model]
        plt.plot(subset["Day"], subset["Accuracy"], marker="o", label=model)
    plt.xlabel("Day")
    plt.ylabel("Accuracy")
    plt.title("Predicting Dy30 Acceptability from Earlier-Day Features")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "accuracy_over_days.png"))
    plt.close()


# ---------- 7. Main ----------
def main():
    parser = argparse.ArgumentParser(description="Predict Dy30 Outcome Using Earlier-Day Features")
    parser.add_argument("--data", required=True, help="Path to merged JSON dataset")
    args = parser.parse_args()

    output_folder = os.path.join(os.getcwd(), "output_temporal_results")
    os.makedirs(output_folder, exist_ok=True)

    data = load_json_data(args.data)
    features_df = extract_features(data)
    label_df = extract_day30_labels(data)

    print(f"Loaded {len(features_df)} feature records from {features_df['Day'].nunique()} days.")
    print(f"Dy30 labels: {len(label_df)} organoids with outcomes.")

    results = train_models_by_day(features_df, label_df)
    if not results.empty:
        csv_path = os.path.join(output_folder, "metrics_over_days.csv")
        results.to_csv(csv_path, index=False)
        plot_results(results, output_folder)
        print(f"\nAll temporal results saved in: {output_folder}")
        print(f"Metrics CSV: {csv_path}")
    else:
        print("\n⚠️ No results generated. Check if Dy30 labels exist and overlap with earlier-day BaseIDs.")


if __name__ == "__main__":
    main()
