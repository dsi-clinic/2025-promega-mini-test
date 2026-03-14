import json
from collections import defaultdict
import pandas as pd


def main():
    try:
        with open("organoid_classification_results_aggregated.json") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: 'organoid_classification_results_aggregated.json' not found.")
        return
    except json.JSONDecodeError:
        print(
            "Error: Could not decode JSON from 'organoid_classification_results_aggregated.json'."
        )
        return

    classification_vs_quality_counts = defaultdict(lambda: defaultdict(int))

    for organoid, evaluations in data.items():
        if not evaluations:
            continue

        classifications = [eval_item["evaluation"] for eval_item in evaluations]
        qualities = [eval_item["quality"] for eval_item in evaluations]

        for classification in set(classifications):
            for quality in set(qualities):
                count = sum(
                    1
                    for eval_item in evaluations
                    if eval_item["evaluation"] == classification
                    and eval_item["quality"] == quality
                )
                if count > 0:
                    classification_vs_quality_counts[classification][quality] += count

    print("\n=== Confusion Matrix (Counts): Classification vs Quality ===")
    if classification_vs_quality_counts:
        df_class_qual_counts = pd.DataFrame.from_dict(
            classification_vs_quality_counts, orient="index"
        )
        all_qualities = sorted(
            list(
                set(
                    [
                        q
                        for counts in classification_vs_quality_counts.values()
                        for q in counts.keys()
                    ]
                )
            )
        )
        df_class_qual_counts = df_class_qual_counts.reindex(
            columns=all_qualities, fill_value=0
        )
        print(df_class_qual_counts)
    else:
        print(
            "No data to generate Confusion Matrix (Counts): Classification vs Quality"
        )

    analysis_results = {}
    stats = {
        "total_organoids": 0,
        "agreement_stats": {
            "complete_agreement": 0,
            "strong_agreement": 0,
            "other_cases": 0,
        },
        "classification_distribution": defaultdict(int),
        "quality_distribution": defaultdict(int),
        "evaluations_per_organoid": defaultdict(int),
        "employee_dissent_stats": defaultdict(
            lambda: {
                "total_evaluations": 0,
                "lone_dissenter_count": 0,
                "one_of_two_dissenters_count": 0,
            }
        ),
    }

    for organoid, evaluations in data.items():
        stats["total_organoids"] += 1
        evaluation_count = len(evaluations)
        stats["evaluations_per_organoid"][evaluation_count] += 1

        class_counts = defaultdict(lambda: {"count": 0, "employees": []})
        quality_counts = defaultdict(lambda: {"count": 0, "employees": []})

        current_organoid_evaluation_details = []

        for eval_item in evaluations:
            employee_last_name = eval_item["employee"].split()[-1]
            stats["employee_dissent_stats"][employee_last_name][
                "total_evaluations"
            ] += 1
            current_organoid_evaluation_details.append(
                {
                    "employee_lastname": employee_last_name,
                    "evaluation_type": eval_item["evaluation"],
                }
            )
            class_counts[eval_item["evaluation"]]["count"] += 1
            class_counts[eval_item["evaluation"]]["employees"].append(
                employee_last_name
            )
            quality_counts[eval_item["quality"]]["count"] += 1
            quality_counts[eval_item["quality"]]["employees"].append(employee_last_name)

        agreement_ratio = 0.0
        current_agreement_level = "other"

        if evaluation_count > 0:
            if class_counts:
                max_class_vote = (
                    max(v["count"] for v in class_counts.values())
                    if class_counts
                    else 0
                )
                agreement_ratio = (
                    max_class_vote / evaluation_count if evaluation_count > 0 else 0
                )

                if len(class_counts) == 1:
                    stats["agreement_stats"]["complete_agreement"] += 1
                    current_agreement_level = "complete"
                elif agreement_ratio > 0.75:
                    stats["agreement_stats"]["strong_agreement"] += 1
                    current_agreement_level = "strong"
                else:
                    stats["agreement_stats"]["other_cases"] += 1
            else:
                stats["agreement_stats"]["other_cases"] += 1
        else:
            stats["agreement_stats"]["other_cases"] += 1

        if evaluation_count > 1 and class_counts:
            max_vote_for_a_class = 0
            if any(v["count"] > 0 for v in class_counts.values()):
                max_vote_for_a_class = max(
                    v_data["count"] for v_data in class_counts.values()
                )

            majority_candidate_classes = [
                cls
                for cls, cls_data in class_counts.items()
                if cls_data["count"] == max_vote_for_a_class
            ]

            if len(majority_candidate_classes) == 1 and max_vote_for_a_class > 0:
                majority_classification = majority_candidate_classes[0]

                dissenting_employee_lastnames = []
                for eval_detail in current_organoid_evaluation_details:
                    if eval_detail["evaluation_type"] != majority_classification:
                        dissenting_employee_lastnames.append(
                            eval_detail["employee_lastname"]
                        )

                num_dissenters = len(dissenting_employee_lastnames)

                if num_dissenters == 1:
                    lone_dissenter_emp_name = dissenting_employee_lastnames[0]
                    stats["employee_dissent_stats"][lone_dissenter_emp_name][
                        "lone_dissenter_count"
                    ] += 1
                elif num_dissenters == 2:
                    for dissenter_emp_name in dissenting_employee_lastnames:
                        stats["employee_dissent_stats"][dissenter_emp_name][
                            "one_of_two_dissenters_count"
                        ] += 1

        for cls, data_item in class_counts.items():
            stats["classification_distribution"][cls] += data_item["count"]
        for qual, data_item in quality_counts.items():
            stats["quality_distribution"][qual] += data_item["count"]

        analysis_results[organoid] = {
            "classifications": {k: v for k, v in class_counts.items()},
            "quality": {k: v for k, v in quality_counts.items()},
            "agreement_level": current_agreement_level,
        }

    with open("organoid_analysis_results.json", "w") as f:
        json.dump({"analysis": analysis_results, "statistics": stats}, f, indent=2)

    print("\n=== Agreement Statistics ===")
    print(f"Total organoids analyzed: {stats['total_organoids']}")
    print(
        f"- Complete agreement (100% same): {stats['agreement_stats']['complete_agreement']} organoids"
    )
    print(
        f"- Strong agreement (>75% same): {stats['agreement_stats']['strong_agreement']} organoids"
    )
    print(f"- Other cases: {stats['agreement_stats']['other_cases']} organoids")

    print("\n=== Evaluation Distribution ===")
    print("\nClassifications:")
    for cls, count_val in sorted(stats["classification_distribution"].items()):
        print(f"- {cls}: {count_val} evaluations")
    print("\nQuality Ratings:")
    for qual, count_val in sorted(stats["quality_distribution"].items()):
        print(f"- {qual}: {count_val} evaluations")

    print("\n=== Employee Dissent Statistics ===")
    if stats["employee_dissent_stats"]:
        for emp, emp_data in sorted(stats["employee_dissent_stats"].items()):
            total_evals = emp_data["total_evaluations"]
            lone_count = emp_data["lone_dissenter_count"]
            two_dissent_count = emp_data["one_of_two_dissenters_count"]

            lone_prop = (lone_count / total_evals) * 100 if total_evals > 0 else 0
            two_dissent_prop = (
                (two_dissent_count / total_evals) * 100 if total_evals > 0 else 0
            )

            if total_evals > 0:
                print(f"- Employee: {emp} (Total Evaluations: {total_evals})")
                print(f"  - Was Lone Dissenter: {lone_count} times ({lone_prop:.1f}%)")
                print(
                    f"  - Was One of Two Dissenters: {two_dissent_count} times ({two_dissent_prop:.1f}%)"
                )
    else:
        print("No employee dissent data to display.")

    print("\nAnalysis saved to 'organoid_analysis_results.json'")

    if not classification_vs_quality_counts:
        return

    from scipy.stats import chi2_contingency

    observed = df_class_qual_counts.values
    chi2, p, dof, expected = chi2_contingency(observed)

    print("\n=== Chi-Square Test of Independence ===")
    print(f"Chi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4f}")
    print(f"Degrees of Freedom: {dof}")
    print("\nExpected Frequencies:")
    print(
        pd.DataFrame(
            expected,
            index=df_class_qual_counts.index,
            columns=df_class_qual_counts.columns,
        )
    )

    alpha = 0.05
    if p < alpha:
        print(
            "\nReject the null hypothesis: There is a statistically significant association between Classification and Quality."
        )
    else:
        print(
            "\nFail to reject the null hypothesis: There is no statistically significant association between Classification and Quality."
        )


if __name__ == "__main__":
    main()
