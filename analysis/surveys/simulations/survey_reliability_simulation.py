"""Simulates survey reliability using random and non-random distributions.

This module calculates agreement measures, computes p-values, and classifies
the p-values into categories. It provides the following functions:

Functions:
- random_distribution: Generates random survey results for simulation.
- non_random_distribution: Generates survey results with a mix of agreement and randomness.
- more_than_x_agreement: Counts items with agreement levels meeting or exceeding a threshold.
- full_agreement_measure: Counts cases of full agreement (default threshold: 5).
- super_majority_agreement_measure: Counts cases of super-majority agreement (default threshold: 4).
- compute_p_value: Compares actual survey results with random simulations to compute p-values.
- classify_p_values: Categorizes p-values into 'low', 'medium', or 'high' categories.
"""

import secrets
from collections.abc import Callable


def random_distribution(
    num_items: int = 100, num_surveys: int = 5, num_simulations: int = 1000
) -> list[list[list[int]]]:
    """Generates random survey results for each item.

    Args:
        num_items: Number of items to be assessed.
        num_surveys: Number of surveys per item (each survey will be 0 or 1).
        num_simulations: Number of Monte Carlo simulations.

    Returns:
        A list of lists where each sublist contains the random survey results for each item.

        Note that the list structure is: random_results[simulation][item][surveyors response].
    """
    random_results = []
    for _ in range(num_simulations):
        simulation = [
            [secrets.choice([0, 1]) for _ in range(num_surveys)]
            for _ in range(num_items)
        ]
        random_results.append(simulation)

    return random_results


def non_random_distribution(
    num_items: int = 100, num_surveys: int = 5, quality_measure: float = 0.8
) -> list[list[int]]:
    """Generates non-random survey results with high agreement (4/5 or 5/5), but also includes some randomness.

    Args:
        num_items: Number of items to be assessed.
        num_surveys: Number of surveyors.
        quality_measure: Probability that the surveyors agree with the truth (0.8 means 80% agreement).

    Returns:
        A list of lists, where each sublist contains the results of `num_surveys` for an item.
    """
    non_random_results = []

    for _ in range(num_items):
        # Generate the "truth" for the item (1 or 0)
        truth = secrets.choice([0, 1])

        item_results = []
        for _ in range(num_surveys):
            # Randomly decide if the surveyor will agree with the truth
            if secrets.randbelow(100) < quality_measure * 100:
                item_results.append(truth)
            else:
                item_results.append(secrets.choice([0, 1]))

        non_random_results.append(item_results)

    return non_random_results


def more_than_x_agreement(result_list: list[int], threshold: int) -> int:
    """Count the number of items in the result list where the count of '1's (indicating agreement) meets or exceeds the specified threshold.

    Args:
        result_list (list[int]): A list of integers representing agreement levels.
        threshold (int): The minimum number of '1's required to count as agreement.

    Returns:
        int: The total number of items that meet or exceed the agreement threshold.
    """
    agreements = 0
    for item_result in result_list:
        if item_result.count(1) >= threshold:
            agreements += 1
    return agreements


def full_agreement_measure(
    result_list: list[int], full_agreement_count: int = 5
) -> int:
    """Count cases of full agreement (default threshold: 5)."""
    return more_than_x_agreement(result_list, full_agreement_count)


def super_majority_agreement_measure(
    result_list: list[int], super_majority_count: int = 4
) -> int:
    """Count cases of super-majority agreement (default threshold: 4)."""
    return more_than_x_agreement(result_list, super_majority_count)


def compute_p_value(
    non_random_result: list[list[int]],
    random_results: list[list[list[int]]],
    agreement_measure_function: Callable[[list], int],
) -> list[float]:
    """Computes the p-value by comparing the actual survey results (non-random) with random simulations.

    Args:
        non_random_result: A list of non-random survey results (actual data).
        random_results: A list of lists containing random survey results (Monte Carlo simulation).
        agreement_measure_function: a function which takes in a list and then returns an int

    Returns:
        A p-value.
    """
    non_random_agreement = agreement_measure_function(non_random_result)

    agreements_more_than_non_random = 0
    for simul in random_results:
        if agreement_measure_function(simul) >= non_random_agreement:
            agreements_more_than_non_random += 1

    p_value = agreements_more_than_non_random / len(random_results)
    return p_value


def full_p_value_calculation(
    random_dist: list, non_random_dist: list, agreement_function: Callable
) -> float:
    """Note: this function looks repetitive and it very much is.

    But we want our logic to be importable outside this code (always!)

    So we 'll add this little bit of logic. This will also allow us to add
    **kwargs if we want to increase the ability to use this further
    """
    p_value = compute_p_value(non_random_dist, random_dist, agreement_function)

    return p_value


if __name__ == "__main__":
    random_distrib = random_distribution()
    non_random_distrib = non_random_distribution(quality_measure=0.4)

    print("---" * 5)
    for agreement_func in [full_agreement_measure, super_majority_agreement_measure]:
        print(
            f"\nCalculating p-value with agreement function: {agreement_func.__name__}"
        )

        p_value = full_p_value_calculation(
            random_distrib, non_random_distrib, agreement_func
        )

        print(f"Calculated p-value: {p_value}")

    print("---" * 5)
