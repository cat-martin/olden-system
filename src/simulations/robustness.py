import pandas as pd

from src.analysis.math import (
    calculate_relative_fragility_scores,
    rescale_domain,
    calculate_range_normalized_scores,
)


def main():

    feature_dataframes = {
        "FHN a": pd.read_csv("src/data/fhn_a_features.csv"),
        "FHN tau": pd.read_csv("src/data/fhn_tau_features.csv"),
        "JR v0": pd.read_csv("src/data/jr_v0_features.csv"),
        "JR q": pd.read_csv("src/data/jr_q_features.csv"),
    }

    # parameter scaling domain check, 0-0.5 scaled to 0-1
    full_raw, full_relative = calculate_relative_fragility_scores(feature_dataframes)

    rescaled_dfs = rescale_domain(
        feature_dataframes,
        max_h=0.5,
    )

    rescaled_raw, rescaled_relative = calculate_relative_fragility_scores(rescaled_dfs)

    # recalculate degradation using range normalized feature deviation

    range_raw, range_relative = calculate_range_normalized_scores(feature_dataframes)

    # need to preserve raw integrated scores and normalized 0-10 scores for comparison

    # set of conditions we'll use to produce the export dataframe
    conditions = [
        (
            "baseline_relative",
            "full",
            full_raw,
            full_relative,
        ),
        (
            "baseline_relative",
            "inner_half",
            rescaled_raw,
            rescaled_relative,
        ),
        (
            "range_normalized",
            "full",
            range_raw,
            range_relative,
        ),
    ]

    scores = []

    # i want to figure out where each test falls based on the conditions so i should make a ranking column
    for method, domain, raw_scores, relative_scores in conditions:

        ordered_tests = sorted(raw_scores, key=raw_scores.get, reverse=True)

        # assigns rank to each of the 4 ordered tests
        ranks = {test_key: rank for rank, test_key in enumerate(ordered_tests, start=1)}

        # this means 4 rows per condition
        for test_key, raw_score in raw_scores.items():
            scores.append(
                {
                    "method": method,
                    "domain": domain,
                    "test_key": test_key,
                    "raw_AUC": raw_score,
                    "relative_fragility": relative_scores[test_key],
                    "rank": ranks[test_key],
                }
            )

    # export comparison table

    scores_df = pd.DataFrame(scores)

    scores_df.to_csv('src/data/robustness_scores.csv', index=False)

    # make the robustness results legible

    condition_labels = {
        ("baseline_relative", "full"):
            "Distance-from-reference normalization — "
            "full parameter domain",

        ("baseline_relative", "inner_half"):
            "Distance-from-reference normalization — "
            "inner half of parameter domain "
            "(original h <= 0.5; integration axis rescaled to 0–1)",

        ("range_normalized", "full"):
            "Range normalization — full parameter domain",
    }

    print('\n****** Robustness Analysis ******')

    for (method, domain), label, in condition_labels.items():
        # find the rows of the df that match the condition
        table = scores_df.loc[
            (scores_df['method'] == method) &
            (scores_df['domain'] == domain)
        ].copy()

        table = table.sort_values('rank')

        # get rid of the rows we don't care about for display
        table = table[[
            'rank',
            'test_key',
            'raw_AUC',
            'relative_fragility',
        ]]

        # give them nice names
        table = table.rename(
            columns={
                'rank': 'Rank',
                'test_key': 'Test',
                'raw_AUC': 'Raw Area Under Curve',
                'relative_fragility': 'Relative Fragility',
            }
        )

        # gotta round the values
        table['Raw Area Under Curve'] = table["Raw Area Under Curve"].round(2)
        table['Relative Fragility'] = table["Relative Fragility"].round(2)

        # to console
        print(f'{label}')
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
