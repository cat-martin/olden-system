import numpy as np
import matplotlib.pyplot as plt


def calculate_relative_fragility_scores(dataframes):
    raw_scores = {}

    for test_key, df_original in dataframes.items():
        df = df_original.copy().sort_values("h")

        std_deviation = (
            df["hetero_std"] - df["homo_std"]
        ).abs() / df["homo_std"].abs()

        ptp_deviation = (
            df["hetero_peak_to_peak"]
            - df["homo_peak_to_peak"]
        ).abs() / df["homo_peak_to_peak"].abs()

        combined_degradation = (
            std_deviation + ptp_deviation
        ) / 2

        # Area under degradation-versus-h curve
        raw_scores[test_key] = np.trapezoid(
            combined_degradation,
            df["h"]
        )

    maximum_score = max(raw_scores.values())

    relative_scores = {
        test_key: 10 * raw_score / maximum_score
        for test_key, raw_score in raw_scores.items()
    }

    return raw_scores, relative_scores