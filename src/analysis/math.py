import numpy as np

def cut_transient(t, transient_frac=0.2):
    end = t[0] + transient_frac * (t[-1] - t[0])
    return t >= end

def rescale_domain(dataframes, max_h=0.5):

    # so we don't alter the input data
    restricted = {}

    for test_key, df_original in dataframes.items():

        # also so we don't alter the input data
        df = df_original.copy()
        # keep only rows where h is <= max_h
        # .loc keeps only the true rows in the mask
        # and we copy to make sure we don't mess with the original
        df = df.loc[df["h"] <= max_h].copy()

        # puts original h values in a special new column
        df["original_h"] = df["h"]
        # rescales working h from 0-0.5 to 0-1
        # prevents the inner-domain raw area under curve from being smaller just because its horizontal axis ends at 0.5
        df["h"] = df["h"] / max_h

        restricted[test_key] = df

    return restricted

# based on distance from baseline normalization
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

# this is basically min-max normalization
def calculate_range_normalized_scores(dataframes):
    '''
    Will be used in robustness.py to test an alternate feature normalization method and see if the main results hold.

    dataframes - the per-test feature dataframes pulled from the official analysis CSVs; has structure:
    {
    "FHN a": dataframe,
    "FHN tau": dataframe,
    "JR v0": dataframe,
    "JR q": dataframe,
    }   
    '''

    raw_scores = {}

    for test_key, df_original in dataframes.items():
        # sort just in case something got shuffled on the way here
        df = df_original.copy().sort_values('h')

        # for the v 1.0 analysis run, we don't need to include the homo stats here but it's safe for later
        std_min = min(
            df["homo_std"].min(),
            df["hetero_std"].min(),
        )
        std_max = max(
            df["homo_std"].max(),
            df["hetero_std"].max(),
        )
        std_range = std_max - std_min

        ptp_min = min(
            df["homo_peak_to_peak"].min(),
            df["hetero_peak_to_peak"].min(),
        )
        ptp_max = max(
            df["homo_peak_to_peak"].max(),
            df["hetero_peak_to_peak"].max(),
        )
        ptp_range = ptp_max - ptp_min

        # check for div zero errors
        if std_range <= 1e-12:
            raise ValueError(f'{test_key} has zero std. dev. range')
        if ptp_range <= 1e-12:
            raise ValueError(f'{test_key} has a zero peak to peak range')

        # results are pandas series w/ one value per retained h level
        # |X_h - X_0| / (X_max - X_min)
        std_dev = (
            df['hetero_std'] - df['homo_std']
            ).abs() / std_range

        ptp_dev = (
            df['hetero_peak_to_peak'] - df['homo_peak_to_peak']
        ).abs() / ptp_range

        combined_degradation = (std_dev + ptp_dev) / 2

        # calculates area under curve of combined deg as h increases
        # area stored under current test name
        raw_scores[test_key] = np.trapezoid(
            combined_degradation,
            df['h'],
        )

    # now we do max relative scaling for the final scores
    max_score = max(raw_scores.values())

    # calculates 10(S_raw/ (S_raw_max))
    # largest score becomes exactly 10
    relative_scores = {
        test_key: 10 * raw_score / max_score
        for test_key, raw_score in raw_scores.items()
    }

    return raw_scores, relative_scores

# combines the test scores per model using equal weighting of raw AUC values
def calculate_model_fragility_scores(raw_scores):
    '''
    Accepts raw AUC dictionary indexed by test key
    '''

    model_tests = {
        'FHN': ['FHN a', 'FHN tau'],
        'JR': ['JR q', 'JR v0'],
    }

    model_raw_scores = {
        'FHN': [],
        'JR': [],
    }

    for test_key, score in raw_scores.items():
        if test_key in model_tests['FHN']:
            model_raw_scores['FHN'].append(score)
        elif test_key in model_tests['JR']:
            model_raw_scores['JR'].append(score)

    # averages raw AUC for each model
    # remember that iterating directly over a dict returns keys
    for model in model_tests:
        model_raw_scores[model] = np.mean(model_raw_scores[model])

    # max normalize them
    max_score = max(model_raw_scores.values())

    # calculates 10(S_raw/ (S_raw_max))
    # largest score becomes exactly 10
    relative_scores = {
        model_key: 10 * raw_score / max_score
        for model_key, raw_score in model_raw_scores.items()
    }

    return model_raw_scores, relative_scores

