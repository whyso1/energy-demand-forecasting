"""Peak-event evaluation, ported from the legacy evaluate_peak_predictions().

Cleaned up: plotting split out from computation (see scripts/ for the
plotting call sites) so this is directly unit-testable, and y_test is
an explicit argument rather than a captured global. The underlying
"nearest predicted peak within N hours" logic is unchanged -- it's a
reasonable idea that's much more meaningful now that predictions come
from a chronological test set instead of a randomly shuffled one, since
"time error in hours to the nearest predicted peak" only means
something when the test index is a real, contiguous timeline.
"""
import numpy as np
import pandas as pd


def evaluate_peak_predictions(
    y_test: pd.Series, model_preds: dict[str, np.ndarray], top_n: int = 10, hit_window_hours: float = 6
) -> tuple[pd.DataFrame, pd.DataFrame]:
    actual_peaks = y_test.nlargest(top_n)

    all_results = []
    summary_stats = []

    for model_name, preds in model_preds.items():
        pred_series = pd.Series(preds, index=y_test.index)
        pred_peaks = pred_series.nlargest(top_n)

        hit_count = 0
        model_records = []

        for t_actual, v_actual in actual_peaks.items():
            time_diffs = pd.Series(
                np.abs((pred_peaks.index - t_actual) / np.timedelta64(1, "h")),
                index=pred_peaks.index,
            )
            closest_time = time_diffs.idxmin()
            closest_pred_value = pred_series.loc[closest_time]
            time_error = time_diffs.loc[closest_time]
            magnitude_error = closest_pred_value - v_actual

            if time_error <= hit_window_hours:
                hit_count += 1

            record = {
                "Model": model_name,
                "Actual Peak Time": t_actual,
                "Actual Peak Value": v_actual,
                "Pred Peak Time": closest_time,
                "Pred Peak Value": closest_pred_value,
                "Time Error (hours)": time_error,
                "Magnitude Error": magnitude_error,
            }
            all_results.append(record)
            model_records.append(record)

        model_df = pd.DataFrame(model_records)
        summary_stats.append(
            {
                "Model": model_name,
                "Hit Rate (%)": 100 * hit_count / top_n,
                "Mean Time Error (hrs)": model_df["Time Error (hours)"].mean(),
                "Median Time Error (hrs)": model_df["Time Error (hours)"].median(),
                "Mean Magnitude Error": model_df["Magnitude Error"].mean(),
                "RMSE Magnitude Error": np.sqrt((model_df["Magnitude Error"] ** 2).mean()),
            }
        )

    results_df = pd.DataFrame(all_results)
    summary_df = pd.DataFrame(summary_stats)
    return results_df, summary_df
