import pandas as pd

from entise.constants import Objects as O


def apply_nightly_schedule(df_occ_schedule, start_hour, end_hour) -> pd.DataFrame:
    """
    Adjusts occupancy predictions so that once a certain evening condition is met,
        occupancy stays ON until 9 AM the next day.
    """
    df_result = df_occ_schedule.copy()

    # Create time masks once
    time_mask = df_result.between_time(start_hour, end_hour)
    evening_data = df_result.loc[time_mask.index]

    # Group by date more efficiently
    evening_data_grouped = evening_data.groupby(evening_data.index.date)

    for _, day_df in evening_data_grouped:
        if len(day_df) < 4:  # Skip if less than 4 data points
            continue

        occ_series = day_df[O.OCCUPANCY]

        # Vectorized rolling window check
        mask = occ_series.rolling(window=4, min_periods=4).sum() == 4

        if mask.any():
            # Define start and end times
            start_time = mask[mask].index[0]
            end_time = start_time.normalize() + pd.Timedelta(days=1, hours=9)

            # Apply rule - use loc for efficient assignment
            mask = (df_result.index >= start_time) & (df_result.index <= end_time)
            df_result.loc[mask, O.OCCUPANCY] = 1

    return df_result
