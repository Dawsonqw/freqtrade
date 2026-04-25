from numpy import format_float_positional
from pandas import DataFrame, Series


def get_tick_size_over_time(candles: DataFrame) -> Series:
    """
    Calculate the number of significant digits for candles over time.
    It's using the Monthly maximum of the number of significant digits for each month.
    :param candles: DataFrame with OHLCV data
    :return: Series with the average number of significant digits for each month
    """
    # Downsample to daily before counting digits — tick_size is monthly,
    # so daily resolution is more than enough and avoids per-row apply on
    # 15m/5m data (96x–288x fewer rows).
    if "date" in candles.columns:
        indexed = candles.set_index("date", drop=False)
    else:
        indexed = candles
    daily = indexed[["open", "high", "low", "close"]].resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()

    # count the number of significant digits for the open and close prices
    for col in ["open", "high", "low", "close"]:
        daily[f"{col}_count"] = (
            daily[col]
            .apply(format_float_positional, precision=14, unique=False, fractional=False, trim="-")
            .str.extract(r"\.(\d*[1-9])")[0]
            .str.len()
        )
    daily["max_count"] = daily[["open_count", "close_count", "high_count", "low_count"]].max(
        axis=1
    )

    # Group by month and calculate the max number of significant digits
    monthly_count_avg1 = daily["max_count"].resample("MS").max()
    # convert monthly_open_count_avg from 5.0 to 0.00001, 4.0 to 0.0001, ...
    monthly_open_count_avg = 1 / 10**monthly_count_avg1

    return monthly_open_count_avg
