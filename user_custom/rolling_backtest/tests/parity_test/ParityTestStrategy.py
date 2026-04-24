"""
High-frequency parity test strategy.

Uses fast EMA crossover + RSI for frequent entries/exits.
Deterministic — no randomness, no external data, no lookahead.
"""
from freqtrade.strategy import IStrategy, IntParameter
from pandas import DataFrame
import talib.abstract as ta


class ParityTestStrategy(IStrategy):
    """
    Fast EMA cross + RSI strategy generating high trade frequency.
    Designed for parity testing between rolling and standard backtest.
    """

    INTERFACE_VERSION = 3
    can_short = True

    # Minimal ROI — let exits handle most closes
    minimal_roi = {"0": 0.03, "30": 0.015, "60": 0.005, "120": 0.0}

    stoploss = -0.02
    trailing_stop = False

    timeframe = "5m"
    startup_candle_count = 50

    # Process only new candles
    process_only_new_candles = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=8)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["volume_ma"] = ta.SMA(dataframe["volume"], timeperiod=20)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long entry: fast EMA crosses above slow + RSI not overbought
        dataframe.loc[
            (
                (dataframe["ema_fast"] > dataframe["ema_slow"])
                & (dataframe["ema_fast"].shift(1) <= dataframe["ema_slow"].shift(1))
                & (dataframe["rsi"] < 70)
                & (dataframe["volume"] > dataframe["volume_ma"] * 0.5)
            ),
            "enter_long",
        ] = 1

        # Short entry: fast EMA crosses below slow + RSI not oversold
        dataframe.loc[
            (
                (dataframe["ema_fast"] < dataframe["ema_slow"])
                & (dataframe["ema_fast"].shift(1) >= dataframe["ema_slow"].shift(1))
                & (dataframe["rsi"] > 30)
                & (dataframe["volume"] > dataframe["volume_ma"] * 0.5)
            ),
            "enter_short",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long exit: fast EMA crosses below slow or RSI overbought
        dataframe.loc[
            (
                (dataframe["ema_fast"] < dataframe["ema_slow"])
                & (dataframe["ema_fast"].shift(1) >= dataframe["ema_slow"].shift(1))
            )
            | (dataframe["rsi"] > 80),
            "exit_long",
        ] = 1

        # Short exit: fast EMA crosses above slow or RSI oversold
        dataframe.loc[
            (
                (dataframe["ema_fast"] > dataframe["ema_slow"])
                & (dataframe["ema_fast"].shift(1) <= dataframe["ema_slow"].shift(1))
            )
            | (dataframe["rsi"] < 20),
            "exit_short",
        ] = 1

        return dataframe
