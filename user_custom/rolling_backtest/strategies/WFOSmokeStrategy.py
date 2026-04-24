"""WFO Smoke Test Strategy — has optimizable hyperopt parameters."""
from __future__ import annotations

from pandas import DataFrame

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
import talib.abstract as ta


class WFOSmokeStrategy(IStrategy):
    timeframe = "5m"
    startup_candle_count = 200
    minimal_roi = {"0": 0.03}
    stoploss = -0.05
    can_short = False

    # Optimizable parameters
    ema_fast_period = IntParameter(5, 30, default=20, space="buy", optimize=True)
    ema_slow_period = IntParameter(30, 100, default=50, space="buy", optimize=True)
    rsi_entry = IntParameter(45, 70, default=55, space="buy", optimize=True)
    rsi_exit = IntParameter(25, 50, default=40, space="sell", optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Pre-compute a range of EMAs for hyperopt
        for period in range(5, 101):
            dataframe[f"ema_{period}"] = ta.EMA(dataframe, timeperiod=period)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        fast = self.ema_fast_period.value
        slow = self.ema_slow_period.value
        dataframe.loc[
            (dataframe[f"ema_{fast}"] > dataframe[f"ema_{slow}"])
            & (dataframe["rsi"] > self.rsi_entry.value),
            ["enter_long", "enter_tag"],
        ] = (1, "ema_cross")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        fast = self.ema_fast_period.value
        slow = self.ema_slow_period.value
        dataframe.loc[
            (dataframe[f"ema_{fast}"] < dataframe[f"ema_{slow}"])
            | (dataframe["rsi"] < self.rsi_exit.value),
            ["exit_long", "exit_tag"],
        ] = (1, "ema_break")
        return dataframe
