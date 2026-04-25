"""
High-frequency parity benchmark strategy for framework comparison.

Designed for comparing builtin freqtrade backtest vs rolling backtest framework.
Key properties:
  - High trade frequency (EMA crossover with short periods)
  - Resource-safe: fixed stake_amount, stoploss prevents blowup
  - Deterministic: no randomness, no external data, no lookahead
  - Both long and short trades
  - Multiple exit paths (signal, ROI, stoploss) for thorough coverage
"""
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta


class FrameworkBenchStrategy(IStrategy):
    """
    Fast EMA cross + RSI + Bollinger Band strategy.
    Generates many trades across diverse market conditions.
    """

    INTERFACE_VERSION = 3
    can_short = True

    # Tiered ROI - takes profit at multiple levels
    minimal_roi = {
        "0": 0.025,    # 2.5% immediate
        "20": 0.015,   # 1.5% after 20 candles (100min)
        "60": 0.008,   # 0.8% after 60 candles (5h)
        "120": 0.003,  # 0.3% after 10h
        "240": 0.0,    # breakeven after 20h
    }

    # Conservative stoploss — won't blow up the account
    stoploss = -0.03  # 3% max loss per trade

    # Trailing stop for trend-following
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.015
    trailing_only_offset_is_reached = True

    timeframe = "5m"
    startup_candle_count = 50

    process_only_new_candles = True

    # Position sizing: use fixed amount, not unlimited
    # This ensures we always have capital for new trades
    # (config should set stake_amount to a fraction of wallet)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Fast/slow EMA for trend
        dataframe["ema8"] = ta.EMA(dataframe, timeperiod=8)
        dataframe["ema21"] = ta.EMA(dataframe, timeperiod=21)

        # RSI for momentum filter
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        # Bollinger Bands for volatility
        bb = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe["bb_upper"] = bb["upperband"]
        dataframe["bb_lower"] = bb["lowerband"]
        dataframe["bb_mid"] = bb["middleband"]

        # Volume filter
        dataframe["volume_ma"] = ta.SMA(dataframe["volume"], timeperiod=20)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long: EMA cross up + RSI not overbought + above lower BB
        dataframe.loc[
            (
                (dataframe["ema8"] > dataframe["ema21"])
                & (dataframe["ema8"].shift(1) <= dataframe["ema21"].shift(1))
                & (dataframe["rsi"] < 70)
                & (dataframe["rsi"] > 30)
                & (dataframe["close"] > dataframe["bb_lower"])
                & (dataframe["volume"] > dataframe["volume_ma"] * 0.5)
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "ema_cross_up")

        # Short: EMA cross down + RSI not oversold + below upper BB
        dataframe.loc[
            (
                (dataframe["ema8"] < dataframe["ema21"])
                & (dataframe["ema8"].shift(1) >= dataframe["ema21"].shift(1))
                & (dataframe["rsi"] > 30)
                & (dataframe["rsi"] < 70)
                & (dataframe["close"] < dataframe["bb_upper"])
                & (dataframe["volume"] > dataframe["volume_ma"] * 0.5)
            ),
            ["enter_short", "enter_tag"],
        ] = (1, "ema_cross_down")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long exit: EMA cross down or RSI extremely overbought
        dataframe.loc[
            (
                (dataframe["ema8"] < dataframe["ema21"])
                & (dataframe["ema8"].shift(1) >= dataframe["ema21"].shift(1))
            )
            | (dataframe["rsi"] > 80),
            ["exit_long", "exit_tag"],
        ] = (1, "ema_cross_exit")

        # Short exit: EMA cross up or RSI extremely oversold
        dataframe.loc[
            (
                (dataframe["ema8"] > dataframe["ema21"])
                & (dataframe["ema8"].shift(1) <= dataframe["ema21"].shift(1))
            )
            | (dataframe["rsi"] < 20),
            ["exit_short", "exit_tag"],
        ] = (1, "ema_cross_exit")

        return dataframe
