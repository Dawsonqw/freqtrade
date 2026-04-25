"""
Short-term finite-window strategy for rolling backtest parity testing.

Key design: ALL indicators use finite lookback (SMA, Stochastic, Bollinger Bands,
candle patterns). NO recursive indicators (EMA, RSI-Wilder, MACD).
This should eliminate window boundary effects entirely when startup_candles >= max(lookback).

- SMA crossover (8/21) instead of EMA crossover
- Stochastic %K/%D (uses min/max over window) instead of RSI (Wilder smoothing)
- Bollinger Bands (SMA-based, finite)
- Volume SMA filter
- ATR replaced by simple high-low range SMA
"""
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta


class ShortTermFiniteStrategy(IStrategy):
    """
    Pure finite-window indicators — no recursive (EMA/RSI/MACD).
    Should produce identical results across window boundaries.
    """

    INTERFACE_VERSION = 3
    can_short = True

    minimal_roi = {
        "0": 0.015,
        "15": 0.01,
        "30": 0.005,
        "60": 0.002,
    }

    stoploss = -0.015
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True

    timeframe = "5m"
    startup_candle_count = 50  # max lookback is 21 (SMA) + buffer

    process_only_new_candles = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # SMA crossover (finite window, NOT recursive like EMA)
        dataframe["sma8"] = ta.SMA(dataframe, timeperiod=8)
        dataframe["sma21"] = ta.SMA(dataframe, timeperiod=21)

        # Stochastic %K/%D — uses min/max over window, fully finite
        stoch = ta.STOCH(dataframe, fastk_period=14, slowk_period=3, slowd_period=3,
                         slowk_matype=0, slowd_matype=0)  # matype=0 = SMA
        dataframe["slowk"] = stoch["slowk"]
        dataframe["slowd"] = stoch["slowd"]

        # Bollinger Bands (SMA-based, finite window)
        bb = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
        dataframe["bb_upper"] = bb["upperband"]
        dataframe["bb_lower"] = bb["lowerband"]
        dataframe["bb_mid"] = bb["middleband"]

        # Volume SMA (finite)
        dataframe["volume_sma"] = ta.SMA(dataframe["volume"], timeperiod=20)

        # Simple range volatility (high - low SMA, NOT ATR which uses Wilder smoothing)
        dataframe["range_sma"] = ta.SMA(dataframe["high"] - dataframe["low"], timeperiod=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long: SMA cross up + Stoch not overbought + price above BB mid + volume
        dataframe.loc[
            (
                (dataframe["sma8"] > dataframe["sma21"])
                & (dataframe["sma8"].shift(1) <= dataframe["sma21"].shift(1))
                & (dataframe["slowk"] > dataframe["slowd"])
                & (dataframe["slowk"] < 75)
                & (dataframe["close"] > dataframe["bb_mid"])
                & (dataframe["close"] < dataframe["bb_upper"])
                & (dataframe["volume"] > dataframe["volume_sma"] * 0.8)
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "sma_cross_up")

        # Short: SMA cross down + Stoch not oversold + price below BB mid + volume
        dataframe.loc[
            (
                (dataframe["sma8"] < dataframe["sma21"])
                & (dataframe["sma8"].shift(1) >= dataframe["sma21"].shift(1))
                & (dataframe["slowk"] < dataframe["slowd"])
                & (dataframe["slowk"] > 25)
                & (dataframe["close"] < dataframe["bb_mid"])
                & (dataframe["close"] > dataframe["bb_lower"])
                & (dataframe["volume"] > dataframe["volume_sma"] * 0.8)
            ),
            ["enter_short", "enter_tag"],
        ] = (1, "sma_cross_down")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long exit: SMA cross down OR Stoch overbought cross
        dataframe.loc[
            (
                (dataframe["sma8"] < dataframe["sma21"])
                & (dataframe["sma8"].shift(1) >= dataframe["sma21"].shift(1))
            )
            | (
                (dataframe["slowk"] < dataframe["slowd"])
                & (dataframe["slowk"].shift(1) >= dataframe["slowd"].shift(1))
                & (dataframe["slowk"] > 75)
            ),
            ["exit_long", "exit_tag"],
        ] = (1, "sma_cross_exit")

        # Short exit: SMA cross up OR Stoch oversold cross
        dataframe.loc[
            (
                (dataframe["sma8"] > dataframe["sma21"])
                & (dataframe["sma8"].shift(1) <= dataframe["sma21"].shift(1))
            )
            | (
                (dataframe["slowk"] > dataframe["slowd"])
                & (dataframe["slowk"].shift(1) <= dataframe["slowd"].shift(1))
                & (dataframe["slowk"] < 25)
            ),
            ["exit_short", "exit_tag"],
        ] = (1, "sma_cross_exit")

        return dataframe
