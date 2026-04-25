"""
High-frequency parity benchmark strategy for framework comparison.

Designed for comparing builtin freqtrade backtest vs rolling backtest framework.
Key properties:
  - High trade frequency (EMA crossover with momentum confirmation)
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
    Fast EMA cross + RSI + MACD strategy with strong momentum filter.
    Generates many trades but with better signal quality.
    """

    INTERFACE_VERSION = 3
    can_short = True

    # Tiered ROI - takes profit at multiple levels
    minimal_roi = {
        "0": 0.015,    # 1.5% immediate
        "15": 0.01,    # 1.0% after 75min
        "30": 0.005,   # 0.5% after 2.5h
        "60": 0.002,   # 0.2% after 5h
    }

    # Tight stoploss — limits per-trade damage
    stoploss = -0.015  # 1.5% max loss per trade

    # Trailing stop for capturing trends
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True

    timeframe = "5m"
    startup_candle_count = 50

    process_only_new_candles = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Fast/slow EMA for trend
        dataframe["ema8"] = ta.EMA(dataframe, timeperiod=8)
        dataframe["ema21"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)

        # RSI for momentum filter
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        # MACD for trend confirmation
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd"] = macd["macd"]
        dataframe["macd_signal"] = macd["macdsignal"]
        dataframe["macd_hist"] = macd["macdhist"]

        # Bollinger Bands for volatility
        bb = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe["bb_upper"] = bb["upperband"]
        dataframe["bb_lower"] = bb["lowerband"]
        dataframe["bb_mid"] = bb["middleband"]

        # Volume filter
        dataframe["volume_ma"] = ta.SMA(dataframe["volume"], timeperiod=20)

        # ATR for volatility
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long: EMA cross up + MACD bullish + RSI in range + trend aligned
        dataframe.loc[
            (
                (dataframe["ema8"] > dataframe["ema21"])
                & (dataframe["ema8"].shift(1) <= dataframe["ema21"].shift(1))
                & (dataframe["macd"] > dataframe["macd_signal"])  # MACD bullish
                & (dataframe["close"] > dataframe["ema50"])  # Above longer trend
                & (dataframe["rsi"] > 40)
                & (dataframe["rsi"] < 65)
                & (dataframe["close"] > dataframe["bb_lower"])
                & (dataframe["volume"] > dataframe["volume_ma"] * 0.8)
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "ema_cross_up")

        # Short: EMA cross down + MACD bearish + RSI in range + trend aligned
        dataframe.loc[
            (
                (dataframe["ema8"] < dataframe["ema21"])
                & (dataframe["ema8"].shift(1) >= dataframe["ema21"].shift(1))
                & (dataframe["macd"] < dataframe["macd_signal"])  # MACD bearish
                & (dataframe["close"] < dataframe["ema50"])  # Below longer trend
                & (dataframe["rsi"] > 35)
                & (dataframe["rsi"] < 60)
                & (dataframe["close"] < dataframe["bb_upper"])
                & (dataframe["volume"] > dataframe["volume_ma"] * 0.8)
            ),
            ["enter_short", "enter_tag"],
        ] = (1, "ema_cross_down")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long exit: EMA cross down OR RSI extreme overbought
        dataframe.loc[
            (
                (dataframe["ema8"] < dataframe["ema21"])
                & (dataframe["ema8"].shift(1) >= dataframe["ema21"].shift(1))
            )
            | (dataframe["rsi"] > 78),
            ["exit_long", "exit_tag"],
        ] = (1, "ema_cross_exit")

        # Short exit: EMA cross up OR RSI extreme oversold
        dataframe.loc[
            (
                (dataframe["ema8"] > dataframe["ema21"])
                & (dataframe["ema8"].shift(1) <= dataframe["ema21"].shift(1))
            )
            | (dataframe["rsi"] < 22),
            ["exit_short", "exit_tag"],
        ] = (1, "ema_cross_exit")

        return dataframe
