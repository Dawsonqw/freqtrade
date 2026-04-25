"""
FullMarketDynamicStrategy - 全市场动态选币策略

核心逻辑:
1. 加载全市场 535+ 合约 pairs 的 15m 数据
2. 每根 15m K线时, 在 confirm_trade_entry 中:
   - 从 dp.get_pair_dataframe 获取各 pair 原始 OHLCV
   - 实时计算波动率 (ATR%), 流动性 (volume_usd), 趋势强度 (ADX)
3. 筛选: 成交额 >= $100,000 的 pairs
4. 综合排名: 波动率(40%) + 流动性(30%) + ADX(30%), 选 top 20
5. 仅对 top 20 pair 允许入场
6. 入场: EMA(9) > EMA(21) + RSI(14) + 波动率/成交额门槛
7. 出场: EMA 交叉 + RSI 反转 + trailing stop
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import IStrategy

logger = logging.getLogger(__name__)


class FullMarketDynamicStrategy(IStrategy):
    """全市场动态选币 + 高波动率交易策略"""

    # --- 基本设置 ---
    timeframe = "15m"
    can_short = True
    startup_candle_count = 40

    # --- 仓位管理 ---
    stake_amount = 200.0
    max_open_trades = 20

    # --- 风控 ---
    stoploss = -0.03
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.015
    trailing_only_offset_is_reached = True

    # --- ROI ---
    minimal_roi = {
        "0": 0.05,
        "60": 0.03,
        "180": 0.015,
        "360": 0.005,
    }

    # --- 选币参数 ---
    min_volume_usd = 100_000
    top_n_pairs = 20
    weight_volatility = 0.4
    weight_liquidity = 0.3
    weight_adx = 0.3

    # 排名缓存: {timestamp_key: set(pair_names)}
    _rank_cache: dict = {}
    _rank_cache_max = 200

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """为每个 pair 计算交易信号指标"""
        # 波动率相关 (用于本 pair 的入场过滤)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["volatility_pct"] = dataframe["atr"] / dataframe["close"] * 100
        dataframe["volume_usd"] = dataframe["volume"] * dataframe["close"]
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)

        # 交易信号
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=9)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        return dataframe

    @staticmethod
    def _calc_atr_pct(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """从原始 OHLCV 快速计算最新 ATR%"""
        if len(close) < period + 1:
            return 0.0
        # True Range
        h = high[-(period+1):]
        l = low[-(period+1):]
        c = close[-(period+1):]
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        atr = np.mean(tr[-period:])
        last_close = c[-1]
        if last_close == 0:
            return 0.0
        return (atr / last_close) * 100

    @staticmethod
    def _calc_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """简化版 ADX 计算 (用 talib)"""
        if len(close) < period * 2:
            return 0.0
        try:
            adx = ta.ADX({"high": high, "low": low, "close": close, "open": close}, timeperiod=period)
            val = adx[-1] if len(adx) > 0 else 0.0
            return float(val) if not np.isnan(val) else 0.0
        except Exception:
            return 0.0

    def _compute_market_ranking(self, current_time: datetime) -> set[str]:
        """
        计算当前时刻全市场排名, 返回 top N pair 名称集合.
        使用 dp.get_pair_dataframe 获取原始 OHLCV, 实时计算排名指标.
        """
        ts_key = int(current_time.timestamp()) if hasattr(current_time, 'timestamp') else hash(current_time)

        if ts_key in self._rank_cache:
            return self._rank_cache[ts_key]

        # 清理过大的缓存
        if len(self._rank_cache) > self._rank_cache_max:
            recent_keys = sorted(self._rank_cache.keys())[-50:]
            self._rank_cache = {k: self._rank_cache[k] for k in recent_keys}

        dp = self.dp
        if dp is None:
            return set()

        scores = []
        for pair in dp.current_whitelist():
            try:
                df = dp.get_pair_dataframe(pair=pair, timeframe=self.timeframe)
                if df is None or len(df) < 30:
                    continue

                high = df["high"].values
                low = df["low"].values
                close = df["close"].values
                volume = df["volume"].values

                last_close = close[-1]
                last_volume = volume[-1]

                if last_close <= 0:
                    continue

                # 美元成交额 (最近 1 根 K 线)
                vol_usd = last_volume * last_close
                if vol_usd < self.min_volume_usd:
                    continue

                # ATR% 波动率
                volatility = self._calc_atr_pct(high, low, close, 14)
                if volatility <= 0:
                    continue

                # ADX 趋势强度
                adx_val = self._calc_adx(high, low, close, 14)

                scores.append({
                    "pair": pair,
                    "volatility": volatility,
                    "volume_usd": vol_usd,
                    "adx": adx_val,
                })
            except Exception:
                continue

        if not scores:
            self._rank_cache[ts_key] = set()
            return set()

        df_scores = pd.DataFrame(scores)
        df_scores["vol_rank"] = df_scores["volatility"].rank(pct=True)
        df_scores["liq_rank"] = df_scores["volume_usd"].rank(pct=True)
        df_scores["adx_rank"] = df_scores["adx"].rank(pct=True)
        df_scores["score"] = (
            self.weight_volatility * df_scores["vol_rank"]
            + self.weight_liquidity * df_scores["liq_rank"]
            + self.weight_adx * df_scores["adx_rank"]
        )

        top = df_scores.nlargest(self.top_n_pairs, "score")
        result = set(top["pair"].tolist())

        self._rank_cache[ts_key] = result
        return result

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """入场信号: 基于 EMA 交叉 + RSI + 波动率/成交额门槛"""
        # Long entry
        dataframe.loc[
            (
                (dataframe["ema_fast"] > dataframe["ema_slow"])
                & (dataframe["ema_fast"].shift(1) <= dataframe["ema_slow"].shift(1))
                & (dataframe["rsi"] < 65)
                & (dataframe["rsi"] > 30)
                & (dataframe["volume_usd"] > self.min_volume_usd)
                & (dataframe["volatility_pct"] > 0.5)
                & (dataframe["adx"] > 20)
            ),
            "enter_long",
        ] = 1

        # Short entry
        dataframe.loc[
            (
                (dataframe["ema_fast"] < dataframe["ema_slow"])
                & (dataframe["ema_fast"].shift(1) >= dataframe["ema_slow"].shift(1))
                & (dataframe["rsi"] > 35)
                & (dataframe["rsi"] < 70)
                & (dataframe["volume_usd"] > self.min_volume_usd)
                & (dataframe["volatility_pct"] > 0.5)
                & (dataframe["adx"] > 20)
            ),
            "enter_short",
        ] = 1

        dataframe.loc[dataframe["enter_long"] == 1, "enter_tag"] = "ema_cross_long"
        dataframe.loc[dataframe["enter_short"] == 1, "enter_tag"] = "ema_cross_short"

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """出场信号"""
        dataframe.loc[
            (
                (dataframe["ema_fast"] < dataframe["ema_slow"])
                | (dataframe["rsi"] > 75)
            ),
            "exit_long",
        ] = 1

        dataframe.loc[
            (
                (dataframe["ema_fast"] > dataframe["ema_slow"])
                | (dataframe["rsi"] < 25)
            ),
            "exit_short",
        ] = 1

        return dataframe

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> bool:
        """
        动态选币过滤: 每笔交易入场前检查 pair 是否在全市场 top 20.
        从 dp 获取原始 OHLCV 实时计算排名.
        """
        top_pairs = self._compute_market_ranking(current_time)
        if pair not in top_pairs:
            return False
        return True

    def custom_stake_amount(
        self,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """固定 200 USDT 仓位"""
        return min(self.stake_amount, max_stake)
