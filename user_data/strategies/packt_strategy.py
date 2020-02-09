# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from joblib import load
import numpy as np

# TODO fazer essa porra com regressao funcionar

class PacktStrategy(IStrategy):
    """

    author@: Gert Wohlgemuth

    idea:
        momentum based strategie. The main idea is that it closes trades very quickly, while avoiding excessive losses. Hence a rather moderate stop loss in this case
    """

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    # minimal_roi = {
    #     "100": 0.01,
    #     "30": 0.03,
    #     "15": 0.06,
    #     "10": 0.15,
    # }

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.25

    # Optimal ticker interval for the strategy
    ticker_interval = '5m'


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Load the pre-built regressor
        dataframe['historical_mean'] = ta.MA(dataframe, timeperiod=33)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        regressor = load(
            "/user_data/ml_scripts/BTC_USDT-5m_33_40_regressor.joblib")
        pred = regressor.predict(np.array(dataframe['close'].iloc[-33:]).reshape(1, -1))

        dataframe['pred_max'] = np.max(pred)

        dataframe.loc[
            (
                    (dataframe['pred_max'] > dataframe['historical_mean'])
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        regressor = load(
            "/user_data/ml_scripts/BTC_USDT-5m_33_40_regressor.joblib")
        pred = regressor.predict(np.array(dataframe['close'].iloc[-33:]).reshape(1, -1))

        dataframe['pred_max'] = np.max(pred)

        dataframe.loc[
            (
                (dataframe['pred_max'] < dataframe['historical_mean'])
            ),
            'sell'] = 1
        return dataframe
