from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Union
import pandas as pd
import numpy as np
import warnings
import pdb


@dataclass
class finTS(ABC):
    ticker: Union[str, list[str]]
    raw_data: pd.DataFrame
    freq: str = 'B'

    # frequency independent methods
    # Any alignment here would be possible
    def align_to(self, target_index: pd.DatetimeIndex, method: str = 'ffill'):
        return self.raw_data.reindex(target_index).fillna(method=method)

    def to_freq(self, freq: str = None, ffill_method: str = 'ffill'):
        """Handles data resampling"""

        if freq is not None:
            data = self.raw_data.asfreq(freq, method=ffill_method)
        else:
            # else fill forward without resampling, unless method is none
            data = self.raw_data.fillna(method=ffill_method)
        return data


@dataclass
class AssetSeries(finTS):

    # property cannot take input arguments
    # this does the heavy lifting

    def get_returns(self, freq: str = None, type: str = 'log') -> pd.DataFrame:
        # TODO: freq has to be a pandas thingy, need to give the list of those here

        # Resample the data to the desired frequency, use pandas offset aliases
        target_freq = freq if freq is not None else self.freq

        data = self.to_freq(freq=target_freq)

        if type == 'log':
            returns = np.log(data/data.shift(1))
        elif type == 'simple':
            returns = data/data.shift(1) - 1
        else:
            raise ValueError("type must be either 'log' or 'simple'")

        if np.isnan(returns.values).any():
            # base class for warnings about dubious runtime behavior
            warnings.warn(
                "There are remaining NaNs in the series", RuntimeWarning)
        return returns

    @property
    def log_returns(self) -> pd.DataFrame:
        return self.get_returns(type='log')

    @property
    def simple_returns(self) -> pd.DataFrame:
        return self.get_returns(type='simple')


@dataclass
class IndicatorSeries(finTS):

    def zscore(self, lookback: int = 21):
        means = self.raw_data.rolling(
            window=lookback, min_periods=lookback).mean()
        std = self.raw_data.rolling(
            window=lookback, min_periods=lookback).std()

        return (self.raw_data - means)/std
