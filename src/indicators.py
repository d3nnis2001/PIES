import pandas as pd 
import numpy as np
from datetime import datetime, date
import math
import numba
from numba import jit
from scipy.signal import argrelextrema

import numpy as np

def VWAP2(df: pd.DataFrame, band):
    # Calculate TP and TradedValue using vectorized operations
    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['TradedValue'] = df['TP'] * df['Volume']

    # Calculate cumulative values using cumsum()
    df['CumulativeTradedValue'] = df.groupby(df.index.date)['TradedValue'].cumsum()
    df['CumulativeVolume'] = df.groupby(df.index.date)['Volume'].cumsum()

    # Calculate VWAP using vectorized division
    df['VWAP'] = df['CumulativeTradedValue'] / df['CumulativeVolume']

    # Calculate TypicalPriceDev, TPVDev, and CumTPVDev using vectorized operations
    df['TypicalPriceDev'] = (df['Close'] - df['VWAP'])**2
    df['TPVDev'] = df['TypicalPriceDev'] * df['Volume']
    df['CumTPVDev'] = df.groupby(df.index.date)['TPVDev'].cumsum()

    # Calculate VWAPStdev using vectorized square root and division
    df['VWAPStdev'] = np.sqrt(df['CumTPVDev'] / df['CumulativeVolume'])

    # Calculate upper and lower bands using vectorized operations
    df['UpperBand1'] = df['VWAP'] + 1 * df['VWAPStdev']
    df['LowerBand1'] = df['VWAP'] - 1 * df['VWAPStdev']
    df['UpperBand2'] = df['VWAP'] + 2 * df['VWAPStdev']
    df['LowerBand2'] = df['VWAP'] - 2 * df['VWAPStdev']
    df['UpperBand3'] = df['VWAP'] + 3 * df['VWAPStdev']
    df['LowerBand3'] = df['VWAP'] - 3 * df['VWAPStdev']

    if band == 0:
        return df['VWAP'].values
    elif band == 1:
        return df['UpperBand1'].values
    elif band == 2:
        return df['LowerBand1'].values
    elif band == 3:
        return df['UpperBand2'].values
    elif band == 4:
        return df['LowerBand2'].values
    elif band == 5:
        return df['UpperBand3'].values
    elif band == 6:
        return df['LowerBand3'].values

@jit(nopython=True)
def calculate_final_bands(final_bands_num, final_bands_numl, upper_band, lower_band, close):
    for i in range(len(final_bands_num)):
        if i == 0:
            final_bands_num[i] = 0
        else:
            if (upper_band[i] < final_bands_num[i - 1]) or (close[i - 1] > final_bands_num[i - 1]):
                final_bands_num[i] = upper_band[i]
            else:
                final_bands_num[i] = final_bands_num[i - 1]

    for i in range(len(final_bands_numl)):
        if i == 0:
            final_bands_numl[i] = 0
        else:
            if (lower_band[i] > final_bands_numl[i - 1]) or (close[i - 1] < final_bands_numl[i - 1]):
                final_bands_numl[i] = lower_band[i]
            else:
                final_bands_numl[i] = final_bands_numl[i - 1]

    return final_bands_num, final_bands_numl

@jit(nopython=True)
def calculate_supertrend(supertrend_num, final_bands_num, final_bands_numl, close_np):
    for i in range(1, len(supertrend_num)):
        if i == 0:
            supertrend_num[i] = 0
        elif supertrend_num[i-1] == final_bands_num[i-1] and close_np[i] < final_bands_num[i]:
            supertrend_num[i] = final_bands_num[i]
        elif supertrend_num[i-1] == final_bands_num[i-1] and close_np[i] > final_bands_num[i]:
            supertrend_num[i] = final_bands_numl[i]
        elif supertrend_num[i-1] == final_bands_numl[i-1] and close_np[i] > final_bands_numl[i]:
            supertrend_num[i] = final_bands_numl[i]
        elif supertrend_num[i-1] == final_bands_numl[i-1] and close_np[i] < final_bands_numl[i]:
            supertrend_num[i] = final_bands_num[i]

    return supertrend_num

@jit(nopython=True)
def calculate_upt_dt(upt, dt, supertrend_num, close_np):
    for i in range(len(supertrend_num)):
        if close_np[i] > supertrend_num[i]:
            upt[i] = supertrend_num[i]
            dt[i] = np.nan
        elif close_np[i] < supertrend_num[i]:
            upt[i] = np.nan
            dt[i] = supertrend_num[i]
        else:
            upt[i] = np.nan
            dt[i] = np.nan
    return upt, dt

def supertrend(data, lookback, multiplier, band):
    high = data["High"]
    low = data["Low"]
    close = data["Close"]

    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
    atr = tr.ewm(lookback).mean()

    # H/L AVG AND BASIC UPPER & LOWER BAND

    hl_avg = (high + low) / 2
    upper_band = (hl_avg + multiplier * atr).fillna(method="ffill").dropna()
    lower_band = (hl_avg - multiplier * atr).dropna()

    high = high.values
    low = low.values
    close = close.values
    upper_band = upper_band.values
    lower_band = lower_band.values


    final_bands_num = np.zeros(len(upper_band))
    final_bands_numl = np.zeros(len(upper_band))

    final_bands_num, final_bands_numl = calculate_final_bands(final_bands_num, final_bands_numl, upper_band, lower_band, close)

    # Supertrend
    supertrend_num = np.zeros(len(upper_band))
    supertrend_num = calculate_supertrend(supertrend_num, final_bands_num, final_bands_numl, close)

    # ST UPTREND/DOWNTREND
    upt = np.empty(len(supertrend_num))
    upt[:] = np.nan
    dt = np.empty(len(supertrend_num))
    dt[:] = np.nan

    upt, dt = calculate_upt_dt(upt, dt, supertrend_num, close)

    if band == 0:
        return dt
    elif band == 1:
        return upt



    
def swings(data, length, timeframe):
    resampled = data.resample(timeframe).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
    upper = resampled['High'].rolling(window=length).max()
    lower = resampled['Low'].rolling(window=length).min()

    os = np.where(resampled['High'].shift(length) > upper, 0, np.where(resampled['Low'].shift(length) < lower, 1, np.nan))
    os = pd.Series(os).fillna(method='ffill')
    top = np.where((os == 0) & (os.shift() != 0), resampled['High'].shift(length), 0)
    btm = np.where((os == 1) & (os.shift() != 1), resampled['Low'].shift(length), 0)
    series1 = pd.Series(top)
    series2 = pd.Series(btm)
    series1.index = resampled.index
    series2.index = resampled.index
    return [series1, series2]

def smc(data, length, band, timeframe):
    swings_data_btm = swings(data, length, timeframe)[1]
    swings_data_top = swings(data, length, timeframe)[0]

    data["btmm"] = swings_data_btm
    data["btmm"].replace(0, np.nan, inplace=True)
    data["topp"] = swings_data_top
    data["topp"].replace(0, np.nan, inplace=True)

    fill1 = data["btmm"].fillna(method='ffill')
    fill2 = data["topp"].fillna(method='ffill')

    if(band==0):
        return fill1
    else:
        return fill2
    

# Wavetrend3D from tw


def wavetrend3d(data: pd.DataFrame, cog_window, cock , timeframe, mirror):
    s_length = 1.75
    if(cock):
        dataclose = cog(data["Close"], cog_window)
    else:
        dataclose = data['Close']

    timechange = dataclose.resample(timeframe).agg({'Close': 'last'})

    signalSlow = getOscillator(timechange["Close"])

    hourly_data = pd.DataFrame(signalSlow, index=timechange.index)

    seriesSlow = hourly_data.reindex(data.index, method='pad')
    seriesSlow = s_length * seriesSlow.squeeze()
    seriesSlowMirror = -seriesSlow
    seriesSlowret = seriesSlow.ffill()
    seriesSlowretMirror = seriesSlowMirror.ffill()
    if not mirror:
        return seriesSlowret
    elif mirror:
        return seriesSlowretMirror
    

def cog(source, length):
    sum = source.rolling(length, min_periods=1).sum()
    num = 0
    for i in range(length):
        price = source.shift(i).fillna(0)
        num += price * (i + 1)
    return -(num / sum).replace([np.inf, -np.inf], np.nan).fillna(0)

def normalizeDeriv(src, quadraticMeanLength):
    derivative = src - src.shift(2)  
    quadraticMean = math.sqrt(np.nansum(np.power(derivative, 2)) / quadraticMeanLength)
    normalizedDeriv = derivative / quadraticMean  
    series = pd.Series(index=src.index, data=normalizedDeriv)
    return series

def tanh(_src):
    vals = -1 + 2/(1 + np.exp(-2*_src))
    series = pd.Series(index=_src.index, data=vals)
    return series

def dualPoleFilter(_src, _lookback):
    _omega = -99 * math.pi / (70 * _lookback)
    _alpha = np.exp(_omega)
    _beta = -np.power(_alpha, 2)
    _gamma = np.cos(_omega) * 2 * _alpha
    _delta = 1 - _gamma - _beta
    _slidingAvg = 0.5 * (_src + np.nan_to_num(_src.shift(1), nan=_src[0]))
    _filter = np.empty_like(_src)
    _filter[0] = np.nan
    
    for i in range(1, len(_src)):
        _filter[i] = (_delta * _slidingAvg[i]) + _gamma * np.nan_to_num(_filter[i-1], nan=0) + _beta * np.nan_to_num(_filter[i-2], nan=0)
    series = pd.Series(_filter, index=_src.index)
    return series

def getOscillator(data, smoothingFrequency=40, quadraticMeanLength=50):
    nDeriv = normalizeDeriv(data, quadraticMeanLength)
    hyperbolicTangent = tanh(nDeriv)
    result = dualPoleFilter(hyperbolicTangent, smoothingFrequency)
    return result


def atr(data, window):
    high = data['High']
    low = data['Low']
    close = data['Close']

    # Calculate True Range
    tr = pd.DataFrame(index=data.index)
    tr['tr0'] = abs(high - low)
    tr['tr1'] = abs(high - close.shift())
    tr['tr2'] = abs(low - close.shift())
    tr['tr'] = tr.max(axis=1)

    # Calculate Average True Range
    tr = tr.iloc[-len(close):]  # Align shapes of tr and close
    atr = pd.Series(tr['tr']).rolling(window=window, min_periods=window).mean()

    return atr

def pivotPoints(df, order, band):
    if(band==1):
        indexes = argrelextrema(df['High'].values, np.greater, order=order)[0]
        listHigh = pd.Series(np.nan, index=df.index)
        listHigh.iloc[indexes] = df["High"].iloc[indexes]
        listHigh = listHigh.fillna(method='ffill')
        return listHigh
    elif(band==0):
        indexeslower = argrelextrema(df['Low'].values, np.less, order=order)[0]
        listLow = pd.Series(np.nan, index=df.index)
        listLow.iloc[indexeslower] = df["Low"].iloc[indexeslower]
        listLow = listLow.fillna(method="ffill")
        return listLow
    
def find_extrema(df, left_bars, right_bars):
    left_max = df['High'].shift(1).rolling(window=left_bars, min_periods=1).max()
    right_max = df['High'].shift(-right_bars).rolling(window=right_bars, min_periods=1).max()
    left_min = df['Low'].shift(1).rolling(window=left_bars, min_periods=1).min()
    right_min = df['Low'].shift(-right_bars).rolling(window=right_bars, min_periods=1).min()

    df['higher_high'] = np.where((df['High'] > left_max) & (df['High'] > right_max), df['High'], np.nan)
    df['lower_low'] = np.where((df['Low'] < left_min) & (df['Low'] < right_min), df['Low'], np.nan)
    df['higher_low'] = np.where((df['Low'] > left_min) & (df['Low'] < right_min), df['Low'], np.nan)
    df['lower_high'] = np.where((df['High'] < left_max) & (df['High'] > right_max), df['High'], np.nan)

    return df[["higher_high", "lower_low", "higher_low", "lower_high"]]

@numba.jit(nopython=True)
def choch_numba(higher_high, lower_low, higher_low, lower_high, close, low, high, right):
    chochLong = np.zeros(len(higher_high))
    chochShort = np.zeros(len(higher_high))
    last_valh = 0
    last_vall = 0
    last_valhh = 0
    last_valll = 0
    last2_valhh = 0
    last2_valll = 0
    valh_candle = 0
    vall_candle = 0
    last_trade = 0
    candlecount = 0

    for i in range(len(higher_high)):
        if higher_high[i] > 0:
            last2_valhh = last_valhh
            last_valhh = higher_high[i]
        if lower_low[i] > 0:
            last2_valll = last_valll
            last_valll = lower_low[i]
        if higher_low[i] > 0:
            last_valh = higher_low[i]
            valh_candle = candlecount
        if lower_high[i] > 0:
            last_vall = lower_high[i]
            vall_candle = candlecount
        if  high[i] > last_vall and close[i] < last_vall and candlecount - vall_candle < 10 and candlecount - vall_candle > right and last_trade > 10 and low[i - 5] < high[i] and last2_valhh > last_valhh:
            chochLong[i] = close[i]
            last_vall = 0
            vall_candle = 0
            last_trade = 0
        if  low[i] < last_valh and close[i] > last_valh and candlecount - valh_candle < 10 and candlecount - valh_candle > right and last_trade > 10 and high[i - 5] > low[i] and last2_valhh < last_valhh:
            chochShort[i] = close[i]
            last_valh = 0
            valh_candle = 0
            last_trade = 0

        last_trade += 1
        candlecount += 1

    return chochShort, chochLong


def choch(df, left, right, timeframe="3T", sholong=0):
    if timeframe != "3T":
        df = df.resample(timeframe).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
    
    highs_lows = find_extrema(df, left, right)
    df["higher_high"] = highs_lows["higher_high"]
    df["lower_low"] = highs_lows["lower_low"]
    df["higher_low"] = highs_lows["higher_low"]
    df["lower_high"] = highs_lows["lower_high"]
    
    lower_high = df["lower_high"].to_numpy()
    higher_low = df["higher_low"].to_numpy()
    higher_high = df["higher_high"].to_numpy()
    lower_low = df["lower_low"].to_numpy()
    close = df["Close"].to_numpy()
    low = df["Low"].to_numpy()
    high = df["High"].to_numpy()

    chochShort, chochLong = choch_numba(higher_high, lower_low, higher_low, lower_high, close, low, high, right)

    if sholong == 0:
        return chochShort
    else:
        return chochLong


def vwma(price, volume, window):
    price_series = pd.Series(price)
    volume_series = pd.Series(volume)
    
    weighted_prices = price_series * volume_series
    rolling_weighted_sum = weighted_prices.rolling(window).sum()
    rolling_volume_sum = volume_series.rolling(window).sum()
    
    vwma = rolling_weighted_sum / rolling_volume_sum
    return pd.Series(vwma, index=price_series.index)