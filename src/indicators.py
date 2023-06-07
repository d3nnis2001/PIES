import pandas as pd 
import numpy as np
from datetime import datetime
import math

def VWAP2(df: pd.DataFrame, band):
    # Group by date
    grouped = df.groupby(df.index.date)

    # Initialize empty lists for the bands
    vwap_list = []
    upper_band1_list = []
    lower_band1_list = []
    upper_band2_list = []
    lower_band2_list = []
    upper_band3_list = []
    lower_band3_list = []

    # Iterate over each group and calculate the bands
    for _, group in grouped:
        group['TP'] = (group['High'] + group['Low'] + group['Close']) / 3
        group['TradedValue'] = group['TP'] * group['Volume']
        group['CumulativeTradedValue'] = group['TradedValue'].cumsum()
        group['CumulativeVolume'] = group['Volume'].cumsum()
        group['VWAP'] = group['CumulativeTradedValue'] / group['CumulativeVolume']

        group['TypicalPriceDev'] = (group['Close'] - group['VWAP'])**2
        group['TPVDev'] = group['TypicalPriceDev'] * group['Volume']
        group['CumTPVDev'] = group['TPVDev'].cumsum()
        group['VWAPStdev'] = np.sqrt(group['CumTPVDev'] / group['CumulativeVolume'])

        # Calculate upper and lower bands for the group
        group['UpperBand1'] = group['VWAP'] + 1 * group['VWAPStdev']
        group['LowerBand1'] = group['VWAP'] - 1 * group['VWAPStdev']
        group['UpperBand2'] = group['VWAP'] + 2 * group['VWAPStdev']
        group['LowerBand2'] = group['VWAP'] - 2 * group['VWAPStdev']
        group['UpperBand3'] = group['VWAP'] + 3 * group['VWAPStdev']
        group['LowerBand3'] = group['VWAP'] - 3 * group['VWAPStdev']

        # Append the bands to the respective lists
        vwap_list.extend(group['VWAP'])
        upper_band1_list.extend(group['UpperBand1'])
        lower_band1_list.extend(group['LowerBand1'])
        upper_band2_list.extend(group['UpperBand2'])
        lower_band2_list.extend(group['LowerBand2'])
        upper_band3_list.extend(group['UpperBand3'])
        lower_band3_list.extend(group['LowerBand3'])

    if band == 0:
        return pd.Series(vwap_list)
    elif band == 1:
        return pd.Series(upper_band1_list)
    elif band == 2:
        return pd.Series(lower_band1_list)
    elif band == 3:
        return pd.Series(upper_band2_list)
    elif band == 4:
        return pd.Series(lower_band2_list)
    elif band == 5:
        return pd.Series(upper_band3_list)
    elif band == 6:
        return pd.Series(lower_band3_list)

def supertrend(data, lookback, multiplier, band):
    high = data["High"]
    low = data["Low"]
    close = data["Close"]
    # ATR

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

    # FINAL UPPER BAND
    final_bands = pd.DataFrame(columns = ['upper', 'lower'], index=upper_band.index)
    final_bands["upper"] = 0.0
    final_bands["lower"] = 0.0

    final_bands_num = final_bands["upper"].values
    final_bands_numl = final_bands["lower"].values

    # FINAL UPPER BAND

    for i in range(len(final_bands)):
        if i == 0:
            final_bands_num[i] = 0
        else:
            if (upper_band[i] < final_bands_num[i-1]) | (close[i-1] > final_bands_num[i-1]):
                final_bands_num[i] = upper_band[i]
            else:
                final_bands_num[i] = final_bands_num[i-1]

    # FINAL LOWER BAND

    for i in range(len(final_bands)):
        if i == 0:
            final_bands_numl[i] = 0
        else:
            if (lower_band[i] > final_bands_numl[i-1]) | (close[i-1] < final_bands_numl[i-1]):
                final_bands_numl[i] = lower_band[i]
            else:
                final_bands_numl[i] = final_bands_numl[i-1]

    # Supertrend

    supertrend = pd.DataFrame(columns = ['supertrend'], index=upper_band.index)
    supertrend["supertrend"] = 0.0

    supertrend_num = supertrend["supertrend"].values

    for i in range(1, len(supertrend)):
        if i == 0:
            supertrend_num[i] = 0
        elif supertrend_num[i-1] == final_bands_num[i-1] and close[i] < final_bands_num[i]:
            supertrend_num[i] = final_bands_num[i]
        elif supertrend_num[i-1] == final_bands_num[i-1] and close[i] > final_bands_num[i]:
            supertrend_num[i] = final_bands_numl[i]
        elif supertrend_num[i-1] == final_bands_numl[i-1] and close[i] > final_bands_numl[i]:
            supertrend_num[i] = final_bands_numl[i]
        elif supertrend_num[i-1] == final_bands_numl[i-1] and close[i] < final_bands_numl[i]:
            supertrend_num[i] = final_bands_num[i]

    # ST UPTREND/DOWNTREND

    upt = []
    dt = []

    for i in range(len(supertrend)):
        if close[i] > supertrend_num[i]:
            upt.append(supertrend_num[i])
            dt.append(np.nan)
        elif close[i] < supertrend_num[i]:
            upt.append(np.nan)
            dt.append(supertrend_num[i])
        else:
            upt.append(np.nan)
            dt.append(np.nan)

    new_index = pd.date_range(start= data.index[0] , periods=1, freq='3min')
    st = pd.Series(supertrend_num, index=data.index)
    upt = pd.Series(upt, index=data.index)
    dt = pd.Series(dt, index=data.index)
    upt = pd.concat([pd.Series([np.nan], index=new_index), upt])
    dt = pd.concat([pd.Series([np.nan], index=new_index), dt])

    dt = dt[~dt.index.duplicated()]
    upt = upt[~upt.index.duplicated()]

    if(band == 0):
        return dt
    elif(band == 1):
        return upt
    
def swings(data, length, timeframe):
    resampled = data.resample(timeframe).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
    upper = resampled['High'].rolling(window=length).max()
    lower = resampled['Low'].rolling(window=length).min()

    os = np.where(resampled['High'].shift(length) > upper, 0, np.where(resampled['Low'].shift(length) < lower, 1, np.nan))
    os = pd.Series(os).fillna(method='ffill')
    top = np.where((os == 0) & (os.shift() != 0), resampled['High'].shift(length), 0)
    btm = np.where((os == 1) & (os.shift() != 1), resampled['Low'].shift(length), 0)
    series1 = pd.Series(top).shift(-length)
    series2 = pd.Series(btm).shift(-length)
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


def wavetrend3d(data: pd.Series, cog_window, timeframe, mirror):
    s_length = 1.75
    timechange = data.resample(timeframe).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})

    # Store the original index
    original_index = data.index

    signalSlow = getOscillator(cog(timechange["Close"], 6))

    # Create a DataFrame with the hourly data and original index
    hourly_data = pd.DataFrame(signalSlow, index=timechange.index)

    # Map the hourly data back to the original 3-minute index
    seriesSlow = hourly_data.asof(original_index)
    seriesSlow = s_length * seriesSlow.squeeze()
    seriesSlowMirror = -seriesSlow
    seriesSlowret = seriesSlow.fillna(0)
    seriesSlowretMirror = seriesSlowMirror.fillna(0)
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
    derivative = src - src.shift(2)  # Calculate the derivative
    quadraticMean = np.sqrt(np.nan_to_num(np.sum(np.power(derivative, 2)) / quadraticMeanLength, nan=0))
    normalizedDeriv = derivative / quadraticMean  # Calculate normalized derivative
    series = pd.Series(index=src.index, data=normalizedDeriv*100)
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

def getOscillator(data, smoothingFrequency=50, quadraticMeanLength=50):
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