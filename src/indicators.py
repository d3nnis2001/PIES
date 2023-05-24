import pandas as pd 
import numpy as np
from datetime import datetime

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

def supertrend(data, lookback=10, multiplier=2.5, band=0):
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
    final_bands = pd.DataFrame(columns = ['upper', 'lower'])
    final_bands.iloc[:,0] = [x for x in upper_band - upper_band]
    final_bands.iloc[:,1] = final_bands.iloc[:,0]
    for i in range(len(final_bands)):
        if i == 0:
            final_bands.iloc[i,0] = 0
        else:
            if (upper_band[i] < final_bands.iloc[i-1,0]) | (close[i-1] > final_bands.iloc[i-1,0]):
                final_bands.iloc[i,0] = upper_band[i]
            else:
                final_bands.iloc[i,0] = final_bands.iloc[i-1,0]
    
    # FINAL LOWER BAND
    
    for i in range(len(final_bands)):
        if i == 0:
            final_bands.iloc[i, 1] = 0
        else:
            if (lower_band[i] > final_bands.iloc[i-1,1]) | (close[i-1] < final_bands.iloc[i-1,1]):
                final_bands.iloc[i,1] = lower_band[i]
            else:
                final_bands.iloc[i,1] = final_bands.iloc[i-1,1]
    
    # SUPERTREND
    
    supertrend = pd.DataFrame(index=upper_band.index, columns=[f'supertrend_{lookback}'])
    supertrend.iloc[:, 0] = upper_band.values
    supertrend.iloc[:,0] = [x for x in final_bands['upper'] - final_bands['upper']]
    
    for i in range(1, len(supertrend)):
        if i == 0:
            supertrend.iloc[i, 0] = 0
        elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 0] and close[i] < final_bands.iloc[i, 0]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 0]
        elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 0] and close[i] > final_bands.iloc[i, 0]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 1]
        elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 1] and close[i] > final_bands.iloc[i, 1]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 1]
        elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 1] and close[i] < final_bands.iloc[i, 1]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 0]
    
    supertrend = supertrend.set_index(upper_band.index)
    supertrend = supertrend.dropna()[1:]
    
    # ST UPTREND/DOWNTREND
    
    upt = []
    dt = []
    close = close.iloc[len(close) - len(supertrend):]

    for i in range(len(supertrend)):
        if close[i] > supertrend.iloc[i, 0]:
            upt.append(supertrend.iloc[i, 0])
            dt.append(np.nan)
        elif close[i] < supertrend.iloc[i, 0]:
            upt.append(np.nan)
            dt.append(supertrend.iloc[i, 0])
        else:
            upt.append(np.nan)
            dt.append(np.nan)
    new_index = pd.date_range(start= data.index[0] , periods=1, freq='3min')
    st, upt, dt = pd.Series(supertrend.iloc[:, 0]), pd.Series(upt), pd.Series(dt)
    upt.index, dt.index = supertrend.index, supertrend.index
    upt = pd.concat([pd.Series([np.nan], index=new_index), upt])
    dt = pd.concat([pd.Series([np.nan], index=new_index), dt])
    data["upt"] = upt
    data["upt"].index = data.index
    data["dt"] = dt
    data["dt"].index = data.index
    if(band==0):
        return data["dt"]
    elif(band==1):
        return data["upt"]
    
def swings(data, length):
    upper = data['High'].rolling(window=length).max()
    lower = data['Low'].rolling(window=length).min()

    os = np.where(data['High'].shift(length) > upper, 0, np.where(data['Low'].shift(length) < lower, 1, np.nan))
    os = pd.Series(os).fillna(method='ffill')
    top = np.where((os == 0) & (os.shift() != 0), data['High'].shift(length), 0)
    btm = np.where((os == 1) & (os.shift() != 1), data['Low'].shift(length), 0)
    return [pd.Series(top).shift(-length), pd.Series(btm).shift(-length)]


def smc(data, length, band):
    swings_data_btm = swings(data, length)[1]
    swings_data_top = swings(data, length)[0]

    swings_data_btm.index = data.index
    swings_data_top.index = data.index
    data["btmm"] = swings_data_btm
    data["btmm"].replace(0, np.nan, inplace=True)
    data["topp"] = swings_data_top
    data["topp"].replace(0, np.nan, inplace=True)
    if(band==0):
        return data["btmm"]
    else:
        return data["topp"]