import pandas as pd

def heikin_ashi(df):
    heikin_ashi_df = pd.DataFrame(index=df.index, columns=['Open', 'High', 'Low', 'Close'])

    heikin_ashi_df['Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    heikin_ashi_df['Close'] = heikin_ashi_df['Close'].apply(lambda x: round(x, 5))

    prev_date = None
    for i in range(len(df)):
        current_date = df.index[i].date()
        if current_date != prev_date:
            # Reset the Open value for a new date
            heikin_ashi_df.iat[i, 0] = df['Open'].iloc[i]

        else:
            heikin_ashi_df.iat[i, 0] = (heikin_ashi_df.iat[i-1, 0] + heikin_ashi_df.iat[i-1, 3]) / 2
            heikin_ashi_df.iat[i, 0] = round(heikin_ashi_df.iat[i, 0], 5)
        prev_date = current_date

    heikin_ashi_df['High'] = heikin_ashi_df.loc[:, ['Open', 'Close']].join(df['High']).max(axis=1)
    heikin_ashi_df['High'] = heikin_ashi_df['High'].apply(lambda x: round(x, 5))

    heikin_ashi_df['Low'] = heikin_ashi_df.loc[:, ['Open', 'Close']].join(df['Low']).min(axis=1)
    heikin_ashi_df['Low'] = heikin_ashi_df['Low'].apply(lambda x: round(x, 5))

    # Convert back to OHLC format
    ohlc_df = heikin_ashi_df[['Open', 'High', 'Low', 'Close']].copy()
    ohlc_df.index = df.index

    return ohlc_df
