from polygon import RESTClient
import datetime as dt
import pandas as pd
import numpy as np
import os
import time

polygonAPIkey = 'n5WUioGpm5YNuvJ0Bfn_No8nQHdfn_eP'
client = RESTClient(api_key=polygonAPIkey)

def getData(ticker, timespan, start, end):
    bars = client.get_aggs(ticker=ticker, multiplier=3, timespan=timespan, from_=start, to=end)
    print(len(bars))


    #list of polygon OptionsContract objects to DataFrame
    downloadedData = pd.DataFrame(bars)

    #create Date column
    downloadedData['Date'] = pd.to_datetime(downloadedData['timestamp'], unit='ms')
    downloadedData['Date'] = pd.to_datetime(downloadedData['Date'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S')))
    downloadedData.set_index('Date', inplace=True)

    #drop unnecessary columns
    downloadedData = downloadedData.drop(['vwap', 'transactions', 'otc'], axis=1)
    downloadedData = downloadedData.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})

    return downloadedData


def downloadAllData(ticker, timespan, start, end, csv_directory):
    current_start = start
    cumulative_data = pd.DataFrame()

    while current_start < end:
        current_end = min(current_start + pd.DateOffset(days=3), end)
        data = getData(ticker, timespan, current_start, current_end)

        if not data.empty:
            # Create directory if it doesn't exist
            os.makedirs(csv_directory, exist_ok=True)

            # Save data for the current day
            csv_file = os.path.join(csv_directory, f"{current_start.strftime('%Y-%m-%d')}.csv")
            data.to_csv(csv_file)

            # Append data to cumulative dataframe
            cumulative_data = pd.concat([cumulative_data, data])

            print(f"Downloaded data for {current_start} to {current_end}")

        current_start = current_end + pd.DateOffset(days=0)
        time.sleep(61)

    # Save cumulative data to a cumulative CSV file
    cumulative_csv_file = os.path.join(csv_directory, "cumulative_data.csv")
    cumulative_data.to_csv(cumulative_csv_file)

    return cumulative_data

"""
Example Usage:

end_date = dt.datetime.now().date()
start_date = end_date - dt.timedelta(days=700)
csv_directory = "data_directory"

data = downloadAllData("C:EURUSD", "minute", start_date, end_date, csv_directory)

""" 