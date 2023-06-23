import pandas as pd
from backtesting import Backtest
import csv
import threading

def evaluatePairs(pairList, strat):
    results = []
    for i in pairList:
        data = pd.read_csv(f"./data_directory/{i}.csv")
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data['timestamp'] = pd.to_datetime(data['timestamp'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S')))
        data.set_index('timestamp', inplace=True)
        data.drop_duplicates(inplace=True)

        bt = Backtest(data, strat, cash=1000000, commission=0.00, exclusive_orders=True)
        output = bt.run()
        results.append(output)
    averageCalculater(results, pairList)

def averageCalculater(results, pairList):
    equFinal = 0
    maxDrawdown = 0
    avgDrawdown = 0
    winRate = 0
    bestTrade = 0
    worstTrade = 0
    avgTrade = 0
    expectancy = 0
    sharpeRatio = 0
    sortinoRatio = 0
    calmarRatio = 0
    exposure = 0
    profitFactor = 0
    numbertrades = 0

    folder_path = "./results_lists/list2.csv"
    headers = ["Pair", "Equity Final [$]", "# Trades", "Max. Drawdown [%]", "Avg. Drawdown [%]", "Win Rate [%]", "Best Trade [%]", "Worst Trade [%]", "Avg. Trade [%]", "Expectancy [%]", "SQN", "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Exposure [%]", "Profit Factor"]

    with open(folder_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for i, result in enumerate(results):
            formatted_row = [
                pairList[i],
                "{:.4f}".format(result['Equity Final [$]']),
                result['# Trades'],
                "{:.4f}".format(result['Max. Drawdown [%]']),
                "{:.4f}".format(result['Avg. Drawdown [%]']),
                "{:.4f}".format(result['Win Rate [%]']),
                "{:.4f}".format(result['Best Trade [%]']),
                "{:.4f}".format(result['Worst Trade [%]']),
                "{:.4f}".format(result['Avg. Trade [%]']),
                "{:.4f}".format(result['Expectancy [%]']),
                "{:.4f}".format(result['Sharpe Ratio']),
                "{:.4f}".format(result['Sortino Ratio']),
                "{:.4f}".format(result['Calmar Ratio']),
                "{:.4f}".format(result['Exposure Time [%]']),
                "{:.4f}".format(result['Profit Factor'])
            ]
            writer.writerow(formatted_row)

            equFinal += result['Equity Final [$]']
            numbertrades += result['# Trades']
            maxDrawdown += result['Max. Drawdown [%]']
            avgDrawdown += result['Avg. Drawdown [%]']
            winRate += result['Win Rate [%]']
            bestTrade += result['Best Trade [%]']
            worstTrade += result['Worst Trade [%]']
            avgTrade += result['Avg. Trade [%]']
            expectancy += result['Expectancy [%]']
            sharpeRatio += result['Sharpe Ratio']
            sortinoRatio += result['Sortino Ratio']
            calmarRatio += result['Calmar Ratio']
            exposure += result['Exposure Time [%]']
            profitFactor += result['Profit Factor']

    averaged_equFinal = equFinal / len(results)
    averaged_maxDrawdown = maxDrawdown / len(results)
    averaged_avgDrawdown = avgDrawdown / len(results)
    averaged_winRate = winRate / len(results)
    averaged_bestTrade = bestTrade / len(results)
    averaged_worstTrade = worstTrade / len(results)
    averaged_avgTrade = avgTrade / len(results)
    averaged_expectancy = expectancy / len(results)
    averaged_calmarRatio = calmarRatio / len(results)
    averaged_sharpeRatio = sharpeRatio / len(results)
    averaged_sortinoRatio = sortinoRatio / len(results)
    averaged_calmarRatio = calmarRatio / len(results)
    averaged_exposure = exposure / len(results)
    averaged_profitFactor = profitFactor / len(results)

    print(f"Averaged Equity Final: {averaged_equFinal}")
    print(f"Averaged Max. Drawdown: {averaged_maxDrawdown}")
    print(f"Averaged Avg. Drawdown: {averaged_avgDrawdown}")
    print(f"Averaged Win Rate: {averaged_winRate}")
    print(f"Averaged Best Trade: {averaged_bestTrade}")
    print(f"Averaged Worst Trade: {averaged_worstTrade}")
    print(f"Averaged Avg. Trade: {averaged_avgTrade}")
    print(f"Averaged Expectancy: {averaged_expectancy}")
    print(f"Averaged Sharpe Ratio: {averaged_sharpeRatio}")
    print(f"Averaged Sortino Ratio: {averaged_sortinoRatio}")
    print(f"Averaged Calmar Ratio: {averaged_calmarRatio}")
    print(f"Averaged Exposure: {averaged_exposure}")
    print(f"Averaged Profit Factor: {averaged_profitFactor}")
    print(f"Total number of Trades: {numbertrades}")

    with open(folder_path, "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        formatted_row = [
            "Averages",
            "{:.4f}".format(averaged_equFinal),
            numbertrades,
            "{:.4f}".format(averaged_maxDrawdown),
            "{:.4f}".format(averaged_avgDrawdown),
            "{:.4f}".format(averaged_winRate),
            "{:.4f}".format(averaged_bestTrade),
            "{:.4f}".format(averaged_worstTrade),
            "{:.4f}".format(averaged_avgTrade),
            "{:.4f}".format(averaged_expectancy),
            "{:.4f}".format(averaged_sharpeRatio),
            "{:.4f}".format(averaged_sortinoRatio),
            "{:.4f}".format(averaged_calmarRatio),
            "{:.4f}".format(averaged_exposure),
            "{:.4f}".format(averaged_profitFactor)
        ]
        writer.writerow(formatted_row)

def format_data(dataname, timeframe="3T"):
    daten = pd.read_csv(f"./data_directory/{dataname}.csv")
    if "timefstamp" in daten.columns:
        daten['timestamp'] = pd.to_datetime(daten['timestamp'], unit='ms')
        daten['timestamp'] = pd.to_datetime(daten['timestamp'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S')))
        daten.set_index('timestamp', inplace=True)
    else:
        daten.set_index("Date", inplace=True)
        daten.index = pd.to_datetime(daten.index)
    daten.drop_duplicates(inplace=True)
    daten = daten.resample(timeframe).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
    daten.dropna(inplace=True)
    return daten