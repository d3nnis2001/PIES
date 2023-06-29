import pandas as pd
from backtesting import Backtest
import csv
import os
import matplotlib.pyplot as plt
import threading

def evaluatePairs(pairList, strat):
    results = []
    for i in pairList:
        data = pd.read_csv(f"./data_directory/{i}.csv")
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data['timestamp'] = pd.to_datetime(data['timestamp'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S')))
        data.set_index('timestamp', inplace=True)
        data.drop_duplicates(inplace=True)

        bt = Backtest(data, strat, cash=10000, commission=0.000045, exclusive_orders=True, margin=0.01)
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

def profitFactorBars(stats):
    statistics = stats['_trades']
    statistics["Hour"] = statistics["EntryTime"].dt.hour
    profits = statistics[statistics['PnL'] > 0].groupby('Hour')['PnL'].sum()
    losses = statistics[statistics['PnL'] < 0].groupby('Hour')['PnL'].sum()
    profitfactor = profits / abs(losses)
    plt.figure(figsize=(20,10))
    plt.bar(profitfactor.index, profitfactor, color='blue')
    plt.axhline(y=1, color='red', linestyle='--')
    plt.xlabel('Hour')
    plt.ylabel('Profit Factor')
    plt.show()

def barsWinLoss(stats):    
    statistics = stats['_trades']
    statistics["Hold"] = statistics["ExitBar"] - statistics["EntryBar"]
    print(f'Der durchschnittliche Trade wird {int(statistics["Hold"].mean())} Bars gehalten')
    statistics["Hour"] = statistics["EntryTime"].dt.hour
    statistics["Equity"] = 10000
    for i in range(1, len(statistics)):
        statistics["Equity"].iloc[i] = statistics["Equity"].iloc[i-1] + statistics["PnL"].iloc[i-1]
    statistics["Procent"] = statistics["PnL"] / statistics["Equity"] * 100
    groupedwins = statistics[statistics['Procent'] > 0].groupby('Hour')['Procent'].sum() / statistics[statistics['Procent'] > 0].groupby('Hour')['Procent'].count()
    groupedlosses = statistics[statistics['Procent'] < 0].groupby('Hour')['Procent'].sum() / statistics[statistics['Procent'] < 0].groupby('Hour')['Procent'].count()
    plt.figure(figsize=(20,10))
    plt.bar(groupedwins.index, groupedwins, color='g')
    plt.bar(groupedlosses.index, abs(groupedlosses), color='r')
    plt.xlabel('Hour')
    plt.ylabel('PnL in %')
    plt.show()

def profitYearly(stats):
    statistics = stats['_trades']
    statistics["Year"] = statistics["EntryTime"].dt.year
    profits = statistics.groupby('Year')['PnL'].sum()
    statistics["Equity"] = 10000
    for i in range(1, len(statistics)):
        statistics["Equity"].iloc[i] = statistics["Equity"].iloc[i-1] + statistics["PnL"].iloc[i-1]
    grouped = statistics.groupby('Year')['Equity'].first()
    procent = profits / grouped * 100
    plt.figure(figsize=(20,10))
    plt.bar(procent.index, procent, color='purple')
    plt.xlabel('Year')
    plt.ylabel('Profit in %')
    plt.show()
    return procent

def profitMonthly(stats):
    statistics = stats['_trades']
    statistics["Month"] = statistics["EntryTime"].dt.month
    profits = statistics.groupby('Month')['PnL'].sum()
    statistics["Equity"] = 10000
    for i in range(1, len(statistics)):
        statistics["Equity"].iloc[i] = statistics["Equity"].iloc[i-1] + statistics["PnL"].iloc[i-1]
    plt.figure(figsize=(20,10))
    plt.bar(profits.index, profits, color='orange')
    plt.xlabel('Month')
    plt.ylabel('Profit in %')
    plt.show()
    return profits

def profitMonthlyYearly(stats):
    statistics = stats['_trades']
    statistics["Month"] = statistics["EntryTime"].dt.month
    statistics["Year"] = statistics["EntryTime"].dt.year
    profits = statistics.groupby(["Year", "Month"])['PnL'].sum()
    num_years = statistics['Year'].nunique()
    first_year = statistics['Year'].min()

    fig, axes = plt.subplots(nrows=num_years, ncols=1, figsize=(8, 4*num_years))

    counter = 0

    for i in range(first_year, first_year+num_years):
        ax = axes[counter]
        counter += 1
        data = profits.loc[i]
        ax.bar(data.index, data.values)
        ax.set_title(f'PnL by Month ({i})')
        ax.set_xlabel('Month')
        ax.set_ylabel('PnL')
        
    plt.tight_layout()
    return profits