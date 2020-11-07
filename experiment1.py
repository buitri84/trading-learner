import pandas as pd
import numpy as np
import util
import random
import matplotlib.pyplot as plt
import datetime as dt
import indicators as ind
import marketsimcode as mkt
import StrategyLearner as sl
import ManualStrategy as ms

def testManual(symbol = "JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), sv = 100000):
    manual = ms.ManualStrategy()
    manual_trades = manual.testPolicy(symbol, sd, ed, sv)
    manual_portvals = mkt.compute_portvals(manual_trades, sv, commission=9.95, impact=0.005)
    return manual_portvals

def testLearner(symbol = "JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), sv = 100000):
    strategy = sl.StrategyLearner(verbose=False, impact=0.005, commission=9.95)
    strategy.addEvidence(symbol, sd, ed, sv)
    strategy_trades = strategy.testPolicy(symbol, sd, ed, sv)
    strategy_portvals = mkt.compute_portvals(strategy_trades, sv, commission=9.95, impact=0.005)
    return strategy_portvals

def testBenchMark(symbol = "JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), sv = 100000):
    symbols = [symbol]
    dates = pd.date_range(sd, ed)
    data = util.get_data(symbols, dates, addSPY=False, colname='Adj Close')
    data.dropna(axis=0, how='all', inplace=True)
    benchmark_trades = pd.DataFrame(data=0.0, index=data.index, columns=[symbol])
    benchmark_trades.iloc[0] = 1000
    benchmark_trades.iloc[-1] = -1000
    benchmark_portvals = mkt.compute_portvals(benchmark_trades, sv, commission=9.95, impact=0.005)
    return benchmark_portvals

def author():
    return 'tbui61'

def main():
    # seed random once. Always use same number
    np.random.seed(90347)
    manual_portvals = testManual(symbol = "JPM", sd='2008-01-01', ed='2009-12-31', sv = 100000)
    manual_portvals_normed = manual_portvals / manual_portvals.ix[0,:]
    strategy_portvals = testLearner(symbol = "JPM", sd='2008-01-01', ed='2009-12-31', sv = 100000)
    strategy_portvals_normed = strategy_portvals / strategy_portvals.ix[0,:]
    benchmark_portvals = testBenchMark(symbol = "JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), sv = 100000)
    benchmark_portvals_normed = benchmark_portvals / benchmark_portvals.ix[0,:]
    # plot the required chart
    fig, ax = plt.subplots()
    ax.set_title('Normalized in-sample performance of Benchmark, Manual, and StrategyLearner')
    ax.plot(benchmark_portvals_normed, label='Benchmark')
    ax.plot(manual_portvals_normed, label='Manual')
    ax.plot(strategy_portvals_normed, label='StrategyLearner')
    ax.legend()
    plt.savefig('experiment1.png')
    # Extra: calculate stats
    cr_strategy, adr_strategy, sddr_strategy, sr_strategy = mkt.calculate_stats(strategy_portvals)
    cr_benchmark, adr_benchmark, sddr_benchmark, sr_benchmark = mkt.calculate_stats(benchmark_portvals)
    cr_manual, adr_manual, sddr_manual, sr_manual = mkt.calculate_stats(manual_portvals)
    #
    print("----------------------------")
    print("---- Experiment 1 ----")
    print("----------------------------")
    print(f"Cumulative return of Strategy: {cr_strategy}")  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Average Daily Return of Strategy: {adr_strategy}")  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Standard deviation of Daily return of Strategy: {sddr_strategy}")
    print(f"Sharpe ratio of Strategy: {sr_strategy}")
    print("----------------------------")
    print(f"Cumulative return of benchmark: {cr_benchmark}")  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Average Daily Return of benchmark: {adr_benchmark}")  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Standard deviation of Daily return of benchmark: {sddr_benchmark}")
    print(f"Sharpe ratio of benchmark: {sr_benchmark}")
    print("----------------------------")
    print(f"Cumulative return of Manual: {cr_manual}")  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Average Daily Return of Manual: {adr_manual}")  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Standard deviation of Daily return of Manual: {sddr_manual}")
    print(f"Sharpe ratio of Manual: {sr_manual}")
    print("----------------------------")

if __name__=="__main__":
    main()