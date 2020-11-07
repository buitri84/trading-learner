import datetime as dt  		 	 		  	 	 			  	 
import pandas as pd
import numpy as np  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import util  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import random
import indicators as ind
import marketsimcode as mkt
import matplotlib.pyplot as plt  	
import BagLearner as bl
import RTLearner as rtl  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
class StrategyLearner(object):  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # constructor  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    def __init__(self, verbose = False, impact=0.005, commission=9.95):  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        self.verbose = verbose  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        self.impact = impact  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        self.commission = commission
        self.learner = bl.BagLearner(learner=rtl.RTLearner, kwargs={'leaf_size': 20}, bags=25, \
            boost=False, verbose=verbose)

    def set_impact(self, impact):
        self.impact = impact
    
    # Old function to get price dataframe     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    def getData(self, symbol, sd, ed):
        # Prepare symbols and dates to retrieve data
        symbols = [symbol]
        dates = pd.date_range(sd, ed)
        data = util.get_data(symbols, dates, addSPY=False, colname='Adj Close')
        data.dropna(axis=0, how='all', inplace=True)
        return data

    def generate_trade_from_signal(self, df):
        df.replace(to_replace=0, value=np.nan, inplace=True)
        df.fillna(method='ffill', inplace=True)
        df.fillna(value=0, inplace=True)
        # diff between 2 consecutive elements gives a non-sero signal for first trade in each sequence
        signal = df.diff(periods=1)
        # process the first element of signal, which is currently NaN
        if (df.iloc[0] != 0)[0]:
            signal.iloc[0] = df.iloc[0]
        else:
            signal.iloc[0] = 0
        # Now signal is just the original df where signal != 0
        signal = df[signal != 0]
        signal.fillna(value=0, inplace=True)
        signal *= 2000
        # first trade is always just -1000 or 1000, so need to divide by 2
        signal.loc[signal.ne(0).idxmax()] /= 2
        return signal

    # Generate Y labels (-1, 0, 1) from Xtrain (features) data. NOTE: input is still price DF sue to easy shift
    # Ysell = return threshold for SELL. Ybuy = return threshold for BUY. Nday = window to look into future
    def generate_Ytrain(self, df_price, Ysell=-0.1, Ybuy=0.1, Nday=20, impact=0.005):
        ret = df_price.shift(-1 * Nday) / df_price - 1
        ret.fillna(value=0, inplace=True)
        anchor = 100
        f = lambda x: x < Ysell - impact
        ret[f(ret)] = -1 * anchor
        f = lambda x: x > Ybuy + impact
        ret[f(ret)] = anchor
        f = lambda x: abs(x) < anchor
        ret[f(ret)] = 0
        ret = ret / anchor
        return ret.to_numpy()

    # this method should create a QLearner, and train it for trading  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    def addEvidence(self, symbol = "IBM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 10000):   	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        # get data  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 		  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        df_data = self.getData(symbol, sd, ed) 
        # Calculate 3 indicators and append into numpy ndarray (the X's)
        df_ROC = ind.calculate_ROC(df_data, period=22)
        df_Bollinger = ind.calculate_Bollinger(df_data, period=16, dev=2)
        df_Stochastic = ind.calculate_Stochastic(df_data, lookback=20, smooth=3) 
        ROC = df_ROC.to_numpy()
        Bollinger = df_Bollinger.to_numpy()
        Stochastic = df_Stochastic.to_numpy()
        Xtrain = np.concatenate((ROC, Bollinger, Stochastic), axis=1)
        np.nan_to_num(Xtrain, copy=False)
        # generate Ytrain
        Ytrain = self.generate_Ytrain(df_data, Ysell=-0.07, Ybuy=0.07, Nday=20, impact=self.impact)
        # train the RT learner. Then evaluate performance on training data
        self.learner.addEvidence(Xtrain, Ytrain)		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # this method should use the existing policy and test it against new data  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    def testPolicy(self, symbol = "IBM", sd=dt.datetime(2009,1,1), ed=dt.datetime(2010,1,1), sv = 10000):  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        # get data  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 		  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        df_data = self.getData(symbol, sd, ed) 
        # Calculate 3 indicators and append into numpy ndarray (the X's)
        df_ROC = ind.calculate_ROC(df_data, period=22)
        df_Bollinger = ind.calculate_Bollinger(df_data, period=16, dev=2)
        df_Stochastic = ind.calculate_Stochastic(df_data, lookback=20, smooth=3) 
        ROC = df_ROC.to_numpy()
        Bollinger = df_Bollinger.to_numpy()
        Stochastic = df_Stochastic.to_numpy()
        Xtest = np.concatenate((ROC, Bollinger, Stochastic), axis=1)
        np.nan_to_num(Xtest, copy=False)
        # Now query existing policy and evaluate performance
        Yout = self.learner.query(Xtest)
        df_Yout = pd.DataFrame(Yout, index=df_data.index, columns=[symbol])
        df_trades = self.generate_trade_from_signal(df_Yout)		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        return df_trades  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 

    # evaluate performance, if called from main()
    def evaluateTrades(self, symbol, df_trades, sd, ed, sv):
        # plot trades and prices
        data = self.getData(symbol, sd, ed) 
        fig, ax = plt.subplots()
        ax.set_title('Prices and executed trades')
        ax.plot(data, label=symbol)
        for buy_point in df_trades.index[df_trades[symbol] >= 1000].tolist():
            ax.axvline(buy_point, color='b')
        for sell_point in df_trades.index[df_trades[symbol] <= -1000].tolist():
            ax.axvline(sell_point, color='k')
        ax.legend()
        plt.savefig('Learner_trades.png')   
        # Benchmark: buy once and hold
        benchmark_trades = pd.DataFrame(data=0.0, index=data.index, columns=[symbol])
        benchmark_trades.iloc[0] = 1000
        benchmark_portvals = mkt.compute_portvals(benchmark_trades, start_val=sv, impact=self.impact, commission=self.commission)
        benchmark_portvals_normed = benchmark_portvals / benchmark_portvals.ix[0,:]
        # Learner trades
        strategy_portvals = mkt.compute_portvals(df_trades, start_val=sv, impact=self.impact, commission=self.commission)
        strategy_portvals_normed = strategy_portvals / strategy_portvals.ix[0,:]
        # Calculate stats
        cr_benchmark, adr_benchmark, sddr_benchmark, sr_benchmark = mkt.calculate_stats(benchmark_portvals)
        cr_strategy, adr_strategy, sddr_strategy, sr_strategy = mkt.calculate_stats(strategy_portvals)
        #
        print("----------------------------")
        print("---- Strategy Learner: ", symbol, "----")
        print("----------------------------")
        print(f"Cumulative return of benchmark: {cr_benchmark}")  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        print(f"Average Daily Return of benchmark: {adr_benchmark}")  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        print(f"Standard deviation of Daily return of benchmark: {sddr_benchmark}")
        print(f"Sharpe ratio of benchmark: {sr_benchmark}")
        print("----------------------------")
        print(f"Cumulative return of strategy: {cr_strategy}")  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        print(f"Average Daily Return of strategy: {adr_strategy}")  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        print(f"Standard deviation of Daily return of strategy: {sddr_strategy}")
        print(f"Sharpe ratio of strategy: {sr_strategy}")
        print("----------------------------")

    def author(self):
        return 'tbui61'

if __name__=="__main__":  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    np.random.seed(90347)
    sl = StrategyLearner(verbose=False, impact=0.005, commission=9.95)
    sl.addEvidence(symbol='JPM', sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 10000)
    trades = sl.testPolicy(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 10000)
    sl.evaluateTrades(symbol='JPM', df_trades=trades, sd='2008-01-01', ed='2009-12-31', sv=100000)
    trades_2 = sl.testPolicy(symbol = "JPM", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv = 10000)
    sl.evaluateTrades(symbol='JPM', df_trades=trades_2, sd='2010-01-01', ed='2011-12-31', sv=100000)  


