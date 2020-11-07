import pandas as pd
import numpy as np
import util
import matplotlib.pyplot as plt
import datetime as dt
import indicators as ind
import marketsimcode as mkt

class ManualStrategy(object):
    
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

    # The composite signal rule is simply: if df1 AND (df2 OR df3) ie: df1 is the base (momentum) criteria
    # If that one is satisfied, any signal from one of those others are ok to trigger a trade.
    def get_composite_signal(self, df1, df2, df3):
        buy = df1[(df1==1) & ((df2==1) | (df3==1))]
        sell = df1[(df1==-1) & ((df2==-1) | (df3==-1))]
        buy.fillna(value=0, inplace=True)
        sell.fillna(value=0, inplace=True)
        df = buy + sell
        return df

    # trade when there is crossover of Stochastic and threshold line. Upward cross of lower = BUY, downward cross of upper = SELL 
    def get_stochastic_signal(self, df, lower=0.2, upper=0.8):
        # 1. Generate BUY signal
        df_diff = df - lower
        sig_buy = df_diff * df_diff.shift(periods=1)
        # now sig_buy is -ve when there is crossover. Keep only the -ve, turn the rest to 0
        f = lambda x: x >= 0
        sig_buy[f(sig_buy)] = 0
        # multiply by diff df to get -ve when there is upwards cross, +ve when there is downwards cross. Keep -ve only
        sig_buy = sig_buy * df_diff
        f = lambda x: x >= 0
        sig_buy[f(sig_buy)] = 0
        # Turn all -ve into BUY signal
        f = lambda x: x < 0
        sig_buy[f(sig_buy)] = 1
        # 2. Now generate SELL signal. Similar to BUY but flip the signs
        df_diff = df - upper
        sig_sell = df_diff * df_diff.shift(periods=1)
        f = lambda x: x >= 0
        sig_sell[f(sig_sell)] = 0
        sig_sell = sig_sell * df_diff
        f = lambda x: x <= 0
        sig_sell[f(sig_sell)] = 0
        f = lambda x: x > 0
        sig_sell[f(sig_sell)] = -1
        # Add them to get final Buy and Sell signal
        sig_stochastic = sig_buy + sig_sell
        # Cosmetic
        sig_stochastic.fillna(value=0, inplace=True)
        return sig_stochastic

    # trade when B% passes threshold (buy when it crosses below 0, sell when it crosses above 1)
    def get_BB_signal(self, df_price, df_SMA, df_Bollinger, lower=0, upper=1):
        # get the B% values
        sig_BB = (df_price-(df_SMA-df_Bollinger))\
            /(2*df_Bollinger)
        f = lambda x: x < lower
        sig_BB[f(sig_BB)] = -2
        f = lambda x: x > upper
        sig_BB[f(sig_BB)] = 2
        f = lambda x: abs(x) < 2
        sig_BB[f(sig_BB)] = 0
        sig_BB = sig_BB / 2 * -1
        # 
        sig_BB.fillna(value=0, inplace=True)
        return sig_BB

    # Trade when ROC changes sign (-ve to +ve = BUY, +ve to -ve = SELL)
    def get_ROC_signal(self, symbol, df_final):
        sig_ROC = df_final['ROC'] * df_final['ROC'].shift(periods=1)
        f = lambda x: x < 0
        sig_ROC[f(sig_ROC)] = -1
        f = lambda x: x > 0
        sig_ROC[f(sig_ROC)] = 0
        sig_ROC = 1 * sig_ROC * df_final['ROC']
        f = lambda x: x < 0
        sig_ROC[f(sig_ROC)] = -1
        f = lambda x: x > 0
        sig_ROC[f(sig_ROC)] = 1
        #sig_ROC = sig_ROC * 1000
        sig_ROC.fillna(value=0, inplace=True)
        #sig_ROC.loc[sig_ROC[sig_ROC!=0].index[0]] /= 2
        # Cosmetic cleanup. For some reason the df keeps insisting on being a Series instead
        sig_ROC=sig_ROC.to_frame()
        sig_ROC.rename(columns={'ROC':symbol}, inplace=True)
        return sig_ROC

    # We are using all other signals to trade against momentum. So allow BUY when momentum is +ve and SELL when momentum is -ve
    def get_momentum(self, df_ROC, threshold=0):
        #sig_momentum = df_ROC.copy()
        f = lambda x: x < threshold
        df_ROC[f(df_ROC)] = -1
        f = lambda x: x > threshold
        df_ROC[f(df_ROC)] = 1
        return df_ROC

    # Trade when the MACD line cross the center line (-ve to +ve = BUY, +ve to -ve = SELL)
    def get_MACD_signal(self, df):
        sig_MACD = df * df.shift(periods=1)
        f = lambda x: x < 0
        sig_MACD[f(sig_MACD)] = -1
        f = lambda x: x > 0
        sig_MACD[f(sig_MACD)] = 0
        sig_MACD = -1 * sig_MACD * df
        f = lambda x: x < 0
        sig_MACD[f(sig_MACD)] = -1
        f = lambda x: x > 0
        sig_MACD[f(sig_MACD)] = 1
        sig_MACD.fillna(value=0, inplace=True)
        #sig_MACD.loc[sig_MACD[sig_MACD!=0].index[0]] /= 2
        # Cosmetic cleanup. For some reason the df keeps insisting on being a Series instead
        #sig_MACD=sig_MACD.to_frame()
        #sig_MACD.rename(columns={'MACD':symbol}, inplace=True)
        return sig_MACD

    def testPolicy(self, symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
        # Prepare symbols and dates to retrieve data
        symbols = [symbol]
        dates = pd.date_range(sd, ed)
        data = util.get_data(symbols, dates, addSPY=False, colname='Adj Close')
        data.dropna(axis=0, how='all', inplace=True)
        # Get indicators
        SMA = ind.calculate_SMA(data, period=20)
        EMA = ind.calculate_EMA(data, period=20)
        ROC = ind.calculate_ROC(data, period=22)
        Bollinger = ind.calculate_Bollinger(data, period=16, dev=2)
        Stochastic = ind.calculate_Stochastic(data, lookback=20, smooth=3)
        MACD = ind.calculate_MACD(data)
        # Get BB% signal
        sig_BB = self.get_BB_signal(data, SMA, Bollinger, lower=0, upper=1)
        # Get stochastic signal
        sig_stochastic = self.get_stochastic_signal(Stochastic, lower=0.2, upper=0.8)
        # Get the momentum
        sig_momentum = self.get_momentum(ROC, threshold=0)
        # Get MACd signal
        sig_MACD = self.get_MACD_signal(MACD)
        #
        sig_composite = self.get_composite_signal(sig_momentum, sig_BB, sig_stochastic)
        df_trades = self.generate_trade_from_signal(sig_composite)
        return df_trades

    def author(self):
        return 'tbui61'

def testManualStrategy_In(symbol='JPM', sd='2008-01-01', ed='2009-12-31', sv=100000, commission=9.95, impact=0.005):
    ms = ManualStrategy()
    df_trades = ms.testPolicy(symbol, sd, ed, sv)
    # Prices for reference
    dates = pd.date_range(sd, ed)
    data = util.get_data([symbol], dates, addSPY=False, colname='Adj Close')
    data.dropna(axis=0, how='all', inplace=True)   
    # Benchmark: buy once and hold
    benchmark_trades = pd.DataFrame(data=0.0, index=data.index, columns=[symbol])
    benchmark_trades.iloc[0] = 1000
    benchmark_portvals = mkt.compute_portvals(benchmark_trades, sv, commission, impact)
    benchmark_portvals_normed = benchmark_portvals / benchmark_portvals.ix[0,:]
    # Manual trades
    manual_portvals = mkt.compute_portvals(df_trades, sv, commission, impact)
    manual_portvals_normed = manual_portvals / manual_portvals.ix[0,:]
    # Plot comparison and trades
    fig, ax = plt.subplots()
    ax.set_title('In-sample performance of Benchmark and Manual')
    ax.plot(benchmark_portvals_normed, color='g', label='Benchmark')
    ax.plot(manual_portvals_normed, label='Manual', color='r')
    for buy_point in df_trades.index[df_trades[symbol] >= 1000].tolist():
            ax.axvline(buy_point, color='b')
    for sell_point in df_trades.index[df_trades[symbol] <= -1000].tolist():
        ax.axvline(sell_point, color='k')
    ax.legend()
    plt.savefig('ManualStrategy_in.png')
    # Extra: plot trades and prices
    fig, ax2 = plt.subplots()
    ax2.set_title('Prices and executed trades')
    ax2.plot(data, label='Prices')
    for buy_point in df_trades.index[df_trades[symbol] >= 1000].tolist():
        ax2.axvline(buy_point, color='b')
    for sell_point in df_trades.index[df_trades[symbol] <= -1000].tolist():
        ax2.axvline(sell_point, color='k')
    ax2.legend()
    plt.savefig('Trades_in.png')
    # Calculate stats
    cr_benchmark, adr_benchmark, sddr_benchmark, sr_benchmark = mkt.calculate_stats(benchmark_portvals)
    cr_manual, adr_manual, sddr_manual, sr_manual = mkt.calculate_stats(manual_portvals)
    #
    print("----------------------------")
    print("---- In sample ----")
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

def testManualStrategy_Out(symbol='JPM', sd='2010-01-01', ed='2011-12-31', sv=100000, commission=9.95, impact=0.005):
    ms = ManualStrategy()
    df_trades = ms.testPolicy(symbol, sd, ed, sv)
    # Prices for reference
    dates = pd.date_range(sd, ed)
    data = util.get_data([symbol], dates, addSPY=False, colname='Adj Close')
    data.dropna(axis=0, how='all', inplace=True)   
    # Benchmark: buy once and hold
    benchmark_trades = pd.DataFrame(data=0.0, index=data.index, columns=[symbol])
    benchmark_trades.iloc[0] = 1000
    benchmark_portvals = mkt.compute_portvals(benchmark_trades, sv, commission, impact)
    benchmark_portvals_normed = benchmark_portvals / benchmark_portvals.ix[0,:]
    # Manual trades
    manual_portvals = mkt.compute_portvals(df_trades, sv, commission, impact)
    manual_portvals_normed = manual_portvals / manual_portvals.ix[0,:]
    # Plot comparison and trades
    fig, ax = plt.subplots()
    ax.set_title('Out-sample performance of Benchmark and Manual')
    ax.plot(benchmark_portvals_normed, color='g', label='Benchmark')
    ax.plot(manual_portvals_normed, label='Manual', color='r')
    for buy_point in df_trades.index[df_trades[symbol] >= 1000].tolist():
            ax.axvline(buy_point, color='b')
    for sell_point in df_trades.index[df_trades[symbol] <= -1000].tolist():
        ax.axvline(sell_point, color='k')
    ax.legend()
    plt.savefig('ManualStrategy_out.png')
    # Calculate stats
    cr_benchmark, adr_benchmark, sddr_benchmark, sr_benchmark = mkt.calculate_stats(benchmark_portvals)
    cr_manual, adr_manual, sddr_manual, sr_manual = mkt.calculate_stats(manual_portvals)
    #
    print("----------------------------")
    print("---- Out sample ----")
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

def main():
    testManualStrategy_In(symbol='JPM', sd='2008-01-01', ed='2009-12-31', sv=100000, commission=9.95, impact=0.005)    
    testManualStrategy_Out(symbol='JPM', sd='2010-01-01', ed='2011-12-31', sv=100000, commission=9.95, impact=0.005)

if __name__ == "__main__":
    main()

