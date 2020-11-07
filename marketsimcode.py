import pandas as pd  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import numpy as np  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import datetime as dt  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import util		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
def author():
    return 'tbui61'

# Function from marketsim. Modified since there is no need to read an order file. Receive trades df with just
# dates and trades already
def compute_portvals(trades, start_val = 100000, commission=0.00, impact=0.000):		  	   		     			  		 			 	 	 		 		 	 		 		 	 
    start_date = trades.index[0]
    end_date = trades.index[-1]
    #1. get adjusted close price, from start to end date
    dates = pd.date_range(start_date, end_date)
    prices = util.get_data([trades.columns[0]], dates)
    #prices = util.get_data([symbol], dates)
    prices = prices.drop(['SPY'], axis=1)  # drop uneeded columns
    prices['Cash'] = ([1.0]*prices.count(axis=0)[0])        # add a last column of Cash prices = 1
    #2. Build the cash column of trades df.
    # impact_df represents the multiplier with impact
    impact_df = pd.Series(data=1.0, index=trades.index)
    f = lambda x: x > 0
    impact_df[f(trades[trades.columns[0]])] += impact
    f = lambda x: x < 0
    impact_df[f(trades[trades.columns[0]])] -= impact 
    trades['Cash'] = trades.iloc[:,0] * prices.iloc[:,0] * impact_df * -1
    # then subtract commission where there is a trade as well
    f = lambda x: x != 0
    trades['Cash'][f(trades['Cash'])] -= commission
    #3. build the holdings dataframe (based on trades and prices). IMPORTANT: Holdings means MARK TO MARKET at end of each day
    cumulative_trades = trades.cumsum(axis=0)
    # Warning if trading above limit
    #f = lambda x: abs(x) > 2000
    #print(cumulative_trades[f(cumulative_trades[trades.columns[0]])])
    #
    holdings = cumulative_trades * prices
    holdings['Cash'] += start_val               # cumsum of daily_holding assumed we start with 0 cash. So just add start_val to Cash
    #4. portfolio value at end of each day
    portvals = holdings.sum(axis=1)
    return portvals

def calculate_stats(portvals):
    cr = portvals.iloc[-1]/portvals.iloc[0] - 1
    dr = portvals.copy()
    dr[1:] = (portvals[1:] / portvals[:-1].values) - 1
    dr = dr[1:]
    adr = dr.mean()
    sddr = dr.std()
    sr = np.sqrt(252) * adr / sddr
    return cr, adr, sddr, sr
