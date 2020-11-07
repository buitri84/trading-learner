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

def author():
    return 'tbui61'

def main():
     # seed random once. Always use same number
    np.random.seed(90347)
    #
    impact = [i for i in np.arange(0.001, 0.011, 0.001)]
    commission = 9.95
    symbol = 'JPM'
    sd = dt.datetime(2008,1,1)
    ed = dt.datetime(2009,12,31)
    sv = 100000
    cr = []
    ntrades = []
    strategy = sl.StrategyLearner()
    for i in impact:
        strategy.set_impact(i)
        strategy.addEvidence(symbol, sd, ed, sv)
        trades = strategy.testPolicy(symbol, sd, ed, sv)
        strategy_portvals = mkt.compute_portvals(trades, start_val=sv, impact=i, commission=commission)
        cr_strategy, adr_strategy, sddr_strategy, sr_strategy = mkt.calculate_stats(strategy_portvals)
        cr.append(cr_strategy)
        ntrades.append((trades!=0).sum()[0])
    # Graph result
    fig, ax = plt.subplots()
    ax.set_title('Cumulative return and Number of trades vs. impact')
    ax.set_xlabel('Impact')
    ax.set_ylabel('Cumulative return')
    ax.plot(impact, cr, label='Cumulative return', marker='+', color='b')
    ax2 = ax.twinx()
    ax2.set_ylabel('Number of trades')
    ax.legend()
    ax2.plot(impact, ntrades, label='No. of trades', marker='o', color='r')
    ax2.legend(loc='lower left')
    plt.savefig('experiment2.png')

if __name__=="__main__":
    main()