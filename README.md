A Random Forest Decision Tree learner that generates BUY and SELL decision (US equities).

1. ManualStrategy.py:
- Contains the manual trading strategy as class ManualStrategy. The main() method contains the code to run the manual strategy
and compare with benchmark, both in-sample and out-of-samle, for the symbol JPM. 
- To run:
PYTHONPATH=../:. python ManualStrategy.py
- Will generate: ManualStrategy_in.png, ManualStrategy_out.png (required), and an extra Trades_in.png

2. StrategyLearner.py:
- Contains the Random Forest Decision Tree strategy as class StrategyLearner.
- The main() method contains code to train a tree, then test both in-sample and out-of-sample data for 
symbol JPM and compare with benchmark. Can be run alone FYI:
PYTHONPATH=../:. python StrategyLearner.py
- If main() method is run, will generate Learner_trades.png (showing price and associated trades for out-of-sample)

3. experiment1.py:
- Contains the code to perform experiment 1 and generate required charts. This compares in-sample performance between Manual Strategy
and the Strategy Learner. To run:
PYTHONPATH=../:. python experiment1.py
- Will generate experiment1.png

4. experiment2.py:
- Contains the code to perform experiment 2 and generate required charts. THis experiments with changing the impact parameter
ie. to reflect the fact that trades are not free. To run:
PYTHONPATH=../:. python experiment2.py
- Will generate experiment2.png

5. testproject.py:
- In 1 go, will perform the ManualStrategy experiment, experiment 1 and experiment 2 above. To run:
PYTHONPATH=../:. python testproject.py
- Will generate the same charts as listed in bullet point 1, 3, and 4 above.

6. indicators.py, marketsimcode.py:
- Used to get the value of technical indicators (not trade signal) and evaluate performance.
Should be put in the same directory and not run separately.

7. BagLearner.py, RTLearner.py:
- Core code for the Random Forest leaner. Should be put in the same directory and not run separately.
