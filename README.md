# AlgoTrader

Unfinished Algorithmic Trading bot for Binance using RESTful API calls. 

Added Machine Learning predictive model for candlestick % change over last candle using Gradient Boosting (XGB), optimized hyperparamters using cross validation. Created functionality to show feature importance. Visualized nicely using matplotlib @ importance.jpg .

To do: add entry and exit logic, backtest on real data, test effectiveness of model vs rng guesses. Need to do cross validation regarding different values for RSI, MACD, BolBands, Candlesticks, etc to minimize RMSE. Split Trader.py into two files using different methods of data generation. Look into potential of using Neural Networks instead of Boosting

Implemented manual calculation of Trading Indicators

Likely can manually generate live candlesticks using stream, will have to look into that.
