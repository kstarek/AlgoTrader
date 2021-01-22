
from Account import *
from xgbHyperParameterOptimization import trainBestXGBmodel, importantFeatures, predictReturn
from TraderFunctions import *
import xgboost as xgb
from datetime import date
from random import sample

# valid intervals for getCandles = 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
# limit for # of candlesticks  for getCandles = 1500
ticker = 'BTCUSDT'
interval = '30m'
btcHrCandles = getNumOfFutCandles(ticker, interval, 1500)
#btcHrCandles = getFutCandles('BTCUSDT', '1h', dateToStamp("2019-09-30"), dateToStamp("2019-10-30"))
btcHrCandles_df = futCandlestoDfFloat(btcHrCandles)
btcHrCandles_df = addRSItoDf(btcHrCandles_df, 6)
btcHrCandles_df = addRSItoDf(btcHrCandles_df, 14)
btcHrCandles_df = addSMAandStdDevtoDf(btcHrCandles_df, 7)
btcHrCandles_df = addSMAandStdDevtoDf(btcHrCandles_df, 20)
btcHrCandles_df = addMACDtoDf(btcHrCandles_df, 12, 26, 9)
btcHrCandles_df = addMACDtoDf(btcHrCandles_df, 5, 13, 1)
btcHrCandles_df = addBolBandstoDf(btcHrCandles_df, 20, 2)
btcHrCandles_df = addBolBandstoDf(btcHrCandles_df, 10, 2.5)
btcHrCandles_df['Return'] = (btcHrCandles_df['close'] - btcHrCandles_df['open'] ) /btcHrCandles_df['open']
btcHrCandles_df.to_csv('btcHrCandles_df.csv', mode='a', header=False)

# preprocessing feature values for gradient boosting Machine Learning Approach

close = btcHrCandles_df
closeScaled = scale(btcHrCandles_df, (0 ,1))
X = closeScaled.iloc[: ,:-1]
y = close.iloc[: ,-1]

#shift by 1 as we are trying to predict the return of the next period
y = y.shift(periods=1)
y = y.fillna(0)
trainBestXGBmodel(X, y) #also saves best machine learning approach to best.model file in cwd, we have already done this, commented out as it is computationally expensive

importantFeatures(X, y)
xgbPredictions = predictReturn(xgb.DMatrix(X, label=y))

naiveAccuracy = naiveModel(y)
xgbAccuracy = predictionAccuracy(y, xgbPredictions)


print(f'Naive (random) Model Accuracy: {naiveAccuracy*100}%')
print(f'XGB Model Accuracy: {xgbAccuracy*100}%')
