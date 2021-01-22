
from Account import *
from xgbHyperParameterOptimization import *
from TraderFunctions import *
import xgboost as xgb
import matplotlib.pyplot as plt
import sklearn
import array
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# valid intervals for getCandles = 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
# limit for # of candlesticks  for getCandles = 1500
btcHrCandles = getFutCandles('BTCUSDT', '1h', 1000)
btcHrCandles_df = futCandlestoDf(btcHrCandles)
btcHrCandles_df = addRSItoDf(btcHrCandles_df, 6)
btcHrCandles_df = addSMAandStdDevtoDf(btcHrCandles_df, 7)
btcHrCandles_df = addSMAandStdDevtoDf(btcHrCandles_df, 20)
btcHrCandles_df = addMACDtoDf(btcHrCandles_df, 5, 13, 1)
btcHrCandles_df = addBolBandstoDf(btcHrCandles_df, 20, 2)
btcHrCandles_df['Return'] = (btcHrCandles_df['close' ] -btcHrCandles_df['open'] ) /btcHrCandles_df['open']

# preprocessing feature values for gradient boosting Machine Learning Approach

close = btcHrCandles_df
y = close.iloc[: ,-1]
closeScaled = scale(btcHrCandles_df, (0 ,1))
X = closeScaled.iloc[: ,:-1]


#trainBestXGBmodel(X, y) #also saves best machine learning approach to best.model file in cwd, we have already done this, commented out as it is computationally expensive


importantFeatures(X, y)
x = predictReturn(xgb.DMatrix(X, label=y))
print(x)