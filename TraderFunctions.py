from sklearn.metrics import accuracy_score
from Account import *
from datetime import datetime
from binance.client import Client
from sklearn.metrics import mean_squared_error
from binance.websockets import BinanceSocketManager  # allows us to automatically stream data as prices get updates
from twisted.internet import reactor
import pandas as pd
import numpy as np
import sklearn
import ciso8601
import time
import matplotlib as plt
import ta
client = Account.client


def terminateSocket(socket):
    reactor.stop()

#convert Date/Time to timestamp in milliseconds
def dateToStamp(date):
    timestamp = ciso8601.parse_datetime(date)
    time.mktime(timestamp.timetuple())

# server time is calculated in milliseconds, covert to date/time using this function
def stampToDate(time):
    return datetime.fromtimestamp(time / 1000)

# get number candlesticks for selected pair and selected time
def getNumOfFutCandles(tradingPair, interval, numofCandles):
    candles = Account.client.futures_klines(symbol=tradingPair, interval=interval, limit=numofCandles)
    return candles

def getFutCandles(tradingPair, interval, startTime, endTime):
    candles = Account.client.futures_klines(symbol=tradingPair, interval=interval, startTime=startTime, endTime=endTime)
    return candles

# write candles to csv
def candlesToCSV(candles):
    with open('btc_candles.csv', 'w') as csv:
        for line in candles:
            csv.write(f'{line[0]}, {line[1]}, {line[2]}, {line[3]}, {line[4]}, {line[5]}, {line[6]}\n')

# write candles to dataframe
def futCandlestoDfFloat(candles):
    x = len((candles[0]))
    for i in range(x - 1, 5, -1):
        for j in candles:
            j.pop(i)

    candles_df = pd.DataFrame.from_records(candles, columns=['date', 'open', 'high', 'low', 'close', 'volume'])

    candles_df.set_index('date', inplace=True)
    for x in candles_df.columns:
        candles_df[x] = candles_df[x].astype(float)
    return candles_df

def futCandlestoDf(candles):
    x = len((candles[0]))
    for i in range(x - 1, 5, -1):
        for j in candles:
            j.pop(i)

    candles_df = pd.DataFrame.from_records(candles, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    for x in candles_df[0]:
        x = stampToDate(x)
    candles_df.set_index('date', inplace=True)

    return candles_df
def getOpenOrders(self):
    return Account.client.get_open_orders()


def getOpenFutOrders(self):
    return Account.client.futures_get_open_orders()


def addSMAandStdDevtoDf(df, period):
    df[f'pandas_SMA_{period}'] = df.iloc[:, 1].rolling(window=period).mean()
    df[f'pandas_StdDev_{period}'] = df.iloc[:, 1].rolling(window=period).std()
    df = df.iloc[period:]
    return df

def rma(x, n, y0):
    a = (n - 1) / n
    ak = a ** np.arange(len(x) - 1, -1, -1)
    return np.r_[np.full(n, np.nan), y0, np.cumsum(ak * x) / ak / n + y0 * a ** np.arange(1, len(x) + 1)]

def addAllTa(df):
    df = ta.add_momentum_ta(df, 'open', 'high', 'low', 'close', 'volume', True)
    return df

def addRSItoDf(df, period):
    df['change'] = df['close'].diff()
    df['gain'] = df.change.mask(df.change < 0, 0.0)
    df['loss'] = -df.change.mask(df.change > 0, -0.0)
    df['avg_gain'] = rma(df.gain[period + 1:].to_numpy(), period, np.nansum(df.gain.to_numpy()[:period + 1]) / period)
    df['avg_loss'] = rma(df.loss[period + 1:].to_numpy(), period, np.nansum(df.loss.to_numpy()[:period + 1]) / period)
    df['rs'] = df.avg_gain / df.avg_loss
    df[f'rsi{period}'] = 100 - (100 / (1 + df.rs))
    df.drop(columns=['change', 'gain', 'loss', 'avg_gain', 'avg_loss', 'rs'], axis=1, inplace=True)
    df = df.iloc[period:]
    return df


def addMACDtoDf(df, fast, slow, signal):
    close = df['close']
    fastCol = close.ewm(span=fast, adjust=False).mean()
    slowCol = close.ewm(span=slow, adjust=False).mean()
    macd = fastCol - slowCol
    sig = macd.ewm(span=9, adjust=False).mean()
    df[f'macd{fast},{slow},{signal}'] = macd

    return df


def Wilder(data, periods):
    start = np.where(~np.isnan(data))[0][0] #Check if nans present in beginning
    Wilder = np.array([np.nan]*len(data))
    Wilder[start+periods-1] = data[start:(start+periods)].mean() #Simple Moving Average
    for i in range(start+periods,len(data)):
        Wilder[i] = (Wilder[i-1]*(periods-1) + data[i])/periods #Wilder Smoothing
    return(Wilder)

def addBolBandstoDf(df, ma, sd):
    if (f'pandas_SMA_{ma}') in df.columns:
        df[f'UpperBand{ma},{sd}'] = df[(f'pandas_SMA_{ma}')] + (df[(f'pandas_StdDev_{ma}')] * sd)
        df[f'LowerBand{ma},{sd}'] = df[(f'pandas_SMA_{ma}')] - (df[(f'pandas_StdDev_{ma}')] * sd)
        return df
    else:
        df = addSMAandStdDevtoDf(df, ma)
        df[f'UpperBand{ma},{sd}'] = df[(f'pandas_SMA_{ma}')] + (df[(f'pandas_StdDev_{ma}')] * sd)
        df[f'LowerBand{ma},{sd}'] = df[(f'pandas_SMA_{ma}')] - (df[(f'pandas_StdDev_{ma}')] * sd)
        return df


def futuresLimitBuyOrder(symbol, side, timeInForce, quantity, price):
    order = client.futures_create_order(symbol=symbol, side=side, type='LIMIT',
                                        timeInForce=timeInForce, quantity=quantity, price=price)
    return order


def scale(dataframe, scale=(0, 1)):
    columns = dataframe.columns
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.feature_range = scale
    return pd.DataFrame(scaler.fit_transform(dataframe), columns=columns).dropna()


def futuresMarketBuy(symbol, side, quantity):
    order = client.futures_create_order(symbol=symbol, side=side, type='MARKET', quantity=quantity)
    return order


binary = lambda data: [1 if x > 0 else 0 for x in data]


#naive approach (random generated) for gauging accuracy
def naiveModel(tradingPair):
    naivePredictions = np.random.randint(0, 2, len(tradingPair))
    accuracy = accuracy_score(binary(tradingPair), naivePredictions)
    return accuracy


def predictionAccuracy(test, predictions):
    accuracy = accuracy_score(
        binary(test),
        binary(predictions)
    )
    return accuracy

def evaluation(X, y, model, n_preds=10, random=True, show_graph=True):
    n_steps = X.shape[1]
    max_random_int = len(y) - n_steps
    y_true, y_pred, prediction_accuracy, slices = [], [], [], []

    for i in range(n_preds):

        if random == True:
            position = np.random.randint(0, max_random_int)
        else:
            position = i

        y_hat = model.predict(X[position:position + 1])[0][0]
        y_pred.append(y_hat)
        y_true.append(y[position])
        y_current = y[position]

        # If we predit return, c = 0, else c = previous sequence position
        if y.min() < 0:
            c = 0
        else:
            c = y[position - 1]

        if ((y_hat > c) & (y_current > c)) or ((y_hat < c) & (y_current < c)):
            acc = 1
        else:
            acc = 0

        prediction_accuracy.append(acc)
        slices.append((list(y[position - n_steps:position + 1]), list(y[position - n_steps:position]) + [y_hat], acc))

    if show_graph == True:
        plt.rcParams['figure.dpi'] = 227
        plt.style.use('seaborn-whitegrid')
        plt.figure(figsize=(16, 6))
        plt.bar(range(n_preds), y_true[:], width=.7, alpha=.6, color="#4ac2fb", label="True")
        plt.bar(range(n_preds), y_pred[:], width=.7, alpha=.6, color="#ff4e97", label="Predicted")
        plt.axhline(0, color="#333333", lw=.8)
        plt.legend(loc=1)
        plt.title('Daily Return Prediction', fontSize=15)
        plt.show()

    print('MSE:', mean_squared_error(y_true, y_pred))
    print('Accuracy: {}%'.format(round((sum(prediction_accuracy) / len(prediction_accuracy)) * 100), 2))
    return slices, np.array(y_true), np.array(y_pred)
