from Account import *
from datetime import datetime
from binance.client import Client
from binance.websockets import BinanceSocketManager  # allows us to automatically stream data as prices get updates
from twisted.internet import reactor
import pandas as pd
import btalib
import numpy as np
import financepy
import sklearn

client = Account.client


def terminateSocket(socket):
    reactor.stop()


# get time of first Bitcoin-Tether timestamp
timestamp = Account.client._get_earliest_valid_timestamp('BTCUSDT', '1d')


# server time is calculated in milliseconds, covert to date/time using this function
def stampToDate(time):
    return datetime.fromtimestamp(time / 1000)


# get candlesticks for selected pair and selected time
def getFutCandles(tradingPair, interval, numofCandles):
    candles = Account.client.futures_klines(symbol=tradingPair, interval=interval, limit=numofCandles)
    return candles


# write candles to csv
def candlesToCSV(candles):
    with open('btc_candles.csv', 'w') as csv:
        for line in candles:
            csv.write(f'{line[0]}, {line[1]}, {line[2]}, {line[3]}, {line[4]}, {line[5]}, {line[6]}\n')


# write candles to dataframe
def futCandlestoDf(candles):
    x = len((candles[0]))
    for i in range(x-1, 5, -1):
        for j in candles:
            j.pop(i)

    candles_df = pd.DataFrame.from_records(candles, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    candles_df.set_index('date', inplace=True)
    for x in candles_df.columns:
        candles_df[x] = candles_df[x].astype(float)
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
    a = (n-1) / n
    ak = a**np.arange(len(x)-1, -1, -1)
    return np.r_[np.full(n, np.nan), y0, np.cumsum(ak * x) / ak / n + y0 * a**np.arange(1, len(x)+1)]


def addRSItoDf(df, period):
    df['change'] = df['close'].diff()
    df['gain'] = df.change.mask(df.change < 0, 0.0)
    df['loss'] = -df.change.mask(df.change > 0, -0.0)
    df['avg_gain'] = rma(df.gain[period + 1:].to_numpy(), period, np.nansum(df.gain.to_numpy()[:period + 1]) / period)
    df['avg_loss'] = rma(df.loss[period + 1:].to_numpy(), period, np.nansum(df.loss.to_numpy()[:period + 1]) / period)
    df['rs'] = df.avg_gain / df.avg_loss
    df['rsi'] = 100 - (100 / (1 + df.rs))
    df.drop(columns=['change', 'gain', 'loss', 'avg_gain', 'avg_loss', 'rs'], axis=1, inplace=True)
    df = df.iloc[period:]
    return df


def addMACDtoDf(df, fast, slow, signal):
    close = df['close']
    fast = close.ewm(span=fast, adjust=False).mean()
    slow = close.ewm(span=slow, adjust=False).mean()
    macd = fast-slow
    signal = macd.ewm(span=9, adjust=False).mean()
    df['macd'] = macd

    return df

def addBolBandstoDf(df, ma, sd):
    if (f'pandas_SMA_{ma}') in df.columns:
        df['UpperBand'] = df[(f'pandas_SMA_{ma}')] + (df[(f'pandas_StdDev_{ma}')] * sd)
        df['LowerBand'] = df[(f'pandas_SMA_{ma}')] - (df[(f'pandas_StdDev_{ma}')] * sd)
        df['outsideUpperBand'] = np.where(df['close'] > df['UpperBand'], 1, 0)
        df['outsideLowerBand'] = np.where(df['close'] < df['LowerBand'], 1, 0)
        return df
    else:
        df = addSMAandStdDevtoDf(df, ma)
        df['UpperBand'] = df[(f'pandas_SMA_{ma}')] + (df[(f'pandas_StdDev_{ma}')] * sd)
        df['LowerBand'] = df[(f'pandas_SMA_{ma}')] - (df[(f'pandas_StdDev_{ma}')] * sd)
        df['outsideUpperBand'] = np.where(df['close'] > df['UpperBand'], 1, 0)
        df['outsideLowerBand'] = np.where(df['close'] < df['LowerBand'], 1, 0)
        return df

def futuresLimitBuyOrder(symbol, side, timeInForce, quantity, price):
    order = client.futures_create_order(symbol=symbol, side=side, type='LIMIT',
                                        timeInForce=timeInForce, quantity=quantity, price=price)
    return order

def scale(dataframe, scale=(0,1)):
    columns = dataframe.columns
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.feature_range = scale
    return pd.DataFrame(scaler.fit_transform(dataframe), columns=columns).dropna()

def futuresMarketBuy(symbol, side, quantity):
    order = client.futures_create_order(symbol=symbol, side=side, type='MARKET', quantity=quantity)
    return order





