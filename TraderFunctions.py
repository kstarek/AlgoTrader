from Account import *
from datetime import datetime
from binance.client import Client
from binance.websockets import BinanceSocketManager  # allows us to automatically stream data as prices get updates
from twisted.internet import reactor
import pandas as pd
import btalib

class TraderFunctions:
    # get latest Bitcoin-Tether price, we declare a dictionary as it has O(1) access
    # it will update from the streamed data prices, containing fields: last and bid

    btcPrice = {'error': False}

    def btc_trade_history(msg, btcPrice=btcPrice):
        # define how to process incoming WebSocket messages
        if msg['e'] != 'error':
            btcPrice['last'] = msg['c']
            btcPrice['bid'] = msg['b']
            btcPrice['ask'] = msg['a']
            btcPrice['priceChange'] = msg['p']
        else:
            btcPrice['error'] = True

    # init and start the WebSocket
    socketManager = BinanceSocketManager(Account.client)
    tradingPair = socketManager.start_symbol_ticker_socket('BTCUSDT', btc_trade_history)
    socketManager.start()

    def terminateSocket(socket):
        reactor.stop()

    # get time of first Bitcoin-Tether timestamp
    timestamp = Account.client._get_earliest_valid_timestamp('BTCUSDT', '1d')

    # server time is calculated in milliseconds, covert to date/time using this function
    def stampToDate(time):
        return datetime.fromtimestamp(time / 1000)

    # get candlesticks for selected pair and selected time
    def getCandles(tradingPair, interval, numofCandles):
        candles = Account.client.get_historical_klines(tradingPair, interval, timestamp, limit=numofCandles)
        return candles

    # write candles to csv
    def candlesToCSV(candles):
        with open('btc_candles.csv', 'w') as csv:
            for line in candles:
                csv.write(f'{line[0]}, {line[1]}, {line[2]}, {line[3]}, {line[4]}, {line[5]}, {line[6]}\n')

    # write candles to dataframe
    def candlestoDataFrame(candles):
        candles_df = pd.DataFrame(candles, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        candles_df.set_index('date', inplace=True)
        return candles_df

    def getOpenOrders():
        return Account.client.get_open_orders()

    def getOpenFutOrders():
        return Account.client.futures_get_open_orders()

    def addRSItoDf(df, period):
        rsi = btalib.rsi(df, period=period)
        df = df.join([rsi.df])
        return df

    def addMACDtoDf(df, fast, slow, signal):
        macd = btalib.macd(df, pfast=fast, pslow=slow, psignal=signal)
        df = df.join([macd.df])
        return df


