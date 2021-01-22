
from time import sleep
import btalib
# Project Specific Class imports
from Account import *
from TraderFunctions import *

# get latest Bitcoin-Tether price, we declare a dictionary as it has O(1) access
# it will update from the streamed data prices, containing fields: last and bid

prices = {'BTCUSDT': pd.DataFrame(columns=['date', 'bid', 'ask', 'price']), 'error': False}


def futures_btc_trade_history(msg):
    # define how to process incoming WebSocket messages
    if msg['data']['e'] != 'error':
        prices['BTCUSDT'].loc[len(prices['BTCUSDT'])] = [pd.Timestamp.now(), float(msg['data']['b']), float(msg['data']['a']), (float(msg['data']['b'])+float(msg['data']['a']))/2]
    else:
        prices['error']: True


# init and start the WebSocket
def streamLiveValues(pairToTrade):
    socketManager = BinanceSocketManager(Account.client)
    tradingPair = socketManager.start_symbol_ticker_futures_socket('BTCUSDT', pairToTrade)
    socketManager.start()

streamLiveValues(futures_btc_trade_history)

# wait until we have data so no odd behaviors occur
while len(prices['BTCUSDT']) == 0:
    sleep(0.1)
sleep(300)

while True:
    df = prices['BTCUSDT']
    #start_time = df.date.iloc[-1] - pd.Timedelta(minutes=2)
    #df = df.loc[df.date >= start_time]
    max_price = df.bid.max()
    min_price = df.bid.min()
    # unfinished entry/exit point logic





