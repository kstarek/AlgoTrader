import time
import logging
import numpy as np
import binance
from binance.client import Client
from binance.websockets import BinanceSocketManager
from twisted.internet import reactor
import os
from binance.exceptions import BinanceAPIException, BinanceOrderException

#Project Specific Class imports
from Account import *
from TraderFunctions import *

buy = Account.client.futures_create_order(
    symbol= 'BTCUSDT',
    side='BUY',
    type='LIMIT',
    timeInForce='GTC',
    quantity=100,
    price=200)








