from binance.client import Client
import os


class Account:
    # api_key and secret are declared as environment vars set in cmd for security purposes

    api_key = os.environ.get('binance_api')
    api_secret = os.environ.get('binance_secret')

    client = Client(api_key, api_secret)
