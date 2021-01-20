from binance.client import Client
import os


class Account:
    # api_key and secret are declared as environment vars set in cmd for security purposes

    api_key = os.environ.get('binance_api')
    api_secret = os.environ.get('binance_secret')

    api_key = 'xbu0sYoMn227tJyEbJP956WwBi8eoeq4eZHs9UnMTq5LiBoRH66hR4Vy8RyRZhH5'  # temp solution delete after
    api_secret = '2H9efSSmBsWv6UssTzgznD42hQBlhMAg5tLhcOhtVuZawxBra94Hcmff49ITDP52'

    client = Client(api_key, api_secret)
