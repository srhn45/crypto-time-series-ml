import tensorflow as tf
from search import guided_hyperparameter_search
from get_data import crypto_data

df = crypto_data(coins = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"])
best_params, best_acc = guided_hyperparameter_search(df)


