def crypto_data(start="1 Jan, 2015", end="1 Mar, 2025", coins=["BTCUSDT", "ETHUSDT"]):
    """
    Gets hourly k-line information for specified coins, and does preprocessing to return a time series dataframe.
    """

    from binance.client import Client
    import pandas as pd
    import numpy as np

    def classify(change, threshold=0.005):
        if change <= -threshold:  # Decrease
            return 0
        elif change <= -threshold / 5:  # Slight decrease
            return 1
        elif change < threshold / 5:  # Stable
            return 2
        elif change < threshold:  # Slight increase
            return 3
        else:  # Increase
            return 4

    client = Client()
    dfs = []

    for coin in coins:
        klines = client.get_historical_klines(
            symbol=coin,
            interval=Client.KLINE_INTERVAL_1HOUR,
            start_str=start,
            end_str=end
        )
    
        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume", 
            "close_time", "quote_asset_volume", "num_trades", 
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["symbol"] = coin
        
        df = df[["timestamp", "symbol", "open", "high", "low", "close", "volume", "num_trades"]]
    
        df[["open", "high", "low", "close", "volume", "num_trades"]] = df[["open", "high", "low", "close", "volume", "num_trades"]].astype(float)
    
        df["high"] = df["high"] / df["open"] - 1  # Since the scale of the prices changes over time, I change the values to percentages with respect to the open price.
        df["low"] = df["low"] / df["open"] - 1
        df["close"] = df["close"] / df["open"] - 1
    
        df = df.drop(columns=["open"])
    
        df["price_change"] = df["close"].shift(-1)  # I use the close price of the next time index as my label.
        df["price_change"] = df["price_change"].apply(lambda x: classify(x))  # Instead of predicting the price or change directly, I predict the direction of movement.
        df = df.dropna()
            
        dfs.append(df)
    
    df = pd.concat(dfs).pivot(index="timestamp", columns="symbol")  # Joining the different coins into a single dataframe.
    df = df.drop(columns=[col for col in df.columns if "price_change" in col and "BTCUSDT" not in col])  # Removing all label columns except the target coin.
    df.columns = ["_".join(col).strip() for col in df.columns]
    df = df.reset_index()
    
    fill_values = {}
    for col in df.columns:
        if "price_change" in col:
            fill_values[col] = 2  # Neutral category
        elif "volume" in col or "num_trades" in col:
            fill_values[col] = 0  # No volume = 0
        else:
            fill_values[col] = 0  # Default fill for all other features
    
    df.fillna(fill_values, inplace=True)
    
    df["day_of_week"] = df["timestamp"].dt.dayofweek  # 0 = Monday, 6 = Sunday
    df["hour_of_day"] = df["timestamp"].dt.hour       # 0 - 23
    
    df = pd.get_dummies(df, columns=["day_of_week", "hour_of_day"])

    return df