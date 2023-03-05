import datetime as dt
import math
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tweepy
from matplotlib import style
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from textblob import TextBlob
import constants as ct

def get_stock_data(symbol, from_date, to_date):
    data = yf.download(symbol, start=from_date, end=to_date)
    df = pd.DataFrame(data=data)
    df = df[['Close']]
    df.rename(columns={"Close": "Price"}, inplace=True)
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = (df['Date'] - df['Date'].min())  / np.timedelta64(1,'D')
    return df

def train_model(df):
    X = df[['Date']] # Features
    y = df[['Price']] # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    maxdate = df['Date'].iloc[-1]
    future_dates = []
    for i in range(300):
        future_dates.append(maxdate + i + 1)
    future_dates = np.array(future_dates)
    future_stock_prices = model.predict(future_dates.reshape(-1, 1))
    fut_stk_price = []
    for future_price in future_stock_prices:
        fut_stk_price.append(future_price[0])
    d = {'Date': future_dates, 'Price': fut_stk_price}
    dataf = pd.DataFrame(data = d)
    return dataf

def train_model2(df):
    X = df[['Date']] # Features
    y = df[['Price']] # Target variable
    model = GradientBoostingRegressor(n_estimators = 500, learning_rate = 0.1, max_depth = 100, random_state = 42)
    model.fit(X, y)
    maxdate = df['Date'].iloc[-1]
    future_dates = []
    for i in range(300):
        future_dates.append(maxdate + i + 1)
    future_dates = np.array(future_dates)
    future_stock_prices = model.predict(future_dates.reshape(-1, 1))
    fut_stk_price = []
    for future_price in future_stock_prices:
        fut_stk_price.append(future_price)
    d = {'Date': future_dates, 'Price': fut_stk_price}
    dataf = pd.DataFrame(data = d)
    return dataf

def train_model3(df):
    X = df[['Date']].to_numpy() # Features
    y = df[['Price']].to_numpy() # Target variable
    poly = PolynomialFeatures(degree = 3, include_bias = False)
    X_poly = poly.fit_transform(X.reshape(-1, 1))
    model = LinearRegression()
    model = model.fit(X_poly, y)
    maxdate = df['Date'].iloc[-1]
    future_dates = []
    for i in range(30):
        future_dates.append(maxdate + i + 1)
    future_dates = np.array(future_dates)
    newdata = poly.fit_transform(future_dates.reshape(-1, 1))
    future_stock_prices = model.predict(newdata)
    fut_stk_price = []
    for future_price in future_stock_prices:
        fut_stk_price.append(future_price[0])
    d = {'Date': future_dates, 'Price': fut_stk_price}
    dataf = pd.DataFrame(data = d)
    return dataf

actual_date = dt.date.today()
past_date = actual_date - dt.timedelta(days = 365*3)
actual_date = actual_date.strftime("%Y-%m-%d")

past_date = past_date.strftime("%Y-%m-%d")
df = get_stock_data(input("Input stock ticker in all caps: "), past_date, actual_date)
future_stock_prices = train_model(df)

frames = [df, future_stock_prices]
result = pd.concat(frames)
result.plot(x='Date', y='Price')

plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(loc = 4)

plt.show()