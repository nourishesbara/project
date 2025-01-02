#!/usr/bin/env python
# coding: utf-8

# importing necesssary library to read data

# In[3]:


import pandas as pd
import numpy as np


# In[4]:


df = pd.read_csv("data.csv")
df


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.isnull().sum()


# In[8]:


df.shape


# In[9]:


df.info()


# In[10]:


df = pd.DataFrame(df)


# In[11]:


df["Date"]

CREATE FUNCTION TO CONVERT DATETIME TO DATE
# In[12]:


from datetime import datetime
#extratct date
def convert_datetime_to_date(date_time):
    dt = datetime.fromisoformat(date_time)

# Extract the date
    date_only = dt.date()
    return date_only



# In[13]:


df["Date"]=df["Date"].apply(convert_datetime_to_date)


# In[14]:


df


# In[19]:


import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Perform Seasonal Decomposition on the Close Price
result = seasonal_decompose(df['Close'], model='multiplicative', period=365)

# Plot the decomposition results
plt.figure(figsize=(20, 3))
result.plot()
plt.tight_layout()
plt.show()


# In[20]:


df['Daily Return'] = df['Close'].pct_change() * 100

df['7-Day MA'] = df['Close'].rolling(window=7).mean()
df['30-Day MA'] = df['Close'].rolling(window=30).mean()
df['50-Day MA'] = df['Close'].rolling(window=50).mean()
df['200-Day MA'] = df['Close'].rolling(window=200).mean()


# In[21]:


from sklearn.preprocessing import MinMaxScaler

# Step 5: Normalize/Scale data
scaler = MinMaxScaler()

# Columns to normalize
columns_to_scale = ['Open', 'High', 'Low', 'Close', 'Volume', 'Daily Return', 
                    '7-Day MA', '30-Day MA','50-Day MA', '200-Day MA']

# Apply MinMaxScaler
data_scaled = df.copy()
data_scaled[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# Display the first fe
data_scaled


# time-series graph
# 

# In[22]:


df = data_scaled


# In[23]:


# Example data: Time (dates), Open and Close values

growth = (df['Close'].iloc[-1]/df['Close'][0])*100

plt.figure(figsize=(13, 6))

# Plot the 'Open' values

# Plot the 'Close' values
plt.plot(df["Date"], df['Close'], label='Close', color='tab:blue')

# Adding labels and title
plt.title(f'Stock Prices:  growth {growth}%')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.legend()

# Display the plot
plt.xticks(rotation=45) # Rotate the x-axis labels for better visibility
plt.tight_layout()
plt.show()


# In[24]:


plt.figure(figsize=(13, 6))

# Plot the 'Close' values

plt.plot(df["Date"], df['7-Day MA'], label='7_Day moving average', color='tab:blue')
plt.plot(df["Date"], df['30-Day MA'], label='30_Day moving average', color='tab:red')

# Adding labels and title
plt.title(f'7_Day moving average v/s 30_Day moving average')
plt.xlabel('Date')
plt.ylabel('moving average')
plt.grid(True)
plt.legend()

# Display the plot
plt.xticks(rotation=45) # Rotate the x-axis labels for better visibility
plt.tight_layout()
plt.show()


# 

# In[25]:


plt.figure(figsize=(13, 6))

# Plot the 'Close' values

plt.plot(df["Date"], df['50-Day MA'], label='short-term moving average', color='tab:blue')
plt.plot(df["Date"], df['200-Day MA'], label='long-term moving average', color='tab:red')

# Adding labels and title
plt.title(f'short-term moving average v/s long-term moving average')
plt.xlabel('Date')
plt.ylabel('moving average')
plt.grid(True)
plt.legend()

# Display the plot
plt.xticks(rotation=45) # Rotate the x-axis labels for better visibility
plt.tight_layout()
plt.show()


# A Golden Cross occurs when the short-term moving average (50-day) crosses above the long-term moving average (200-day), signaling a bullish trend a potential buying opportunity.
# 
# A Death Cross occurs when the short-term moving average crosses below the long-term moving average, signaling a bearish trend a potential selling opportunity.
# 

# In[26]:


plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df['Daily Return'], label='Daily Returns', color='blue', linewidth=0.5)
plt.title('Normalized Daily Returns Over Time')
plt.xlabel('Date')
plt.ylabel('Normalized Daily Return')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.show()


# In[27]:


plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df['Volume'], label='Volume', color='blue', linewidth=1)
plt.title('Normalized Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Normalized Volume')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.show()


# In[28]:


plt.figure(figsize=(8, 6))
plt.hist(df['Daily Return'], bins=125, color='teal', edgecolor='black')
plt.title('Histogram of Daily Returns')
plt.xlabel('Normalized Daily Return')
plt.ylabel('Frequency')
plt.grid()
plt.show()


# ADDING NEW COUMNS TO THE DATASET:

# use lag to store it historical value we store 6 lag value because of weekly analysis

# In[29]:


for lag in range(1, 6):
    df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)

# Check the dataset to confirm new columns
df.head(10)


# In[30]:


print(df.isnull().sum())


# dealing with null values

# In[31]:


df.dropna(inplace=True)

# Verify no NaN values remain
print(df.isnull().sum())


# PRICE VOLATILITY MEASURE

# In[32]:


df['Rolling_Std'] = df['Close'].rolling(window=10).std()


# PRICE RATIO

# In[33]:


df['High_Low_Ratio'] = df['High'] / df['Low']


# FEATURE SCALING

# normalize the data using minmaxscalar

# In[34]:


features_to_scale = df.columns
features_to_scale = features_to_scale.drop('Date')

scaler = MinMaxScaler()

# Apply scaling to the selected features
data_scaled = df.copy()
data_scaled[features_to_scale] = scaler.fit_transform(df[features_to_scale])

# Display the first few rows of the scaled dataset
print(data_scaled.head())


# In[35]:


df = data_scaled


# In[36]:


df.dropna(inplace=True)


# In[37]:


df.head()


# TRAIN TEST SPLIT

# In[38]:


train_size = int(len(df) * 0.8)  # 80% for training, 20% for testing

# Step 3: Split the data
train = df.iloc[:train_size]  # First 80% as training data
test = df.iloc[train_size:]   # Remaining 20% as testing data
print(f"Training data size: {len(train)} rows")
print(f"Testing data size: {len(test)} rows")


# Visualization of spilit

# In[39]:


plt.figure(figsize=(12, 6))

plt.plot(train["Date"], train["Close"], label='train data', color='tab:blue')
plt.plot(test["Date"], test["Close"], label='test', color='tab:orange')

plt.title(f'train and test data visualization')
plt.xlabel('Date')
plt.ylabel('train - test')
plt.grid(True)
plt.legend()

# Display the plot
plt.xticks(rotation=45) # Rotate the x-axis labels for better visibility
plt.tight_layout()
plt.show()


# Forecasting Using ARIMA (AutoRegressive Integrated Moving Average

# In[40]:


train.index[-1]


# In[41]:


from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(train['Close'], order=(5, 1, 0))  # Adjust p, d, q after testing
fitted_model = model.fit()

# Forecast future values
forecast_steps = 24  # 24 months
forecast = fitted_model.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=train.index[-1], periods=forecast_steps+1, freq='M')[1:]

forecast_values = forecast.predicted_mean
confidence_intervals = forecast.conf_int()


# In[42]:


plt.figure(figsize=(12, 6))
plt.plot(train['Date'], train['Close'], label='Training Data')
plt.plot(test['Date'], test['Close'], label='Testing Data')

plt.plot(forecast_index, forecast_values, label='Forecast', color='red')
plt.fill_between(forecast_index, confidence_intervals.iloc[:, 0],
                 confidence_intervals.iloc[:, 1], 
                 color='pink', alpha=0.3, label='Confidence Interval')

plt.title('ARIMA Stock Price Forecast')
plt.legend()
plt.show()


# In[44]:


from sklearn.linear_model import LinearRegression

# Feature engineering: Moving averages
train['50-Day MA'] = train['Close'].rolling(window=50).mean()
train['200-Day MA'] = train['Close'].rolling(window=200).mean()

# Drop NaN values caused by rolling
train.dropna(inplace=True)

# Prepare data
X_train = train[['50-Day MA', '200-Day MA']]
y_train = train['Close']

# Train the model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict on test data
test['50-Day MA'] = test['Close'].rolling(window=50).mean()
test['200-Day MA'] = test['Close'].rolling(window=200).mean()
test.dropna(inplace=True)

X_test = test[['50-Day MA', '200-Day MA']]
y_test = test['Close']

predictions = lr_model.predict(X_test)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(test['Date'], y_test, label='True Prices')
plt.plot(test['Date'], predictions, label='Predictions', color='red')
plt.title('Linear Regression with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Close Price')

plt.legend()
plt.show()


# In[ ]:





# In[ ]:




