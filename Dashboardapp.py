
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

#download data

sugar_data = pd.read_csv('psd_sugar.csv')
delhi_weather = pd.read_csv('Delhi_NCR_1990_2022_Safdarjung.csv')
lucknow_weather = pd.read_csv('Lucknow_1990_2022.csv')
mumbai_weather = pd.read_csv('Mumbai_1990_2022_Santacruz.csv')
sugar_price = pd.read_csv('PSUGAISAUSDM.csv')

#clean data

delhi_weather.rename(columns={'time': 'Date'}, inplace=True)
delhi_weather.rename(columns={'tavg': 'Delhi_avg_weather'}, inplace=True)
delhi_weather.rename(columns={'prcp': 'Delhi_prcp'}, inplace=True)

delhi_weather['Date'] = pd.to_datetime(delhi_weather['Date'], dayfirst=True, errors='coerce')

#clean version of delhi weather
delhi_weather_data = delhi_weather[['Date', 'Delhi_avg_weather', 'Delhi_prcp']]


lucknow_weather.rename(columns={'time': 'Date'}, inplace=True)
lucknow_weather.rename(columns={'tavg': 'Lucknow_avg_weather'}, inplace=True)
lucknow_weather.rename(columns={'prcp': 'Lucknow_prcp'}, inplace=True)


lucknow_weather['Date'] = pd.to_datetime(lucknow_weather['Date'], dayfirst=True, errors='coerce')

#clean version of lucknow weather
lucknow_weather_data = lucknow_weather[['Date', 'Lucknow_avg_weather', 'Lucknow_prcp']]

mumbai_weather.rename(columns={'time': 'Date'}, inplace=True)
mumbai_weather.rename(columns={'tavg': 'Mumbai_avg_weather'}, inplace=True)
mumbai_weather.rename(columns={'prcp': 'Mumbai_prcp'}, inplace=True)

mumbai_weather['Date'] = pd.to_datetime(mumbai_weather['Date'], dayfirst=True, errors='coerce')


#clean version of mumabi weather 
mumbai_weather_data = mumbai_weather[['Date', 'Mumbai_avg_weather', 'Mumbai_prcp']]

mumbai_lucknow_weather = pd.merge(mumbai_weather_data, lucknow_weather_data, on='Date')
india_weather = pd.merge(mumbai_lucknow_weather, delhi_weather_data, on='Date')


india_weather['Date'] = pd.to_datetime(india_weather['Date'])
india_weather.set_index('Date', inplace=True)


countries= ['Brazil', 'India']
sugar_data.rename(columns={'Calendar_Year': 'Date'}, inplace=True)
sugar_data.rename(columns={'Value': 'Total_Cane_Sugar_Production_1000MT'}, inplace=True)

sugar_data['Date'] = pd.to_datetime(sugar_data['Date'], format='%Y', errors='coerce')

brl_inr = sugar_data[sugar_data['Country_Name'].isin(countries)]
sugar_brl_inr = brl_inr[brl_inr['Attribute_Description'] == 'Cane Sugar Production']


#Clear version of brazil and India total sugar production from 1960
brazil_india_sugar = sugar_brl_inr[['Date','Country_Name', 'Total_Cane_Sugar_Production_1000MT']]

brazil_india_sugar = brazil_india_sugar.copy()
brazil_india_sugar.loc[:, 'Date'] = pd.to_datetime(brazil_india_sugar['Date'])
brazil_india_sugar.set_index('Date', inplace=True)

brazil_india_sugar = brazil_india_sugar.groupby('Date').sum()


daily_sugar_production = brazil_india_sugar.resample('D').ffill()
daily_sugar_production['daily_production'] = daily_sugar_production['Total_Cane_Sugar_Production_1000MT'] / 365
daily_sugar_production = daily_sugar_production[['daily_production']]

sugar_price.rename(columns={'PSUGAISAUSDM': 'US_Cents_per_Pound_sugar'}, inplace=True)
sugar_price['observation_date'] = pd.to_datetime(sugar_price['observation_date'], errors='coerce')
sugar_price.set_index('observation_date', inplace=True)

daily_sugar_price = sugar_price.resample('D').interpolate(method='linear')

daily_features = pd.merge(india_weather, daily_sugar_production, left_index=True, right_index=True, how='inner')
daily_data = pd.merge(daily_features, daily_sugar_price, left_index=True, right_index=True, how='inner')

x = daily_data.drop(columns=['US_Cents_per_Pound_sugar'])
y = daily_data['US_Cents_per_Pound_sugar']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler()),               
    ('lr', LinearRegression())                  
])

pipeline.fit(x_train, y_train)

y_pred = pipeline.predict(x_test)

mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error (MAE):", mae)

daily_data['forecasted_price'] = pipeline.predict(x)
daily_data['adjusted_price'] = (1.03 * daily_data['forecasted_price']) + (0.2 * daily_data['forecasted_price'])


st.title("Sugar Price Forecast Dashboard")
col1, col2 = st.columns(2)
today_forecast = daily_data['forecasted_price'].iloc[-1]  # Last row as today's forecast
today_adjusted = daily_data['adjusted_price'].iloc[-1]  # Last row as today's adjusted price


with col1:
    st.metric("Today's Forecasted Price (US Cents/Lb)", round(today_forecast, 2))

with col2:
    st.metric("Today's Adjusted Price (US Cents/Lb)", round(today_adjusted, 2))

# **Toggle Section for Time Selection**
st.subheader("Explore Prices Over Time")
selected_date = st.slider(
    "Select a date:",
    min_value=daily_data.index.min(),
    max_value=daily_data.index.max(),
    value=daily_data.index[-1],  # Default to most recent date
    format="YYYY-MM-DD"
)

selected_forecast = daily_data.loc[selected_date, 'forecasted_price']
selected_adjusted = daily_data.loc[selected_date, 'adjusted_price']

col3, col4 = st.columns(2)
with col3:
    st.metric("Forecasted Price (US Cents/Lb)", round(selected_forecast, 2))
with col4:
    st.metric("Adjusted Price (US Cents/Lb)", round(selected_adjusted, 2))


st.subheader("Forecasted vs Adjusted Prices Over Time")
st.line_chart(daily_data[['forecasted_price', 'adjusted_price']].rename(
    columns={"forecasted_price": "Forecasted Price", "adjusted_price": "Adjusted Price"}
))


# Data Display
st.subheader("Detailed Data")
st.dataframe(daily_data[['forecasted_price', 'adjusted_price', 'US_Cents_per_Pound_sugar']])
