import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly

from plotly import graph_objs as go

START = "2019-1-1"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("基于Yahoo金融数据的股票价格预测 Stock Price Prediction")
st.write("本项目仅用作测试功能! This Web App is for Testing Only! - qwx")

stocks = ("AAPL", "GOOG", "TSLA", "NVDA")

selected_stocks = st.selectbox("选择示例股票代码 Select for prediction", stocks)

n_years = st.slider("预测年份 Years of prediction:", 1, 5)
period = n_years * 365


@st.cache_data
# streamlit can save data in cache when selecting stocks
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text("加载数据 Loading Data...")
data = load_data(selected_stocks)
data_load_state.text("加载数据完成 Loading Data Done！")

st.subheader("原始数据 Raw Data")
st.write(data.tail())


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_raw_data()

# Forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds",
                                    "Close": "y"})  # this is facebook prophet api requirements

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader("预测数据 Forecast Data")
st.write(forecast.tail())

st.write('预测数据图形 Forecast Data Figure')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("预测数据组成部分 Forecast Components")
fig2 = m.plot_components(forecast)
st.write(fig2)