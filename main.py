
import streamlit as st
from datetime import date

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')
st.subheader('Austin Powell')

# stocks = ('TSLA','GOOG', 'AAPL', 'MSFT', 'GME')
# selected_stock = st.selectbox('Select dataset for prediction ({})'.format(stocks), stocks)

default_value_goes_here = 'GME'
selected_stock = st.text_input("Input stock ticker symbol for prediction", default_value_goes_here)

n_years = st.slider('Years ahead to predict prediction:', 1, 4)
period = n_years * 365


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

st.spinner(text='Training progress...')
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

add_selectbox = st.sidebar.text_area(
    "Any feedback on app?", "---> Love it! \n ---> Not sure how to process the data? \ns---> Suggested Improvements?"
)
 
# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()

m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.markdown("""
## Forecast
### Forecast data
""")
st.write(forecast.tail())
    
st.markdown("### Forecast plot for {} years".format(n_years))
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

#st.write("Forecast components")
st.markdown("""
### Forecast seasonal components:
Time series broken appart into seasonal components. Can give you an idea of what happens on a regular basis with the data.
""")
fig2 = m.plot_components(forecast)
st.write(fig2)