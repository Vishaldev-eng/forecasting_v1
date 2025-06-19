import streamlit as st
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pyodbc
import os
import pandas as pd
from dotenv import load_dotenv

# Load .env file
load_dotenv()


server = st.secrets["DB_SERVER"]
user = st.secrets["DB_USER"]
password = st.secrets["DB_PASSWORD"]
initial_database=st.secrets["DB_NAME"]

# Initial connection string (for listing databases)
conn_str_initial = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={initial_database};UID={user};PWD={password}"

st.set_page_config(page_title='Forecasting', layout='centered')
# st.markdown(
#     """
#     <style>
#     .stApp {
#         background-color: black;
#         color: white;
#     }
#     /* Style for buttons */
#     div.stButton > button {
#         background-color: #444444;
#         color: white;
#         border-radius: 8px;
#         padding: 0.5em 1em;
#         border: none;
#     }
#     div.stButton > button:hover {
#         background-color: #666666;
#         color: white;
#     }
#     /* Style for text input boxes */
#     .stTextInput > div > div > input {
#         background-color: #222222;
#         color: white;
#     }
#     /* Style for selectboxes */
#     .stSelectbox > div > div > div {
#         background-color: #222222;
#         color: white;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )


st.title('Forecasting Application')

st.write("""
Easily connect to your Server, import data by writing SQL queries, and generate future forecasts.
         
1️⃣ Select a database  
2️⃣ Write and run your SQL query  
3️⃣ Select date and target columns for forecasting  
4️⃣ Run the forecast and view interactive plots  
""")

# Initialize session state
if "query_executed" not in st.session_state:
    st.session_state.query_executed = False

try:
    connection_initial = pyodbc.connect(conn_str_initial)
    #st.success("Connected to server successfully.")

    # Fetch list of databases
    db_query = "SELECT name FROM sys.databases"
    dbs = pd.read_sql(db_query, connection_initial)
    db_list = dbs['name'].tolist()

    selected_db = st.selectbox("Select Database", db_list)

    if selected_db:
        # Connect to selected database
        conn_str_db = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={selected_db};UID={user};PWD={password}"
        connection_db = pyodbc.connect(conn_str_db)

        st.write(f"Connected to database: **{selected_db}**")

        sql_query = st.text_area("Enter your SQL query", height=200)

        if st.button("Import Data"):
            try:
                df = pd.read_sql_query(sql_query, connection_db)
                st.session_state['df'] = df
                st.session_state.query_executed = True
                st.success("Data imported successfully.")
            except Exception as e:
                st.error(f"Failed to import data: {e}")
                st.session_state.query_executed = False

        # If query was executed, show forecasting options
        if st.session_state.query_executed:
            df = st.session_state['df']
            st.subheader("Imported Data")
            st.dataframe(df)

            date_column = st.selectbox("Select Date Column", df.columns, key='date_column')
            value_column = st.selectbox("Select Target Column", df.columns, key='value_column')

            freq_options = {
                'D: calendar day': 'D',
                'W: weekly': 'W',
                # 'h: hourly': 'h',
                # 'min: minutely': 'min',
                # 's: secondly': 's',
                'MS: month start frequency': 'MS',
                'ME: month end frequency': 'ME',
                'QS: quarter start frequency': 'QS',
                'QE: quarter end frequency': 'QE',
                'YS: year start frequency': 'YS',
                'YE: year end frequency': 'YE'
            }
            freq_label = st.selectbox("Select Frequency", list(freq_options.keys()), key='freq')
            freq_input = freq_options[freq_label]

            periods_input = st.number_input('Periods to Forecast into Future:', min_value=1, max_value=500, value=100, key='periods')

            model_choice = st.radio("Choose Forecasting Model", ("Prophet", "Exponential Smoothing", "SARIMA", "Moving Average"))

            ma_window = None
            if model_choice == "Moving Average":
                ma_window = st.slider("Moving Average Window (periods)", min_value=2, max_value=24, value=3)

            if st.button("Run Forecast"):
                if model_choice == "Prophet":
                    try:
                        # Prepare data
                        df_subset = df[[date_column, value_column]].copy()
                        df_subset = df_subset.rename(columns={date_column: 'ds', value_column: 'y'})
                        df_subset['ds'] = pd.to_datetime(df_subset['ds'])
                        df_subset['y'] = pd.to_numeric(df_subset['y'], errors='coerce')
                        df_subset.dropna(subset=['ds', 'y'], inplace=True)

                        st.subheader("Input Data for Forecasting")
                        st.dataframe(df_subset.sort_values(by='ds'))

                        # Build model
                        model = Prophet()
                        model.fit(df_subset)

                        future = model.make_future_dataframe(periods=periods_input, freq=freq_input)
                        forecast = model.predict(future)

                        st.subheader("Forecast Results")
                        forecast_output = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods_input)
                        st.dataframe(forecast_output)

                        st.subheader("Forecast Plot")
                        fig1 = plot_plotly(model, forecast)
                        st.plotly_chart(fig1)

                        st.subheader("Forecast Components")
                        fig2 = plot_components_plotly(model, forecast)
                        st.plotly_chart(fig2)

                    except Exception as e:
                        st.error(f"Forecast failed: {e}")


                elif model_choice == "Exponential Smoothing":
                    df_subset = df[[date_column, value_column]].copy()
                    df_subset = df_subset.rename(columns={date_column: 'ds', value_column: 'y'})
                    df_subset['ds'] = pd.to_datetime(df_subset['ds'])
                    df_subset['y'] = pd.to_numeric(df_subset['y'], errors='coerce')
                    df_subset.dropna(subset=['ds', 'y'], inplace=True)

                    st.subheader("Input Data for Forecasting")
                    st.dataframe(df_subset.sort_values(by='ds'))

                    df_subset.set_index('ds', inplace=True)
                    df_subset.sort_index(inplace=True)
                    ts_data = df_subset['y']
                    model = ExponentialSmoothing(ts_data, trend='add', seasonal='add')
                    model_fit = model.fit()
                    forecast = model_fit.forecast(periods_input)

                    forecast_df = forecast.reset_index()
                    forecast_df.columns = ['date', 'forecast']
                    forecast_df['date'] = forecast_df['date'].dt.strftime('%Y-%m-%d')
                    forecast_df['forecast'] = forecast_df['forecast'].round(2)

                    st.subheader("Forecasted Data")
                    st.dataframe(forecast_df)

                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(ts_data, label="Original Data")
                    ax.plot(model_fit.fittedvalues, label="Fitted")
                    ax.plot(forecast, label="Forecast")
                    ax.set_title("Triple Exponential Smoothing Forecast")
                    ax.legend()

                    # Format x-axis as YYYY-MM-DD
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    plt.xticks(rotation=90)

                    # Ensure tick locations match data points for fitted and forecast
                    ax.set_xticks(list(ts_data.index) + list(forecast.index))

                    st.pyplot(fig)

                    

                elif model_choice == "SARIMA":
                    df_subset = df[[date_column, value_column]].copy()
                    df_subset = df_subset.rename(columns={date_column: 'ds', value_column: 'y'})
                    df_subset['ds'] = pd.to_datetime(df_subset['ds'])
                    df_subset['y'] = pd.to_numeric(df_subset['y'], errors='coerce')
                    df_subset.dropna(subset=['ds', 'y'], inplace=True)

                    st.subheader("Input Data for Forecasting")
                    st.dataframe(df_subset.sort_values(by='ds'))

                    df_subset.set_index('ds', inplace=True)
                    df_subset.sort_index(inplace=True)
                    ts_data = df_subset['y']
                
                    
                    model = SARIMAX(ts_data, order=(0, 1, 1), seasonal_order=(2, 1, 1, 4))
                    model_fit = model.fit()
                    forecast = model_fit.predict(start=len(ts_data), end=(len(ts_data) + periods_input - 1))


                    forecast_df = forecast.reset_index()
                    forecast_df.columns = ['date', 'forecast']
                    forecast_df['date'] = forecast_df['date'].dt.strftime('%Y-%m-%d')
                    forecast_df['forecast'] = forecast_df['forecast'].round(2)

                    st.subheader("Forecasted Data")
                    st.dataframe(forecast_df)

                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(ts_data, label="Original Data")
                    ax.plot(forecast, label="SARIMA Forecast")
                    ax.set_title("SARIMA Forecast")
                    ax.legend()

                    # Format x-axis as YYYY-MM-DD
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    plt.xticks(rotation=90)

                    # Ensure tick locations match data points for fitted and forecast
                    ax.set_xticks(list(ts_data.index) + list(forecast.index))
                    
                    st.pyplot(fig)

                    

                elif model_choice == "Moving Average":
                    df_subset = df[[date_column, value_column]].copy()
                    df_subset = df_subset.rename(columns={date_column: 'ds', value_column: 'y'})
                    df_subset['ds'] = pd.to_datetime(df_subset['ds'])
                    df_subset['y'] = pd.to_numeric(df_subset['y'], errors='coerce')
                    df_subset.dropna(subset=['ds', 'y'], inplace=True)

                    st.subheader("Input Data for Forecasting")
                    st.dataframe(df_subset.sort_values(by='ds'))

                    df_subset.set_index('ds', inplace=True)
                    df_subset.sort_index(inplace=True)
                    ts_data = df_subset['y']

                    

                    rolling_mean = ts_data.rolling(window=ma_window).mean()
                    last_avg = rolling_mean.dropna().iloc[-1]

                    forecast = pd.Series(
                        [last_avg] * periods_input,
                        index=pd.date_range(start=ts_data.index[-1] + pd.DateOffset(1),
                                            periods=periods_input, freq = freq_input)  
                    )

                    forecast_df = forecast.reset_index()
                    forecast_df.columns = ['date', 'forecast']
                    forecast_df['date'] = forecast_df['date'].dt.strftime('%Y-%m-%d')
                    forecast_df['forecast'] = forecast_df['forecast'].round(2)

                    st.subheader("Forecasted Data")
                    st.dataframe(forecast_df)

                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(ts_data, label="Original Data")
                    ax.plot(rolling_mean, label=f"{ma_window}-Period Moving Average", linestyle='--')
                    ax.plot(forecast, label="Forecast")
                    ax.set_title(f"Moving Average Forecast ({ma_window} periods)")
                    ax.legend()

                    # Format x-axis as YYYY-MM-DD
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    plt.xticks(rotation=90)

                    # Ensure tick locations match data points for fitted and forecast
                    ax.set_xticks(list(ts_data.index) + list(forecast.index))


                    st.pyplot(fig)

                    
                else:
                    st.write("To Do")

except Exception as e:
    st.error(f"Connection failed: {e}")
