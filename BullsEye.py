import datetime
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from streamlit_option_menu import option_menu
import warnings
warnings.filterwarnings('ignore')

# Function to get stock data using yfinance
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="max")  # This will fetch all available historical data
    df.reset_index(inplace=True)  # Convert Date from index to column
    return df

# Navigation Menu
selected = option_menu(None,
    options=["Home", "Visualization", "Prediction", "Accuracy"],
    icons=["house", "graph", "book", "envelope"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)

##------------------------------- HOME PAGE -------------------------------
if selected == "Home":
    st.title("Welcome to Bull's Eye📈")
    st.write("Now have a better glance of market with this ML powered stock price predictor & invest better.")
    st.write("\n")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image("Media/icici.png", width=140)
        st.write("\n")
        st.image("Media/sbi.png", width=140)
        st.write("\n")
        st.image("Media/adani.png", width=140)

    with col2:
        st.image("Media/RIL.png",width=160)
        st.write("\n")
        st.image("Media/eicher.png", width=140)
        st.write("\n")
        st.image("Media/brit.png",width=140)

    with col3:
        st.image("Media/tatapower.png", width=160)
        st.write("\n")
        st.write("\n")
        st.image("Media/mrf.png", width=140)
        st.write("\n")
        st.image("Media/Unilever.png", width=130)
        
    with col4:
        st.image("Media/hdfc.jpg", width=120)
        st.write("\n")
        st.image("Media/nestle.jpg", width=140)
        st.write("\n")
        st.write("\n")
        st.image("Media/bajaj.png", width=140)
        st.write("\n")
        st.image("Media/titan.png", width=140)

##------------------------------- PREDICTION PAGE -------------------------------
elif selected == "Prediction":
    st.title("Predictor")
    # Updated ticker symbols without .NS (yfinance format)
    ticker = st.selectbox("Pick any stock or index to predict:",
        ("APOLLOHOSP.NS","TATACONSUM.NS","TATASTEEL.NS","RELIANCE.NS","LT.NS","BAJAJ-AUTO.NS",
         "WIPRO.NS","BAJAJFINSV.NS","KOTAKBANK.NS","ULTRACEMCO.NS","BRITANNIA.NS","TITAN.NS",
         "INDUSINDBK.NS","ICICIBANK.NS","ONGC.NS","NTPC.NS","ITC.NS","BAJFINANCE.NS",
         "NESTLEIND.NS","TECHM.NS","HDFCLIFE.NS","HINDALCO.NS","BHARTIARTL.NS","CIPLA.NS",
         "TCS.NS","ADANIENT.NS","HEROMOTOCO.NS","MARUTI.NS","COALINDIA.NS","BPCL.NS",
         "HCLTECH.NS","ADANIPORTS.NS","DRREDDY.NS","EICHERMOT.NS","ASIANPAINT.NS","GRASIM.NS",
         "JSWSTEEL.NS","DIVISLAB.NS","TATACONSUM.NS","SBIN.NS","HDFCBANK.NS","HDFC.NS",
         "WIPRO.NS","UPL.NS","POWERGRID.NS","TATAPOWER.NS","TATAMOTORS.NS","SUNPHARMA.NS",
         "HINDUNILVR.NS","SBILIFE.NS","INFY.NS","AXISBANK.NS"))
    
    if st.button('Predict'):
        df = get_stock_data(ticker)
        # Data Cleaning
        for column in ['Open', 'High', 'Low', 'Close']:
            mean = df[column].mean()
            df[column] = df[column].fillna(mean)

        X = df[['Open','High','Low']]
        y = df['Close'].values.reshape(-1,1)
        
        # Splitting dataset into Training and Testing
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fitting Linear Regression
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        
        # Predicting for all data points
        n = len(df)
        pred = []
        for i in range(n):
            open_price = df['Open'].values[i]
            high = df['High'].values[i]
            low = df['Low'].values[i]
            output = reg.predict([[open_price, high, low]])
            pred.append(output)

        pred1 = np.concatenate(pred)
        predicted = pred1.flatten().tolist()

        latest_prediction = predicted[-1]
        st.subheader("Your latest predicted closing price is: ")
        st.title(f"₹{latest_prediction:.2f}")

    st.write('You selected:', ticker)

##------------------------------- DATA VISUALIZATION -------------------------------
elif selected == "Visualization":
    ticker = st.selectbox("Pick any stock or index to visualize:",
        ("APOLLOHOSP.NS","TATACONSUM.NS","TATASTEEL.NS","RELIANCE.NS","LT.NS","BAJAJ-AUTO.NS",
         "WIPRO.NS","BAJAJFINSV.NS","KOTAKBANK.NS","ULTRACEMCO.NS","BRITANNIA.NS","TITAN.NS",
         "INDUSINDBK.NS","ICICIBANK.NS","ONGC.NS","NTPC.NS","ITC.NS","BAJFINANCE.NS",
         "NESTLEIND.NS","TECHM.NS","HDFCLIFE.NS","HINDALCO.NS","BHARTIARTL.NS","CIPLA.NS",
         "TCS.NS","ADANIENT.NS","HEROMOTOCO.NS","MARUTI.NS","COALINDIA.NS","BPCL.NS",
         "HCLTECH.NS","ADANIPORTS.NS","DRREDDY.NS","EICHERMOT.NS","ASIANPAINT.NS","GRASIM.NS",
         "JSWSTEEL.NS","DIVISLAB.NS","TATACONSUM.NS","SBIN.NS","HDFCBANK.NS","HDFC.NS",
         "WIPRO.NS","UPL.NS","POWERGRID.NS","TATAPOWER.NS","TATAMOTORS.NS","SUNPHARMA.NS",
         "HINDUNILVR.NS","SBILIFE.NS","INFY.NS","AXISBANK.NS"))
    
    if st.button('Show Dataframe'):
        df = get_stock_data(ticker)
        st.dataframe(df)
    
    df = get_stock_data(ticker)
    st.write("Closing Price Trend")
    st.line_chart(data=df, x='Date', y='Close', use_container_width=True)
    st.write("Opening Price Trend")
    st.line_chart(data=df, x='Date', y='Open', use_container_width=True)
    st.write("Day High Trend")
    st.line_chart(data=df, x='Date', y='High', use_container_width=True)

##------------------------------- ACCURACY PAGE -------------------------------
elif selected == "Accuracy":
    st.title("Accuracy Evaluation Metrics")

    ticker = st.selectbox("Pick any stock to find its accuracy:",
        ("APOLLOHOSP.NS","TATACONSUM.NS","TATASTEEL.NS","RELIANCE.NS","LT.NS","BAJAJ-AUTO.NS",
         "WIPRO.NS","BAJAJFINSV.NS","KOTAKBANK.NS","ULTRACEMCO.NS","BRITANNIA.NS","TITAN.NS",
         "INDUSINDBK.NS","ICICIBANK.NS","ONGC.NS","NTPC.NS","ITC.NS","BAJFINANCE.NS",
         "NESTLEIND.NS","TECHM.NS","HDFCLIFE.NS","HINDALCO.NS","BHARTIARTL.NS","CIPLA.NS",
         "TCS.NS","ADANIENT.NS","HEROMOTOCO.NS","MARUTI.NS","COALINDIA.NS","BPCL.NS",
         "HCLTECH.NS","ADANIPORTS.NS","DRREDDY.NS","EICHERMOT.NS","ASIANPAINT.NS","GRASIM.NS",
         "JSWSTEEL.NS","DIVISLAB.NS","TATACONSUM.NS","SBIN.NS","HDFCBANK.NS","HDFC.NS",
         "WIPRO.NS","UPL.NS","POWERGRID.NS","TATAPOWER.NS","TATAMOTORS.NS","SUNPHARMA.NS",
         "HINDUNILVR.NS","SBILIFE.NS","INFY.NS","AXISBANK.NS"))
    
    df = get_stock_data(ticker)
    
    # Data Cleaning
    for column in ['Open', 'High', 'Low', 'Close']:
        mean = df[column].mean()
        df[column] = df[column].fillna(mean)

    X = df[['Open','High','Low']]
    y = df['Close'].values.reshape(-1,1)
    
    # Splitting dataset
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fitting Linear Regression
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    
    # Evaluating the model
    import sklearn.metrics as metrics
    r2 = metrics.r2_score(y_test, y_pred)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    col1, col2 = st.columns(2)
    
    col1.metric("R² Score", f"{r2:.4f}", "±5%")
    col2.metric("Mean Absolute Error", f"₹{mae:.2f}", "±5%")
    col1.metric("Mean Squared Error", f"₹{mse:.2f}", "±5%")
    col2.metric("Root Mean Squared Error", f"₹{rmse:.2f}", "±5%")
