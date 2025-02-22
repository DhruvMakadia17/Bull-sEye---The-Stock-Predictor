import datetime
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Bull's Eye - Stock Predictor",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
        .main {
            padding: 0rem 0rem;
        }
        .stApp {
            background-color: #f5f5f5;
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
        }
        h1 {
            color: #1E3D59;
            font-size: 2.5rem !important;
            font-weight: 700 !important;
        }
        h2 {
            color: #1E3D59;
            font-weight: 600 !important;
        }
        .stSelectbox label {
            color: #1E3D59;
            font-weight: 500;
        }
        .metric-card {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .css-1d391kg {
            padding-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Function to get stock data
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="max")
        df.reset_index(inplace=True)
        return df, None
    except Exception as e:
        return None, str(e)

# Function to create candlestick chart
def create_candlestick_chart(df):
    fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                                        open=df['Open'],
                                        high=df['High'],
                                        low=df['Low'],
                                        close=df['Close'])])
    fig.update_layout(
        title='Stock Price Movement',
        yaxis_title='Stock Price (₹)',
        template='plotly_white',
        height=600
    )
    return fig

# Function to create volume chart
def create_volume_chart(df):
    fig = px.bar(df, x='Date', y='Volume', 
                 title='Trading Volume Over Time',
                 template='plotly_white',
                 height=400)
    fig.update_traces(marker_color='rgba(58, 71, 80, 0.6)')
    return fig

# Navigation Menu with custom styling
selected = option_menu(
    None,
    options=["Home", "Analysis", "Prediction", "Performance"],
    icons=["house-fill", "graph-up", "calculator-fill", "speedometer2"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#1E3D59"},
        "icon": {"color": "white", "font-size": "25px"}, 
        "nav-link": {
            "font-size": "16px",
            "text-align": "center",
            "margin": "0px",
            "--hover-color": "#276678",
            "color": "white"
        },
        "nav-link-selected": {"background-color": "#276678"},
    }
)

# Stock list
STOCK_LIST = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "HCLTECH.NS", "ASIANPAINT.NS",
    "MARUTI.NS", "TITAN.NS", "WIPRO.NS", "NTPC.NS", "POWERGRID.NS",
    "ULTRACEMCO.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "ADANIENT.NS", "ONGC.NS"
]

##------------------------------- HOME PAGE -------------------------------
if selected == "Home":
    st.title("Welcome to Bull's Eye 📈")
    
    # Introduction container
    with st.container():
        st.markdown("""
        <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
            <h2>Your AI-Powered Stock Market Assistant</h2>
            <p style='font-size: 1.1em; color: #555;'>
                Bull's Eye combines advanced machine learning with real-time market data to help you make informed investment decisions. 
                Our platform provides comprehensive analysis, accurate predictions, and detailed market insights for Indian stocks.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")  # Spacing
    
    # Features section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background-color: white; padding: 20px; border-radius: 10px; height: 200px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
            <h3 style='color: #1E3D59;'>📊 Real-Time Analysis</h3>
            <p>Get instant access to comprehensive stock analysis with interactive charts and key metrics.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background-color: white; padding: 20px; border-radius: 10px; height: 200px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
            <h3 style='color: #1E3D59;'>🎯 Price Prediction</h3>
            <p>ML-powered price predictions to help you make data-driven investment decisions.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style='background-color: white; padding: 20px; border-radius: 10px; height: 200px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
            <h3 style='color: #1E3D59;'>📈 Performance Metrics</h3>
            <p>Track model accuracy and performance metrics for reliable predictions.</p>
        </div>
        """, unsafe_allow_html=True)

##------------------------------- ANALYSIS PAGE -------------------------------
elif selected == "Analysis":
    st.title("Market Analysis")
    
    # Stock selection and time range
    col1, col2 = st.columns([2, 1])
    with col1:
        ticker = st.selectbox("Select a Stock", STOCK_LIST)
    with col2:
        time_range = st.selectbox("Select Time Range", 
                                 ["1 Month", "3 Months", "6 Months", "1 Year", "5 Years", "Max"])
    
    # Fetch data
    df, error = get_stock_data(ticker)
    
    if error:
        st.error(f"Error fetching data: {error}")
    elif df is not None:
        # Filter data based on selected time range
        end_date = datetime.now()
        if time_range == "1 Month":
            start_date = end_date - timedelta(days=30)
        elif time_range == "3 Months":
            start_date = end_date - timedelta(days=90)
        elif time_range == "6 Months":
            start_date = end_date - timedelta(days=180)
        elif time_range == "1 Year":
            start_date = end_date - timedelta(days=365)
        elif time_range == "5 Years":
            start_date = end_date - timedelta(days=1825)
        else:
            start_date = df['Date'].min()
        
        df = df[df['Date'] >= pd.Timestamp(start_date)]
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        latest_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        price_change = latest_price - prev_price
        price_change_pct = (price_change / prev_price) * 100
        
        with col1:
            st.metric("Latest Price", f"₹{latest_price:.2f}", 
                     f"{price_change_pct:+.2f}%")
        
        with col2:
            st.metric("Day's High", f"₹{df['High'].iloc[-1]:.2f}")
        
        with col3:
            st.metric("Day's Low", f"₹{df['Low'].iloc[-1]:.2f}")
        
        with col4:
            st.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")
        
        # Charts
        st.plotly_chart(create_candlestick_chart(df), use_container_width=True)
        st.plotly_chart(create_volume_chart(df), use_container_width=True)

##------------------------------- PREDICTION PAGE -------------------------------
elif selected == "Prediction":
    st.title("Stock Price Prediction")
    
    ticker = st.selectbox("Select a Stock for Prediction", STOCK_LIST)
    
    if st.button('Generate Prediction', use_container_width=True):
        with st.spinner('Analyzing market data...'):
            df, error = get_stock_data(ticker)
            
            if error:
                st.error(f"Error fetching data: {error}")
            elif df is not None:
                # Data Cleaning
                for column in ['Open', 'High', 'Low', 'Close']:
                    mean = df[column].mean()
                    df[column] = df[column].fillna(mean)

                X = df[['Open', 'High', 'Low']]
                y = df['Close'].values.reshape(-1, 1)
                
                # Model training
                from sklearn.model_selection import train_test_split
                from sklearn.linear_regression import LinearRegression
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                # Latest prediction
                latest_data = df.iloc[-1]
                prediction = model.predict([[latest_data['Open'], latest_data['High'], latest_data['Low']]])[0][0]
                
                # Display prediction
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
                        <h3 style='color: #1E3D59; margin-bottom: 15px;'>Predicted Closing Price</h3>
                        <h2 style='color: #276678; margin: 0;'>₹{:.2f}</h2>
                    </div>
                    """.format(prediction), unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
                        <h3 style='color: #1E3D59; margin-bottom: 15px;'>Current Price</h3>
                        <h2 style='color: #276678; margin: 0;'>₹{:.2f}</h2>
                    </div>
                    """.format(df['Close'].iloc[-1]), unsafe_allow_html=True)
                
                # Historical accuracy
                st.write("")
                st.subheader("Prediction Analysis")
                
                y_pred = model.predict(X_test)
                from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("""
                    <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
                        <h4 style='color: #1E3D59; margin: 0;'>Model Accuracy</h4>
                        <h2 style='color: #276678; margin: 0;'>{:.2f}%</h2>
                    </div>
                    """.format(r2_score(y_test, y_pred) * 100), unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
                        <h4 style='color: #1E3D59; margin: 0;'>Mean Absolute Error</h4>
                        <h2 style='color: #276678; margin: 0;'>₹{:.2f}</h2>
                    </div>
                    """.format(mean_absolute_error(y_test, y_pred)), unsafe_allow_html=True)
                
                with col3:
                    st.markdown("""
                    <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
                        <h4 style='color: #1E3D59; margin: 0;'>Root Mean Squared Error</h4>
                        <h2 style='color: #276678; margin: 0;'>₹{:.2f}</h2>
                    </div>
                    """.format(np.sqrt(mean_squared_error(y_test, y_pred))), unsafe_allow_html=True)

##------------------------------- PERFORMANCE PAGE -------------------------------
elif selected == "Performance":
    st.title("Model Performance Analysis")
    
    ticker = st.selectbox("Select a Stock", STOCK_LIST)
    
    if st.button('Analyze Performance', use_container_width=True):
        with st.spinner('Calculating performance metrics...'):
            df, error = get_stock_data(ticker)
            
            if error:
                st.error(f"Error fetching data: {error}")
            elif df is not None:
                # Data preparation
                # Data preparation
                for column in ['Open', 'High', 'Low', 'Close']:
                    mean = df[column].mean()
                    df[column] = df[column].fillna(mean)

                X = df[['Open', 'High', 'Low']]
                y = df['Close'].values.reshape(-1, 1)
                
                # Model training and evaluation
                from sklearn.model_selection import train_test_split, cross_val_score
                from sklearn.linear_regression import LinearRegression
                from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                # Cross validation scores
                cv_scores = cross_val_score(model, X, y, cv=5)
                
                # Display metrics in cards
                st.write("")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
                        <h3 style='color: #1E3D59; margin-bottom: 15px;'>Model Accuracy Metrics</h3>
                        <table style='width: 100%;'>
                            <tr>
                                <td style='padding: 10px 0;'><strong>R² Score:</strong></td>
                                <td style='text-align: right;'>{:.4f}</td>
                            </tr>
                            <tr>
                                <td style='padding: 10px 0;'><strong>Mean Absolute Error:</strong></td>
                                <td style='text-align: right;'>₹{:.2f}</td>
                            </tr>
                            <tr>
                                <td style='padding: 10px 0;'><strong>Root Mean Squared Error:</strong></td>
                                <td style='text-align: right;'>₹{:.2f}</td>
                            </tr>
                        </table>
                    </div>
                    """.format(r2, mae, rmse), unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
                        <h3 style='color: #1E3D59; margin-bottom: 15px;'>Cross Validation Results</h3>
                        <p><strong>Mean CV Score:</strong> {:.4f}</p>
                        <p><strong>Standard Deviation:</strong> {:.4f}</p>
                    </div>
                    """.format(cv_scores.mean(), cv_scores.std()), unsafe_allow_html=True)
                
                # Prediction vs Actual Plot
                st.write("")
                st.subheader("Prediction vs Actual Values")
                
                fig = go.Figure()
                
                # Add actual values
                fig.add_trace(go.Scatter(
                    y=y_test.flatten(),
                    name='Actual Values',
                    mode='markers',
                    marker=dict(color='blue', size=8, opacity=0.6),
                    showlegend=True
                ))
                
                # Add predicted values
                fig.add_trace(go.Scatter(
                    y=y_pred.flatten(),
                    name='Predicted Values',
                    mode='markers',
                    marker=dict(color='red', size=8, opacity=0.6),
                    showlegend=True
                ))
                
                fig.update_layout(
                    title='Prediction vs Actual Stock Prices',
                    xaxis_title='Data Points',
                    yaxis_title='Stock Price (₹)',
                    template='plotly_white',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Error Distribution
                st.write("")
                st.subheader("Error Distribution")
                
                errors = y_test.flatten() - y_pred.flatten()
                
                fig = px.histogram(
                    x=errors,
                    nbins=50,
                    title='Distribution of Prediction Errors',
                    labels={'x': 'Prediction Error (₹)', 'y': 'Count'},
                    template='plotly_white'
                )
                
                fig.update_layout(
                    showlegend=False,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature Importance
                st.write("")
                st.subheader("Feature Importance")
                
                importance_df = pd.DataFrame({
                    'Feature': ['Opening Price', 'High Price', 'Low Price'],
                    'Importance': np.abs(model.coef_[0])
                })
                importance_df['Importance'] = importance_df['Importance'] / importance_df['Importance'].sum()
                
                fig = px.bar(
                    importance_df,
                    x='Feature',
                    y='Importance',
                    title='Feature Importance in Prediction',
                    template='plotly_white'
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

# Required pip installations (add to requirements.txt):
# streamlit
# yfinance
# pandas
# numpy
# plotly
# scikit-learn
# streamlit-option-menu
