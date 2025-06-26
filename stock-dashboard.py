from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import tempfile
import base64
import os
import json
from datetime import datetime, timedelta

# Optional: Ollama if installed and used
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Optional: Gemini if configured
try:
    import google.generativeai as genai
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", "")
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        gen_model = genai.GenerativeModel('gemini-2.0-flash')
        GEMINI_AVAILABLE = True
    else:
        GEMINI_AVAILABLE = False
except Exception:
    GEMINI_AVAILABLE = False

pio.kaleido.scope.default_format = "png"

# ---------- FUNCTIONS ----------
def fetch_stock_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end)

def compute_vwap(data):
    return (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()

def add_indicators(fig, data, indicators):
    if "VWAP" in indicators:
        data['VWAP'] = compute_vwap(data)
    for indicator in indicators:
        if indicator == "20-Day SMA":
            sma = data['Close'].rolling(window=20).mean()
            fig.add_trace(go.Scatter(x=data.index, y=sma, mode='lines', name='SMA (20)'))
        elif indicator == "20-Day EMA":
            ema = data['Close'].ewm(span=20).mean()
            fig.add_trace(go.Scatter(x=data.index, y=ema, mode='lines', name='EMA (20)'))
        elif indicator == "20-Day Bollinger Bands":
            sma = data['Close'].rolling(window=20).mean()
            std = data['Close'].rolling(window=20).std()
            fig.add_trace(go.Scatter(x=data.index, y=sma + 2*std, mode='lines', name='BB Upper'))
            fig.add_trace(go.Scatter(x=data.index, y=sma - 2*std, mode='lines', name='BB Lower'))
        elif indicator == "VWAP":
            fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], mode='lines', name='VWAP'))

def plot_chart(data, indicators, ticker):
    fig = go.Figure(data=[
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Candlestick')
    ])
    add_indicators(fig, data, indicators)
    fig.update_layout(xaxis_rangeslider_visible=False, height=600,
                      title=f"{ticker} Candlestick Chart")
    return fig

def save_figure_image(fig):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        fig.write_image(tmpfile.name)
        return tmpfile.name

def encode_image_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def analyze_with_gemini(ticker, fig):
    image_path = save_figure_image(fig)
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    os.remove(image_path)

    contents = [
        {"role": "user", "parts": [
            f"You are a Stock Trader specializing in Technical Analysis. Analyze the stock chart for {ticker}. "
            "Explain patterns, signals, and trends. Provide a JSON output with 'action' and 'justification'."
        ]},
        {"role": "user", "parts": [{"data": image_bytes, "mime_type": "image/png"}]}
    ]

    response = gen_model.generate_content(contents=contents)
    text = response.text
    try:
        start, end = text.find('{'), text.rfind('}')+1
        result = json.loads(text[start:end])
    except:
        result = {"action": "Error", "justification": text}
    return result

def analyze_with_ollama(fig):
    image_path = save_figure_image(fig)
    image_b64 = encode_image_base64(image_path)
    os.remove(image_path)

    messages = [{
        "role": "user",
        "content": (
            "You are a Stock Trader. Analyze this stock chart. Give Buy/Hold/Sell, then justification."),
        "images": [image_b64]
    }]
    response = ollama.chat(model='llama3.2-vision', messages=messages)
    return {"action": "Unstructured", "justification": response["message"]["content"]}

# ---------- STREAMLIT APP ----------
st.set_page_config(layout="wide")
st.title("Sentinel: AI-Powered Technical Stock Dashboard")
st.sidebar.header("Configuration")

model_choice = st.sidebar.radio("Choose AI Model", ["Gemini", "LLaMA" if OLLAMA_AVAILABLE else "(LLaMA Not Installed)"])
tickers_input = st.sidebar.text_input("Enter Tickers (comma-separated):", "AAPL,MSFT,GOOG")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

start = st.sidebar.date_input("Start Date", datetime.today() - timedelta(days=365))
end = st.sidebar.date_input("End Date", datetime.today())
indicators = st.sidebar.multiselect("Indicators", ["20-Day SMA", "20-Day EMA", "20-Day Bollinger Bands", "VWAP"], ["20-Day SMA"])

if st.sidebar.button("Fetch Data"):
    all_data = {}
    for t in tickers:
        df = fetch_stock_data(t, start, end)
        if not df.empty:
            all_data[t] = df
        else:
            st.warning(f"No data for {t}")
    st.session_state["stock_data"] = all_data

if "stock_data" in st.session_state and st.session_state["stock_data"]:
    tab_names = ["Overall Summary"] + list(st.session_state["stock_data"].keys())
    tabs = st.tabs(tab_names)
    summary = []

    for i, ticker in enumerate(st.session_state["stock_data"]):
        data = st.session_state["stock_data"][ticker]
        fig = plot_chart(data, indicators, ticker)
        if model_choice == "Gemini" and GEMINI_AVAILABLE:
            result = analyze_with_gemini(ticker, fig)
        elif model_choice.startswith("LLaMA") and OLLAMA_AVAILABLE:
            result = analyze_with_ollama(fig)
        else:
            result = {"action": "Unavailable", "justification": "AI model not available."}
        summary.append({"Stock": ticker, "Recommendation": result.get("action", "N/A")})

        with tabs[i+1]:
            st.subheader(f"Analysis for {ticker}")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Justification:**")
            st.write(result.get("justification", "No justification available."))

    with tabs[0]:
        st.subheader("Overall Recommendations")
        st.table(pd.DataFrame(summary))
else:
    st.info("Please load stock data from the sidebar!")
