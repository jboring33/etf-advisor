import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="ETF Strategy Engine", layout="wide")

st.title("📈 Custom ETF Screener & Signal Engine")
st.markdown("Screen candidate ETFs based on custom risk profiles, then evaluate technical/fundamental rules for trade signals.")

# --- SIDEBAR: EDITABLE SCREENING & RISK RULES ---
st.sidebar.header("1. Screening & Risk Rules")

risk_profile = st.sidebar.selectbox(
    "Target Risk Profile",
    ["Conservative (Capital Preservation)", "Moderate (Balanced)", "Aggressive (Growth)"]
)

# Editable parameters
max_expense_ratio = st.sidebar.slider("Max Expense Ratio (%)", 0.03, 0.75, 0.20, 0.01)
min_yield = st.sidebar.slider("Min Dividend Yield (%)", 0.0, 5.0, 1.5, 0.1)

# Pre-populated ticker universe based on risk profile
UNIVERSE = {
    "Conservative (Capital Preservation)": ["BND", "SCHP", "VTIP", "SHY", "VCIT"],
    "Moderate (Balanced)": ["SPY", "VOO", "VTI", "SCHD", "VYM", "BND"],
    "Aggressive (Growth)": ["QQQ", "VUG", "SCHG", "IWM", "ARKK", "SMH"]
}

candidate_tickers = UNIVERSE[risk_profile]

# --- MAIN TAB 1: ETF SCREENER ---
tab1, tab2 = st.tabs(["🔍 Candidate Screener", "📊 Deep Dive & Buy/Sell Signals"])

with tab1:
    st.subheader(f"Screening Universe: {risk_profile}")
    st.write(f"Evaluating candidate tickers: `{', '.join(candidate_tickers)}` against Max Expense Ratio: **{max_expense_ratio}%** and Min Yield: **{min_yield}%**.")
    
    screener_data = []
    
    with st.spinner("Fetching ETF fundamentals..."):
        for ticker in candidate_tickers:
            t = yf.Ticker(ticker)
            info = t.info
            
            exp_ratio = info.get("expenseRatio", 0.0) * 100 if info.get("expenseRatio") else 0.09 # fallback estimate
            div_yield = info.get("yield", 0.0) * 100 if info.get("yield") else (info.get("dividendYield", 0.0) or 0.0) * 100
            aum = info.get("totalAssets", 0)
            
            # Pass/Fail Check
            passes_rules = (exp_ratio <= max_expense_ratio) and (div_yield >= min_yield)
            
            screener_data.append({
                "Ticker": ticker,
                "Name": info.get("shortName", ticker),
                "Category": info.get("category", "N/A"),
                "Expense Ratio (%)": round(exp_ratio, 2),
                "Yield (%)": round(div_yield, 2),
                "Status": "✅ Pass" if passes_rules else "❌ Filtered Out"
            })
            
    df_screener = pd.DataFrame(screener_data)
    st.dataframe(df_screener, use_container_width=True)

# --- MAIN TAB 2: DETAILED RULE ENGINE & BUY/SELL/HOLD SIGNAL ---
with tab2:
    st.subheader("2. Ticker Selection & Signal Rules")
    
    # Allow user to pick from the screened list
    selected_ticker = st.selectbox("Select an ETF to evaluate:", candidate_tickers)
    
    st.markdown("---")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Editable Technical Rules")
        sma_short_period = st.number_input("Short Moving Average (Days)", min_value=5, max_value=50, value=20)
        sma_long_period = st.number_input("Long Moving Average (Days)", min_value=20, max_value=200, value=50)
        rsi_overbought = st.slider("RSI Overbought (Sell Barrier)", 60, 85, 70)
        rsi_oversold = st.slider("RSI Oversold (Buy Barrier)", 15, 40, 30)

    # Fetch price history for analysis
    ticker_obj = yf.Ticker(selected_ticker)
    hist = ticker_obj.history(period="1y")
    
    if len(hist) > sma_long_period:
        # Technical calculations
        hist['SMA_Short'] = hist['Close'].rolling(window=sma_short_period).mean()
        hist['SMA_Long'] = hist['Close'].rolling(window=sma_long_period).mean()
        
        # Relative Strength Index (RSI) calculation
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        hist['RSI'] = 100 - (100 / (1 + rs))
        
        latest_close = hist['Close'].iloc[-1]
        latest_sma_short = hist['SMA_Short'].iloc[-1]
        latest_sma_long = hist['SMA_Long'].iloc[-1]
        latest_rsi = hist['RSI'].iloc[-1]
        
        # --- RECOMMENDATION LOGIC ---
        signal = "HOLD"
        reasoning = []
        
        if latest_sma_short > latest_sma_long and latest_rsi < rsi_overbought:
            signal = "BUY"
            reasoning.append(f"Bullish Moving Average Crossover ({sma_short_period} SMA > {sma_long_period} SMA).")
        elif latest_sma_short < latest_sma_long or latest_rsi > rsi_overbought:
            signal = "SELL"
            reasoning.append(f"Bearish Trend ({sma_short_period} SMA < {sma_long_period} SMA) OR RSI Overbought ({latest_rsi:.1f} > {rsi_overbought}).")
            
        if latest_rsi < rsi_oversold:
            signal = "BUY"
            reasoning.append(f"RSI Oversold ({latest_rsi:.1f} < {rsi_oversold}), presenting a mean-reversion buying opportunity.")
            
        if not reasoning:
            reasoning.append("Price is within normal technical boundaries. Allocation is balanced.")

        with col2:
            st.markdown(f"### Signal for **{selected_ticker}**")
            
            # Display colored badge for signal
            if signal == "BUY":
                st.success(f"# RECOMMENDATION: {signal}")
            elif signal == "SELL":
                st.error(f"# RECOMMENDATION: {signal}")
            else:
                st.warning(f"# RECOMMENDATION: {signal}")
                
            st.markdown("**Key Indicator Values:**")
            st.write(f"- Current Price: **${latest_close:.2f}**")
            st.write(f"- {sma_short_period}-Day SMA: **${latest_sma_short:.2f}**")
            st.write(f"- {sma_long_period}-Day SMA: **${latest_sma_long:.2f}**")
            st.write(f"- Relative Strength Index (14-Day RSI): **{latest_rsi:.1f}**")
            
            st.markdown("**Rule Engine Output:**")
            for r in reasoning:
                st.write(f"• {r}")

        # Price chart with moving averages
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Close Price', line=dict(color='gray', width=1)))
        fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_Short'], name=f'{sma_short_period}-Day SMA', line=dict(color='blue', width=1.5)))
        fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_Long'], name=f'{sma_long_period}-Day SMA', line=dict(color='orange', width=1.5)))
        fig.update_layout(title=f"{selected_ticker} Price & Moving Averages", xaxis_title="Date", yaxis_title="Price ($)", height=400)
        st.plotly_chart(fig, use_container_width=True)
