import os
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(
    page_title="Weekly ETF Screener & Rule Engine",
    page_icon="",
    layout="wide"
)

st.markdown("""
<style>
div.stButton > button[kind="primary"] {
    background-color: #1E88E5 !important;
    border-color: #1E88E5 !important;
    color: #FFFFFF !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

#
# === EXACT URL & SESSION STATE SYNCHRONIZATION (UPPERCASE ENFORCED) ===
#
def get_url_tickers() -> str:
    """Extracts ticker query parameter matching any casing and returns it converted to UPPERCASE."""
    try:
        params_lower = {str(k).lower(): v for k, v in st.query_params.items()}
        raw_val = ""
        if "tickers" in params_lower:
            raw_val = params_lower["tickers"]
        elif "ticker" in params_lower:
            raw_val = params_lower["ticker"]
        if isinstance(raw_val, list):
            raw_val = raw_val[0] if raw_val else ""
        return str(raw_val).upper().strip() if raw_val else ""
    except Exception:
        pass
    return ""

url_tickers_clean = get_url_tickers()

if url_tickers_clean:
    st.session_state["tickers_input_field"] = url_tickers_clean
elif "tickers_input_field" not in st.session_state:
    st.session_state["tickers_input_field"] = ""

def sync_and_uppercase_params():
    """Callback function: Uppercases any input in the text area and updates URL query params."""
    current_val = st.session_state.get("tickers_input_field", "")
    upper_val = current_val.upper().strip()
    st.session_state["tickers_input_field"] = upper_val
    
    if upper_val:
        st.query_params["tickers"] = upper_val
    else:
        for k in list(st.query_params.keys()):
            if k.lower() in ["tickers", "ticker"]:
                del st.query_params[k]

if "config_df_v2" not in st.session_state:
    st.session_state["config_df_v2"] = pd.DataFrame([
        {"Rule #": "Rule 1", "Rule Name": "Weekly Trend (10/30 EMA)", "Parameters": "10-Wk EMA vs 30-Wk EMA", "My Weight": 15},
        {"Rule #": "Rule 2", "Rule Name": "12-Week Absolute Return", "Parameters": "12 Wks | Min +2.0%", "My Weight": 10},
        {"Rule #": "Rule 3", "Rule Name": "Weekly OBV Trend", "Parameters": "OBV vs 20-Wk SMA", "My Weight": 10},
        {"Rule #": "Rule 4", "Rule Name": "12-Week Relative Strength", "Parameters": "12 Wks | Min +1.0% vs SPY", "My Weight": 15},
        {"Rule #": "Rule 5", "Rule Name": "MACD Line & Hist Expansion", "Parameters": "12, 26, 9 MACD | MACD>Signal & Hist>0", "My Weight": 10},
        {"Rule #": "Rule 6", "Rule Name": "26-Week Max Drawdown", "Parameters": "26 Wks | Max Drawdown ≤ 12.0%", "My Weight": 12},
        {"Rule #": "Rule 7", "Rule Name": "52-Week High Proximity", "Parameters": "52 Wks | Distance ≤ 10.0%", "My Weight": 8},
        {"Rule #": "Rule 8", "Rule Name": "Weekly RSI Band Filter", "Parameters": "14-Wk RSI | 48.0 to 62.0", "My Weight": 5},
        {"Rule #": "Rule 9", "Rule Name": "1-Week Trigger (Hard Gate)", "Parameters": "1 Wk Return ≥ 0.0% (Mandatory for BUY)", "My Weight": 10},
        {"Rule #": "Rule 10", "Rule Name": "12-Week Money Flow Index", "Parameters": "12 Wks | Min Score ≥ 50.0", "My Weight": 5},
    ])

#
# === DATA FETCHING & WEEKLY RESAMPLING ===
#
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_weekly_etf_history(ticker: str) -> pd.DataFrame:
    ticker_clean = ticker.strip().upper()
    try:
        df = yf.download(
            ticker_clean,
            period="2y",
            progress=False,
            auto_adjust=True,
            threads=False,
            ignore_tz=True
        )
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.loc[:, ~df.columns.duplicated()]
            if "Close" in df.columns and len(df) > 60:
                weekly_df = pd.DataFrame()
                weekly_df["Open"] = df["Open"].resample("W-FRI").first()
                weekly_df["High"] = df["High"].resample("W-FRI").max()
                weekly_df["Low"] = df["Low"].resample("W-FRI").min()
                weekly_df["Close"] = df["Close"].resample("W-FRI").last()
                weekly_df["Volume"] = df["Volume"].resample("W-FRI").sum() if "Volume" in df.columns else 0
                return weekly_df.dropna(subset=["Close"]).reset_index()
    except Exception:
        pass
    return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_etf_metadata(ticker: str) -> dict:
    ticker_clean = ticker.strip().upper()
    try:
        t = yf.Ticker(ticker_clean)
        info = t.info if t and hasattr(t, 'info') and isinstance(t.info, dict) else {}
        
        desc = (
            info.get("longBusinessSummary") or 
            info.get("description") or 
            info.get("fundSummary") or 
            f"No summary available for {ticker_clean}."
        )
        geo = (
            info.get("region") or 
            info.get("country") or 
            info.get("market") or 
            ("US" if info.get("exchange") in ["NYQ", "NMS", "NAS", "PCX", "ARC"] else "US / Global")
        )
        cat = (
            info.get("category") or 
            info.get("categoryName") or 
            info.get("quoteType") or 
            info.get("assetClass") or 
            "Equity / ETF"
        )
        return {
            "description": str(desc),
            "region": str(geo).upper() if len(str(geo)) <= 3 else str(geo).title(),
            "category": str(cat).title()
        }
    except Exception:
        return {
            "description": f"Details unavailable for {ticker_clean}.",
            "region": "US / Global",
            "category": "Equity / ETF"
        }

#
# === TECHNICAL HELPER FUNCTIONS ===
#
def calculate_weekly_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1] if not rsi.empty else 50.0
    return float(val) if not pd.isna(val) else 50.0

def derive_action_signal(score: int, pass_1w: bool) -> tuple[str, str, str]:
    if score >= 70:
        if not pass_1w:
            return "HOLD", "warning", "Downgraded to HOLD: Score is strong (≥70), but failed Rule 9 (1-Wk Pullback Gate). Avoid entering into active selloffs."
        return "BUY", "success", "Strong multi-week structural momentum and institutional accumulation."
    elif score >= 45:
        return "HOLD", "warning", "Consolidation or neutral weekly trend. Maintain current positioning."
    else:
        return "SELL", "error", "Weekly technical indicators indicate structural trend decay or drawdown."

#
# === WEEKLY RULE ENGINE ===
#
def evaluate_weekly_rules(ticker: str, df: pd.DataFrame, benchmark_df: pd.DataFrame, params: dict):
    if df.empty or len(df) < 35:
        return None

    ticker_upper = ticker.strip().upper()
    close = pd.Series(df["Close"].values.flatten())
    volume = pd.Series(df["Volume"].values.flatten()) if "Volume" in df.columns else pd.Series(np.zeros(len(df)))

    latest_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])

    # 1. Weekly Trend (10/30 EMA)
    ema_fast = close.ewm(span=params["ema_fast_w"], adjust=False).mean()
    ema_slow = close.ewm(span=params["ema_slow_w"], adjust=False).mean()
    fast_val = float(ema_fast.iloc[-1])
    slow_val = float(ema_slow.iloc[-1])
    rule_ma_passed = fast_val > slow_val
    comm_ma = (
        f"**What:** Short-term weekly moving average ({params['ema_fast_w']}-EMA) vs structural weekly baseline ({params['ema_slow_w']}-EMA).\n\n"
        f"**Why:** Ensures the ETF is in a sustained primary uptrend.\n\n"
        f"**Data:** {params['ema_fast_w']}-Wk EMA (${fast_val:.2f}) vs {params['ema_slow_w']}-Wk EMA (${slow_val:.2f})\n\n"
        f"**Expected:** {params['ema_fast_w']}-WK EMA > {params['ema_slow_w']}-WK EMA."
    )

    # Calculate Structural Stop-Loss Level (10-Wk EMA vs 8% Max Risk Floor)
    stop_loss_price = max(fast_val, latest_close * 0.92)
    stop_loss_pct = ((latest_close - stop_loss_price) / latest_close) * 100

    # 2. 12-Week Absolute Return
    lookback_weeks = min(params["perf_weeks"], len(close) - 1)
    past_close = float(close.iloc[-lookback_weeks])
    period_return_pct = ((latest_close - past_close) / past_close) * 100
    rule_perf_passed = period_return_pct >= params["min_return_pct"]
    comm_perf = (
        f"**What:** 3-month trailing absolute percentage change.\n\n"
        f"**Why:** Confirms underlying price expansion and positive quarterly momentum.\n\n"
        f"**Data:** {lookback_weeks}-Week Return: {period_return_pct:+.2f}%\n\n"
        f"**Expected:** Return ≥ +{params['min_return_pct']}%."
    )

    # 3. Weekly OBV Trend
    price_diff = close.diff()
    direction = np.where(price_diff > 0, 1, np.where(price_diff < 0, -1, 0))
    obv = (volume * direction).cumsum()
    obv_sma20 = obv.rolling(window=20).mean()
    latest_obv = float(obv.iloc[-1]) if not obv.empty else 0.0
    latest_obv_sma = float(obv_sma20.iloc[-1]) if not obv_sma20.empty else 0.0
    rule_obv_passed = latest_obv > latest_obv_sma
    comm_obv = (
        f"**What:** Cumulative On-Balance Volume relative to its 20-week simple moving average.\n\n"
        f"**Why:** Verifies volume is expanding on up weeks, signalling institutional accumulation.\n\n"
        f"**Data:** OBV ({latest_obv:,.0f}) vs 20-Wk SMA ({latest_obv_sma:,.0f})\n\n"
        f"**Expected:** Weekly OBV > 20-Wk OBV SMA."
    )

    # 4. 12-Week Relative Strength vs SPY
    alpha_pct = 0.0
    rule_rs_passed = False
    rs_display = "X Fail"
    if ticker_upper == "SPY":
        rs_display = "N/A"
        comm_rs = (
            f"**What:** Outperformance (Alpha) compared to SPY over {lookback_weeks} weeks.\n\n"
            f"**Why:** Evaluated ticker is SPY (benchmark baseline).\n\n"
            f"**Data:** N/A (Benchmark Baseline)"
        )
    else:
        if not benchmark_df.empty and len(benchmark_df) >= lookback_weeks:
            bench_close = pd.Series(benchmark_df["Close"].values.flatten())
            bench_latest = float(bench_close.iloc[-1])
            bench_past = float(bench_close.iloc[-min(lookback_weeks, len(bench_close) - 1)])
            bench_return = ((bench_latest - bench_past) / bench_past) * 100
            alpha_pct = period_return_pct - bench_return
            rule_rs_passed = alpha_pct >= params["min_alpha_pct"]
            rs_display = " PASS" if rule_rs_passed else "X Fail"
            comm_rs = (
                f"**What:** Excess return (Alpha) generated over SPY over {lookback_weeks} weeks.\n\n"
                f"**Why:** Filters for market leaders outperforming the broad index.\n\n"
                f"**Data:** {lookback_weeks}-Week Alpha vs SPY: {alpha_pct:+.2f}%\n\n"
                f"**Expected:** Alpha ≥ +{params['min_alpha_pct']}%."
            )

    # 5. MACD Line & Histogram Expansion (Tightened Rule)
    ema12 = close.ewm(span=params["macd_fast"], adjust=False).mean()
    ema26 = close.ewm(span=params["macd_slow"], adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=params["macd_signal"], adjust=False).mean()
    histogram = macd_line - signal_line
    
    latest_macd = float(macd_line.iloc[-1])
    latest_sig = float(signal_line.iloc[-1])
    latest_hist = float(histogram.iloc[-1])
    prev_hist = float(histogram.iloc[-2])
    
    # Strictly requires MACD > Signal Line AND Hist > 0 & Expanding
    rule_macd_passed = (latest_macd > latest_sig) and (latest_hist > 0) and (latest_hist >= prev_hist)
    
    macd_status = "Bullish Cross" if latest_macd > latest_sig else "Bearish Signal Cross"
    comm_macd = (
        f"**What:** Evaluates MACD Line vs Signal Line AND Histogram expansion slope.\n\n"
        f"**Why:** Prevents entering positions when short-term momentum has rolled over below signal line.\n\n"
        f"**Data:** MACD ({latest_macd:+.2f}) vs Signal ({latest_sig:+.2f}) [{macd_status}] | Hist: {latest_hist:+.3f}\n\n"
        f"**Expected:** MACD > Signal Line AND Histogram > 0 & Expanding."
    )

    # 6. 26-Week Max Drawdown
    max_dd_pct = 0.0
    rule_dd_passed = False
    if len(close) >= 26:
        tail_26 = close.tail(26)
        rolling_max = tail_26.cummax()
        drawdown = (tail_26 - rolling_max) / rolling_max
        max_dd_pct = abs(float(drawdown.min())) * 100
        rule_dd_passed = max_dd_pct <= params["max_drawdown_pct"]
        comm_dd = (
            f"**What:** Maximum peak-to-trough decline over 26 weeks.\n\n"
            f"**Why:** Filters out highly volatile, crash-prone assets.\n\n"
            f"**Data:** 26-Week Max Drawdown: {max_dd_pct:.2f}%\n\n"
            f"**Expected:** Drawdown ≤ {params['max_drawdown_pct']}%."
        )

    # 7. 52-Week High Proximity
    dist_52w_high_pct = 0.0
    rule_52w_passed = False
    if len(close) >= 52:
        high_52w = float(close.tail(52).max())
        dist_52w_high_pct = ((high_52w - latest_close) / high_52w) * 100
        rule_52w_passed = dist_52w_high_pct <= params["max_dist_52w_pct"]
        comm_52w = (
            f"**What:** Percentage distance from current price to the 52-week high.\n\n"
            f"**Why:** Leading assets trade near high ground; avoids structural laggards.\n\n"
            f"**Data:** Distance from 52-Wk High: {dist_52w_high_pct:.2f}%\n\n"
            f"**Expected:** Distance < {params['max_dist_52w_pct']}%."
        )

    # 8. Weekly RSI Band Filter
    rsi_val = calculate_weekly_rsi(close, period=14)
    rule_rsi_passed = (rsi_val >= params["min_rsi"]) and (rsi_val <= params["max_rsi"])
    comm_rsi = (
        f"**What:** 14-week Relative Strength Index value.\n\n"
        f"**Why:** Avoids buying peak overextended levels.\n\n"
        f"**Data:** 14-Week RSI: {rsi_val:.1f}\n\n"
        f"**Expected:** RSI between {params['min_rsi']} and {params['max_rsi']}."
    )

    # 9. 1-Week Pullback Guardrail (Hard Gate)
    return_1w_pct = ((latest_close - prev_close) / prev_close) * 100
    rule_1w_passed = return_1w_pct >= 0.0
    comm_1w = (
        f"**What:** Price performance of the current week relative to last week's close (MANDATORY HARD GATE).\n\n"
        f"**Why:** Avoids buying into active short-term pullbacks; forces signal to HOLD if negative.\n\n"
        f"**Data:** 1-Week Return: {return_1w_pct:+.2f}%\n\n"
        f"**Expected:** 1-Week Return ≥ 0.0%."
    )

    # 10. 12-Week Money Flow Index
    hist_vol = volume.tail(12)
    hist_close = close.tail(12)
    p_diff = hist_close.diff()
    directional_vol = np.where(p_diff >= 0, hist_vol, -hist_vol)
    net_vol = np.nan_to_num(directional_vol).sum()
    avg_vol = hist_vol.mean()
    flow_score = 50 if avg_vol == 0 else int(min(100, max(0, 50 + (net_vol / (avg_vol * 6)) * 50)))
    rule_flow_passed = flow_score >= params["min_flow_score"]
    comm_flow = (
        f"**What:** Volume-weighted price direction score over trailing 12 weeks.\n\n"
        f"**Why:** Measures net capital inflows vs outflows.\n\n"
        f"**Data:** 12-Week Flow Index: {flow_score}/100\n\n"
        f"**Expected:** Flow Index ≥ {params['min_flow_score']:.0f}."
    )

    total_score = 0
    if rule_ma_passed: total_score += params["weight_ma"]
    if rule_perf_passed: total_score += params["weight_perf"]
    if rule_obv_passed: total_score += params["weight_obv"]
    if rule_rs_passed: total_score += params["weight_rs"]
    if rule_macd_passed: total_score += params["weight_macd"]
    if rule_dd_passed: total_score += params["weight_dd"]
    if rule_52w_passed: total_score += params["weight_52w"]
    if rule_rsi_passed: total_score += params["weight_rsi"]
    if rule_1w_passed: total_score += params["weight_1w"]
    if rule_flow_passed: total_score += params["weight_flow"]

    return {
        "Score": total_score,
        "Close": latest_close,
        "Stop_Loss": stop_loss_price,
        "Stop_Loss_Pct": stop_loss_pct,
        "Pass_MA": rule_ma_passed, "Comm_MA": comm_ma,
        "Pass_Perf": rule_perf_passed, "Comm_Perf": comm_perf,
        "Pass_OBV": rule_obv_passed, "Comm_OBV": comm_obv,
        "Pass_RS": rule_rs_passed, "RS_Display": rs_display, "Comm_RS": comm_rs,
        "Pass_MACD": rule_macd_passed, "Comm_MACD": comm_macd,
        "Pass_DD": rule_dd_passed, "Comm_DD": comm_dd,
        "Pass_52W": rule_52w_passed, "Comm_52W": comm_52w,
        "Pass_RSI": rule_rsi_passed, "Comm_RSI": comm_rsi,
        "Pass_1W": rule_1w_passed, "Comm_1W": comm_1w,
        "Pass_Flow": rule_flow_passed, "Comm_Flow": comm_flow
    }

#
# === MODAL SCORECARD WINDOW (@st.dialog) ===
#
@st.dialog(" Weekly Scorecard Breakdown", width="large")
def show_scorecard_modal(ticker: str, benchmark_df: pd.DataFrame, params: dict):
    st.subheader(f"Weekly Scorecard: {ticker}")
    with st.spinner(f"Analyzing {ticker} on Weekly scale..."):
        df = fetch_weekly_etf_history(ticker)
        meta = fetch_etf_metadata(ticker)
        res = evaluate_weekly_rules(ticker, df, benchmark_df, params)
        
        if res is not None:
            col_meta1, col_meta2 = st.columns([1, 1])
            with col_meta1:
                st.metric("Geographic Area", meta["region"])
            with col_meta2:
                st.metric("ETF Type / Category", meta["category"])
            st.caption(f"**Description:** {meta['description']}")
            st.markdown("---")

            st.write("### Weekly Candlestick Trend")
            fig = go.Figure(data=[
                go.Candlestick(
                    x=df['Date'],
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    increasing_line_color='#00E676',
                    increasing_fillcolor='rgba(0,0,0,0)',
                    decreasing_line_color='#FF5252',
                    decreasing_fillcolor='#FF5252'
                )
            ])
            fig.update_layout(
                xaxis_rangeslider_visible=False,
                margin=dict(l=10, r=10, t=10, b=10),
                height=300,
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("---")

            # Score Summary + Stop Loss Display
            action_label, action_type, action_desc = derive_action_signal(res["Score"], res["Pass_1W"])
            c_metric1, c_metric2, c_metric3 = st.columns([1, 1, 1])
            with c_metric1:
                st.metric(label=f"Weekly Score", value=f"{res['Score']} / 100 Points")
            with c_metric2:
                st.metric(label="Rec. Stop-Loss Level", value=f"${res['Stop_Loss']:.2f}", delta=f"-{res['Stop_Loss_Pct']:.1f}% Risk", delta_color="inverse")
            with c_metric3:
                if action_type == "success":
                    st.success(f"### Signal: {action_label}\n{action_desc}")
                elif action_type == "warning":
                    st.warning(f"### Signal: {action_label}\n{action_desc}")
                else:
                    st.error(f"### Signal: {action_label}\n{action_desc}")
            
            st.markdown("---")

            # Rule Cards
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("1. Weekly Trend", " PASS" if res["Pass_MA"] else "X FAIL",
                          delta=f"{params['weight_ma'] if res['Pass_MA'] else 0} / {params['weight_ma']} pts")
                st.info(res["Comm_MA"])
            with c2:
                st.metric("2. 12W Return", " PASS" if res["Pass_Perf"] else "X FAIL",
                          delta=f"{params['weight_perf'] if res['Pass_Perf'] else 0} / {params['weight_perf']} pts")
                st.info(res["Comm_Perf"])
            with c3:
                st.metric("3. Weekly OBV", " PASS" if res["Pass_OBV"] else "X FAIL",
                          delta=f"{params['weight_obv'] if res['Pass_OBV'] else 0} / {params['weight_obv']} pts")
                st.info(res["Comm_OBV"])

            st.markdown("---")

            c4, c5, c6 = st.columns(3)
            with c4:
                st.metric("4. 12W Rel Strength", res["RS_Display"], 
                          delta=f"{params['weight_rs'] if res['Pass_RS'] else 0} / {params['weight_rs']} pts")
                st.info(res["Comm_RS"])
            with c5:
                st.metric("5. MACD Line & Exp", " PASS" if res["Pass_MACD"] else "X FAIL",
                          delta=f"{params['weight_macd'] if res['Pass_MACD'] else 0} / {params['weight_macd']} pts")
                st.info(res["Comm_MACD"])
            with c6:
                st.metric("6. 26W Drawdown", " PASS" if res["Pass_DD"] else "X FAIL",
                          delta=f"{params['weight_dd'] if res['Pass_DD'] else 0} / {params['weight_dd']} pts")
                st.info(res["Comm_DD"])

            st.markdown("---")

            c7, c8, c9 = st.columns(3)
            with c7:
                st.metric("7. 52W High Prox.", " PASS" if res["Pass_52W"] else "X FAIL",
                          delta=f"{params['weight_52w'] if res['Pass_52W'] else 0} / {params['weight_52w']} pts")
                st.info(res["Comm_52W"])
            with c8:
                st.metric("8. Weekly RSI Band", " PASS" if res["Pass_RSI"] else "X FAIL",
                          delta=f"{params['weight_rsi'] if res['Pass_RSI'] else 0} / {params['weight_rsi']} pts")
                st.info(res["Comm_RSI"])
            with c9:
                st.metric("9. 1W Gate", " PASS" if res["Pass_1W"] else "X FAIL",
                          delta=f"{params['weight_1w'] if res['Pass_1W'] else 0} / {params['weight_1w']} pts")
                st.info(res["Comm_1W"])

            st.markdown("---")

            c10, _ = st.columns([1, 2])
            with c10:
                st.metric("10. Money Flow Index", " PASS" if res["Pass_Flow"] else "X FAIL",
                          delta=f"{params['weight_flow'] if res['Pass_Flow'] else 0} / {params['weight_flow']} pts")
                st.info(res["Comm_Flow"])
        else:
            st.error(f"Could not retrieve historical data for '{ticker}'.")

#
# === SIDEBAR ===
#
with st.sidebar:
    st.header(" Weekly Points Configurator")
    st.caption("Adjust weight allocations across weekly rules (Must sum to 100).")
    
    edited_df = st.data_editor(
        st.session_state["config_df_v2"],
        hide_index=True,
        use_container_width=True,
        column_config={
            "Rule #": st.column_config.TextColumn("Rule #", disabled=True),
            "Rule Name": st.column_config.TextColumn("Rule Name", disabled=True),
            "Parameters": st.column_config.TextColumn("Threshold Parameters", disabled=True),
            "My Weight": st.column_config.NumberColumn("My Weight", min_value=0, max_value=100, step=1, format="%d")
        },
        key="rule_weights_editor_sidebar"
    )
    
    st.session_state["config_df_v2"] = edited_df
    total_raw_points = int(edited_df["My Weight"].sum())
    is_points_valid = (total_raw_points == 100)
    
    st.markdown(f"### **Total Points:** `{total_raw_points}`")
    if is_points_valid:
        st.success(" **Weight total equals 100 pts.**")
    else:
        diff = 100 - total_raw_points
        action_str = f"Add {diff} pts" if diff > 0 else f"Subtract {abs(diff)} pts"
        st.error(f"! Total is **{total_raw_points} pts** ({action_str}).")

    weights = edited_df["My Weight"].tolist()
    RULE_PARAMS = {
        "ema_fast_w": 10, "ema_slow_w": 30, "weight_ma": int(weights[0]),
        "perf_weeks": 12, "min_return_pct": 2.0, "weight_perf": int(weights[1]),
        "weight_obv": int(weights[2]),
        "min_alpha_pct": 1.0, "weight_rs": int(weights[3]),
        "macd_fast": 12, "macd_slow": 26, "macd_signal": 9, "weight_macd": int(weights[4]),
        "max_drawdown_pct": 12.0, "weight_dd": int(weights[5]),
        "max_dist_52w_pct": 10.0, "weight_52w": int(weights[6]),
        "min_rsi": 48.0, "max_rsi": 62.0, "weight_rsi": int(weights[7]),
        "weight_1w": int(weights[8]),
        "min_flow_score": 50.0, "weight_flow": int(weights[9])
    }

#
# === MAIN INTERFACE ===
#
st.title(" Weekly ETF Screener & Analysis")

benchmark_df = fetch_weekly_etf_history("SPY")

if not is_points_valid:
    st.error(f"! Points allocation total is currently {total_raw_points} pts. Please balance weights to 100 in the Sidebar Configurator.")

tickers_input = st.text_area(
    "Tickers to Score:",
    height=120,
    key="tickers_input_field",
    on_change=sync_and_uppercase_params
)

btn_run_screen = st.button(
    "Run Ticker Screen",
    type="primary",
    disabled=not is_points_valid or not tickers_input.strip(),
    use_container_width=True
)

should_run = btn_run_screen or ("auto_ran_on_load" not in st.session_state and bool(tickers_input.strip()))

if should_run:
    st.session_state["auto_ran_on_load"] = True
    active_tickers = [t.strip().upper() for t in tickers_input.replace("\n", ",").split(",") if t.strip()]
    results = []
    
    progress_bar = st.progress(0)
    for idx, ticker in enumerate(active_tickers):
        df = fetch_weekly_etf_history(ticker)
        eval_res = evaluate_weekly_rules(ticker, df, benchmark_df, RULE_PARAMS)
        if eval_res is not None:
            action_sig, _, _ = derive_action_signal(eval_res["Score"], eval_res["Pass_1W"])
            results.append({
                "Ticker": ticker,
                "Action": action_sig,
                "Score": eval_res["Score"],
                "Price_Raw": eval_res['Close'],
                "Price": f"${eval_res['Close']:.2f}",
                "Rec. Stop": f"${eval_res['Stop_Loss']:.2f} (-{eval_res['Stop_Loss_Pct']:.1f}%)",
                "Trend": " Pass" if eval_res["Pass_MA"] else "X Fail",
                "Return": " Pass" if eval_res["Pass_Perf"] else "X Fail",
                "OBV": " Pass" if eval_res["Pass_OBV"] else "X Fail",
                "Rel Strength": eval_res["RS_Display"],
                "MACD Exp": " Pass" if eval_res["Pass_MACD"] else "X Fail",
                "Drawdown": " Pass" if eval_res["Pass_DD"] else "X Fail",
                "52W High": " Pass" if eval_res["Pass_52W"] else "X Fail",
                "RSI Band": " Pass" if eval_res["Pass_RSI"] else "X Fail",
                "1W Direction": " Pass" if eval_res["Pass_1W"] else "X Fail",
                "Flow": " Pass" if eval_res["Pass_Flow"] else "X Fail",
            })
        progress_bar.progress((idx + 1) / len(active_tickers))
    progress_bar.empty()

    if results:
        res_df = pd.DataFrame(results)
        res_df = res_df.sort_values(by="Score", ascending=False).reset_index(drop=True)
        st.session_state["last_screener_df"] = res_df
        st.session_state["last_screener_title"] = "Weekly Scoring Matrix Results"
    else:
        st.warning("Could not retrieve valid historical data for any provided tickers.")

if "last_screener_df" in st.session_state and not st.session_state["last_screener_df"].empty:
    st.subheader(st.session_state.get("last_screener_title", "Weekly Scoring Matrix Results"))
    st.caption(" Select any row to pop open its detailed Scorecard modal window.")
    
    screener_df = st.session_state["last_screener_df"]
    event = st.dataframe(
        screener_df.drop(columns=["Price_Raw"]),
        hide_index=True,
        column_order=[
            "Ticker", "Action", "Score", "Price", "Rec. Stop",
            "Trend", "Return", "OBV", "Rel Strength", "MACD Exp",
            "Drawdown", "52W High", "RSI Band", "1W Direction", "Flow"
        ],
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker"),
            "Action": st.column_config.TextColumn("Signal", help="BUY (≥70 & 1W Pass), HOLD (45-69 or 1W Fail), SELL (<45)"),
            "Score": st.column_config.NumberColumn("Score", format="%d pts"),
            "Rec. Stop": st.column_config.TextColumn("Rec. Stop", help="10-Wk EMA structural trailing stop level")
        },
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row"
    )

    if event and event.selection and event.selection.rows:
        selected_index = event.selection.rows[0]
        selected_ticker = screener_df.iloc[selected_index]["Ticker"]
        show_scorecard_modal(selected_ticker, benchmark_df, RULE_PARAMS)
