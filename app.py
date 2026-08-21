"""
app.py
======
Modular ETF Rule Configurator & Scoring Engine (10 Rules)
Features:
- Independent "Run Portfolio Screen" and "Run Watchlist Screen" buttons.
- Separate result view for Portfolio vs Watchlist tickers.
- Reliable text-file persistence across reboots.
- Modal detailed scorecard on row click.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

PORTFOLIO_FILE = "saved_portfolio.txt"
WATCHLIST_FILE = "saved_watchlist.txt"

DEFAULT_PORTFOLIO = "SCHD, VFLO, DIVI, JPST, JAAA"
DEFAULT_WATCHLIST = "VEA, SCYB, EMXC"

# Page setup
st.set_page_config(
    page_title="Portfolio ETF Screener & Rule Engine",
    page_icon="⚙️",
    layout="wide"
)

# Custom Styling
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


# ==============================================================================
# PERSISTENCE & SESSION STATE
# ==============================================================================

def load_ticker_file(file_path: str, default_val: str) -> str:
    """Reads saved tickers from a local text file if present."""
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    return content
        except Exception:
            pass
    return default_val

def save_ticker_file(file_path: str, text_content: str):
    """Writes updated ticker text to a local text file to persist reboots."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text_content.strip())
    except Exception:
        pass

# Initialize session state values from disk once
if "portfolio_tickers" not in st.session_state:
    st.session_state["portfolio_tickers"] = load_ticker_file(PORTFOLIO_FILE, DEFAULT_PORTFOLIO)

if "watchlist_tickers" not in st.session_state:
    st.session_state["watchlist_tickers"] = load_ticker_file(WATCHLIST_FILE, DEFAULT_WATCHLIST)

if "config_df_v2" not in st.session_state:
    st.session_state["config_df_v2"] = pd.DataFrame([
        {"Rule #": "Rule 1", "Rule Name": "Moving Average Trend", "My Weight": 12},
        {"Rule #": "Rule 2", "Rule Name": "Absolute Return", "My Weight": 8},
        {"Rule #": "Rule 3", "Rule Name": "Institutional Money Flow", "My Weight": 11},
        {"Rule #": "Rule 4", "Rule Name": "Relative Strength vs SPY", "My Weight": 15},
        {"Rule #": "Rule 5", "Rule Name": "Volume Expansion", "My Weight": 6},
        {"Rule #": "Rule 6", "Rule Name": "Max Trailing Drawdown", "My Weight": 16},
        {"Rule #": "Rule 7", "Rule Name": "52-Week High Proximity", "My Weight": 7},
        {"Rule #": "Rule 8", "Rule Name": "RSI Band Filter", "My Weight": 5},
        {"Rule #": "Rule 9", "Rule Name": "Sharpe Ratio Filter", "My Weight": 15},
        {"Rule #": "Rule 10", "Rule Name": "ATR Volatility Squeeze", "My Weight": 5},
    ])


# ==============================================================================
# DATA FETCHING ENGINE
# ==============================================================================

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_etf_history(ticker: str) -> pd.DataFrame:
    """Fetches 1 year of daily price history using yfinance."""
    ticker_clean = ticker.strip().upper()
    try:
        df = yf.download(
            ticker_clean,
            period="1y",
            progress=False,
            auto_adjust=True,
            threads=False,
            ignore_tz=True
        )
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.loc[:, ~df.columns.duplicated()]
            if "Close" in df.columns and len(df) > 30:
                return df.dropna(subset=["Close"]).reset_index()
    except Exception:
        pass
    return pd.DataFrame()


# ==============================================================================
# TECHNICAL HELPER FUNCTIONS & RULE ENGINE
# ==============================================================================

def calculate_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1] if not rsi.empty else 50.0
    return float(val) if not pd.isna(val) else 50.0

def calculate_atr_ratio(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period + 1 or "High" not in df.columns or "Low" not in df.columns:
        return 0.0
    high = pd.Series(df["High"].values.flatten())
    low = pd.Series(df["Low"].values.flatten())
    close = pd.Series(df["Close"].values.flatten())
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    latest_close = close.iloc[-1]
    return float((atr / latest_close) * 100) if latest_close > 0 else 0.0

def derive_action_signal(score: int) -> tuple[str, str, str]:
    """Generates a Buy, Hold, or Sell signal based on score."""
    if score >= 70:
        return "🟢 BUY", "success", "Strong institutional alignment and multi-factor momentum. Favorable candidate for fresh capital allocation."
    elif score >= 45:
        return "🟡 HOLD", "warning", "Neutral performance or consolidation phase. Maintain existing exposure but await further breakout confirmation before adding."
    else:
        return "🔴 SELL", "error", "Technical metrics indicate lagging momentum, elevated drawdown, or capital outflow. Consider trimming or reallocating."

def evaluate_rules(df: pd.DataFrame, benchmark_df: pd.DataFrame, params: dict):
    if df.empty or len(df) < params["ema_slow"]:
        return None

    close = pd.Series(df["Close"].values.flatten())
    volume = pd.Series(df["Volume"].values.flatten()) if "Volume" in df.columns else pd.Series(np.zeros(len(df)))

    # 1. Trend
    ema_fast = close.ewm(span=params["ema_fast"], adjust=False).mean()
    ema_slow = close.ewm(span=params["ema_slow"], adjust=False).mean()
    latest_close = float(close.iloc[-1])
    fast_val = float(ema_fast.iloc[-1])
    slow_val = float(ema_slow.iloc[-1])
    rule_ma_passed = fast_val > slow_val
    comm_ma = (
        f"**Data:** 20 EMA (${fast_val:.2f}) vs 50 EMA (${slow_val:.2f})\n\n"
        f"**Why it Matters:** Confirms medium-term bullish momentum.\n\n"
        f"**Expected Range:** 20 EMA > 50 EMA."
    )

    # 2. Performance
    lookback_days = min(params["perf_days"], len(close) - 1)
    past_close = float(close.iloc[-lookback_days])
    period_return_pct = ((latest_close - past_close) / past_close) * 100
    rule_perf_passed = period_return_pct >= params["min_return_pct"]
    comm_perf = (
        f"**Data:** {lookback_days}d Return: {period_return_pct:+.2f}%\n\n"
        f"**Why it Matters:** Filters out lagging assets.\n\n"
        f"**Expected Range:** Target ≥ +{params['min_return_pct']}%."
    )

    # 3. Flow
    hist_vol = volume.tail(22)
    hist_close = close.tail(22)
    price_diff = hist_close.diff()
    directional_vol = np.where(price_diff >= 0, hist_vol, -hist_vol)
    net_vol = np.nan_to_num(directional_vol).sum()
    avg_vol = hist_vol.mean()
    flow_score = 50 if avg_vol == 0 else int(min(100, max(0, 50 + (net_vol / (avg_vol * 10)) * 50)))
    rule_flow_passed = flow_score >= params["min_flow_score"]
    comm_flow = (
        f"**Data:** Flow Index: {flow_score}/100\n\n"
        f"**Why it Matters:** Identifies accumulation vs distribution.\n\n"
        f"**Expected Range:** 50-100."
    )

    # 4. Relative Strength
    alpha_pct = 0.0
    rule_rs_passed = False
    if not benchmark_df.empty and len(benchmark_df) >= lookback_days:
        bench_close = pd.Series(benchmark_df["Close"].values.flatten())
        bench_latest = float(bench_close.iloc[-1])
        bench_past = float(bench_close.iloc[-min(lookback_days, len(bench_close) - 1)])
        bench_return = ((bench_latest - bench_past) / bench_past) * 100
        alpha_pct = period_return_pct - bench_return
        rule_rs_passed = alpha_pct >= params["min_alpha_pct"]
    comm_rs = (
        f"**Data:** Alpha vs SPY: {alpha_pct:+.2f}%\n\n"
        f"**Why it Matters:** Outperforming market benchmark.\n\n"
        f"**Expected Range:** ≥ +1.0% alpha."
    )

    # 5. Vol Exp
    vol_ratio = 1.0
    rule_vol_exp_passed = False
    if len(volume) >= 50:
        vol_5d = volume.tail(5).mean()
        vol_50d = volume.tail(50).mean()
        vol_ratio = (vol_5d / vol_50d) if vol_50d > 0 else 1.0
        rule_vol_exp_passed = vol_ratio >= params["min_vol_ratio"]
    comm_vol = (
        f"**Data:** 5d/50d Volume Ratio: {vol_ratio:.2f}x\n\n"
        f"**Why it Matters:** Signals institutional buying spikes.\n\n"
        f"**Expected Range:** ≥ 1.1x."
    )

    # 6. Drawdown
    max_dd_pct = 0.0
    rule_dd_passed = False
    if len(close) >= 60:
        tail_60 = close.tail(60)
        rolling_max = tail_60.cummax()
        drawdown = (tail_60 - rolling_max) / rolling_max
        max_dd_pct = abs(float(drawdown.min())) * 100
        rule_dd_passed = max_dd_pct <= params["max_drawdown_pct"]
    comm_dd = (
        f"**Data:** 60d Max Drawdown: {max_dd_pct:.2f}%\n\n"
        f"**Why it Matters:** Penalizes high risk/volatile drops.\n\n"
        f"**Expected Range:** ≤ 10.0%."
    )

    # 7. 52W High
    dist_52w_high_pct = 0.0
    rule_52w_passed = False
    if len(close) >= 120:
        high_52w = float(close.tail(252).max()) if len(close) >= 252 else float(close.max())
        dist_52w_high_pct = ((high_52w - latest_close) / high_52w) * 100
        rule_52w_passed = dist_52w_high_pct <= params["max_dist_52w_pct"]
    comm_52w = (
        f"**Data:** Distance from 52W High: {dist_52w_high_pct:.2f}%\n\n"
        f"**Why it Matters:** Evaluates proximity to new highs.\n\n"
        f"**Expected Range:** ≤ 8.0%."
    )

    # 8. RSI
    rsi_val = calculate_rsi(close, period=14)
    rule_rsi_passed = (rsi_val >= params["min_rsi"]) and (rsi_val <= params["max_rsi"])
    comm_rsi = (
        f"**Data:** 14d RSI: {rsi_val:.1f}\n\n"
        f"**Why it Matters:** Avoids oversold or overbought extremes.\n\n"
        f"**Expected Range:** 45 to 70."
    )

    # 9. Sharpe
    daily_returns = close.pct_change().dropna()
    ann_return = daily_returns.mean() * 252
    ann_std = daily_returns.std() * np.sqrt(252)
    sharpe_ratio = (ann_return / ann_std) if ann_std > 0 else 0.0
    rule_sharpe_passed = sharpe_ratio >= params["min_sharpe"]
    comm_sharpe = (
        f"**Data:** Ann. Sharpe Ratio: {sharpe_ratio:.2f}\n\n"
        f"**Why it Matters:** Measures risk-adjusted returns.\n\n"
        f"**Expected Range:** ≥ 0.5."
    )

    # 10. ATR
    atr_pct = calculate_atr_ratio(df, period=14)
    rule_atr_passed = atr_pct <= params["max_atr_pct"]
    comm_atr = (
        f"**Data:** ATR % of Price: {atr_pct:.2f}%\n\n"
        f"**Why it Matters:** Measures volatility squeeze.\n\n"
        f"**Expected Range:** ≤ 2.5%."
    )

    # Calculate Total Score
    total_score = 0
    if rule_ma_passed: total_score += params["weight_ma"]
    if rule_perf_passed: total_score += params["weight_perf"]
    if rule_flow_passed: total_score += params["weight_flow"]
    if rule_rs_passed: total_score += params["weight_rs"]
    if rule_vol_exp_passed: total_score += params["weight_vol_exp"]
    if rule_dd_passed: total_score += params["weight_dd"]
    if rule_52w_passed: total_score += params["weight_52w"]
    if rule_rsi_passed: total_score += params["weight_rsi"]
    if rule_sharpe_passed: total_score += params["weight_sharpe"]
    if rule_atr_passed: total_score += params["weight_atr"]

    return {
        "Score": total_score,
        "Close": latest_close,
        "Pass_MA": rule_ma_passed, "Comm_MA": comm_ma,
        "Pass_Perf": rule_perf_passed, "Comm_Perf": comm_perf,
        "Pass_Flow": rule_flow_passed, "Comm_Flow": comm_flow,
        "Pass_RS": rule_rs_passed, "Comm_RS": comm_rs,
        "Pass_VolExp": rule_vol_exp_passed, "Comm_VolExp": comm_vol,
        "Pass_DD": rule_dd_passed, "Comm_DD": comm_dd,
        "Pass_52W": rule_52w_passed, "Comm_52W": comm_52w,
        "Pass_RSI": rule_rsi_passed, "Comm_RSI": comm_rsi,
        "Pass_Sharpe": rule_sharpe_passed, "Comm_Sharpe": comm_sharpe,
        "Pass_ATR": rule_atr_passed, "Comm_ATR": comm_atr
    }


# ==============================================================================
# MODAL SCORECARD WINDOW (@st.dialog)
# ==============================================================================

@st.dialog("🔍 Scorecard Breakdown", width="large")
def show_scorecard_modal(ticker: str, benchmark_df: pd.DataFrame, params: dict):
    """Renders the ETF Scorecard with Buy/Hold/Sell banner inside a popup modal."""
    st.subheader(f"Scorecard: {ticker}")

    with st.spinner(f"Analyzing {ticker}..."):
        df = fetch_etf_history(ticker)
        res = evaluate_rules(df, benchmark_df, params)

    if res is not None:
        action_label, action_type, action_desc = derive_action_signal(res["Score"])

        c_metric1, c_metric2 = st.columns([1, 1])
        with c_metric1:
            st.metric(
                label=f"Composite Score for {ticker}",
                value=f"{res['Score']} / 100 Points"
            )
        with c_metric2:
            if action_type == "success":
                st.success(f"### Signal: {action_label}\n{action_desc}")
            elif action_type == "warning":
                st.warning(f"### Signal: {action_label}\n{action_desc}")
            else:
                st.error(f"### Signal: {action_label}\n{action_desc}")

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            status_ma = "✅ PASS" if res["Pass_MA"] else "❌ FAIL"
            pts_ma = params['weight_ma'] if res['Pass_MA'] else 0
            st.metric("1. Trend Status", status_ma, delta=f"{pts_ma} / {params['weight_ma']} pts")
            st.info(res["Comm_MA"])

        with c2:
            status_perf = "✅ PASS" if res["Pass_Perf"] else "❌ FAIL"
            pts_perf = params['weight_perf'] if res['Pass_Perf'] else 0
            st.metric("2. Return Status", status_perf, delta=f"{pts_perf} / {params['weight_perf']} pts")
            st.info(res["Comm_Perf"])

        with c3:
            status_flow = "✅ PASS" if res["Pass_Flow"] else "❌ FAIL"
            pts_flow = params['weight_flow'] if res['Pass_Flow'] else 0
            st.metric("3. Flow Status", status_flow, delta=f"{pts_flow} / {params['weight_flow']} pts")
            st.info(res["Comm_Flow"])

        st.markdown("---")
        c4, c5, c6 = st.columns(3)
        with c4:
            status_rs = "✅ PASS" if res["Pass_RS"] else "❌ FAIL"
            pts_rs = params['weight_rs'] if res['Pass_RS'] else 0
            st.metric("4. Rel Strength", status_rs, delta=f"{pts_rs} / {params['weight_rs']} pts")
            st.info(res["Comm_RS"])

        with c5:
            status_vol = "✅ PASS" if res["Pass_VolExp"] else "❌ FAIL"
            pts_vol = params['weight_vol_exp'] if res['Pass_VolExp'] else 0
            st.metric("5. Volume Ratio", status_vol, delta=f"{pts_vol} / {params['weight_vol_exp']} pts")
            st.info(res["Comm_VolExp"])

        with c6:
            status_dd = "✅ PASS" if res["Pass_DD"] else "❌ FAIL"
            pts_dd = params['weight_dd'] if res['Pass_DD'] else 0
            st.metric("6. Max Drawdown", status_dd, delta=f"{pts_dd} / {params['weight_dd']} pts")
            st.info(res["Comm_DD"])

        st.markdown("---")
        c7, c8, c9 = st.columns(3)
        with c7:
            status_52w = "✅ PASS" if res["Pass_52W"] else "❌ FAIL"
            pts_52w = params['weight_52w'] if res['Pass_52W'] else 0
            st.metric("7. 52W Proximity", status_52w, delta=f"{pts_52w} / {params['weight_52w']} pts")
            st.info(res["Comm_52W"])

        with c8:
            status_rsi = "✅ PASS" if res["Pass_RSI"] else "❌ FAIL"
            pts_rsi = params['weight_rsi'] if res['Pass_RSI'] else 0
            st.metric("8. RSI Band", status_rsi, delta=f"{pts_rsi} / {params['weight_rsi']} pts")
            st.info(res["Comm_RSI"])

        with c9:
            status_sharpe = "✅ PASS" if res["Pass_Sharpe"] else "❌ FAIL"
            pts_sharpe = params['weight_sharpe'] if res['Pass_Sharpe'] else 0
            st.metric("9. Sharpe Ratio", status_sharpe, delta=f"{pts_sharpe} / {params['weight_sharpe']} pts")
            st.info(res["Comm_Sharpe"])

        st.markdown("---")
        c10, _ = st.columns([1, 2])
        with c10:
            status_atr = "✅ PASS" if res["Pass_ATR"] else "❌ FAIL"
            pts_atr = params['weight_atr'] if res['Pass_ATR'] else 0
            st.metric("10. ATR Volatility", status_atr, delta=f"{pts_atr} / {params['weight_atr']} pts")
            st.info(res["Comm_ATR"])
    else:
        st.error(f"Could not retrieve historical data for '{ticker}'.")


# ==============================================================================
# SIDEBAR: POINTS CONFIGURATOR (⚙️)
# ==============================================================================

with st.sidebar:
    st.header("⚙️ Points Configurator")
    st.caption("Adjust weight allocations across rules (Must sum to 100).")

    edited_df = st.data_editor(
        st.session_state["config_df_v2"],
        hide_index=True,
        use_container_width=True,
        column_config={
            "Rule #": st.column_config.TextColumn("Rule #", disabled=True),
            "Rule Name": st.column_config.TextColumn("Rule Name", disabled=True),
            "My Weight": st.column_config.NumberColumn("My Weight", min_value=0, max_value=100, step=1, format="%d")
        },
        key="rule_weights_editor_sidebar"
    )

    st.session_state["config_df_v2"] = edited_df

    total_raw_points = int(edited_df["My Weight"].sum())
    is_points_valid = (total_raw_points == 100)

    st.markdown(f"### **Total Points:** `{total_raw_points}`")

    if is_points_valid:
        st.success("✅ **Weight total equals 100 pts.**")
    else:
        diff = 100 - total_raw_points
        action_str = f"Add {diff} pts" if diff > 0 else f"Subtract {abs(diff)} pts"
        st.error(f"⚠️ Total is **{total_raw_points} pts** ({action_str}).")


# Extract active weights
weights = edited_df["My Weight"].tolist()
RULE_PARAMS = {
    "ema_fast": 20, "ema_slow": 50, "weight_ma": int(weights[0]),
    "perf_days": 60, "min_return_pct": 2.0, "weight_perf": int(weights[1]),
    "min_flow_score": 50.0, "weight_flow": int(weights[2]),
    "min_alpha_pct": 1.0, "weight_rs": int(weights[3]),
    "min_vol_ratio": 1.1, "weight_vol_exp": int(weights[4]),
    "max_drawdown_pct": 10.0, "weight_dd": int(weights[5]),
    "max_dist_52w_pct": 8.0, "weight_52w": int(weights[6]),
    "min_rsi": 45.0, "max_rsi": 70.0, "weight_rsi": int(weights[7]),
    "min_sharpe": 0.5, "weight_sharpe": int(weights[8]),
    "max_atr_pct": 2.5, "weight_atr": int(weights[9])
}


# ==============================================================================
# MAIN INTERFACE: PORTFOLIO & WATCHLIST SCREENER
# ==============================================================================

st.title("🎯 Portfolio & Watchlist ETF Screener")

benchmark_df = fetch_etf_history("SPY")

if not is_points_valid:
    st.error(f"⚠️ Points allocation total is currently {total_raw_points} pts. Please balance weights to 100 in the ⚙️ Sidebar Configurator.")

col_port, col_watch = st.columns(2)

with col_port:
    portfolio_input = st.text_area(
        "1. Portfolio Tickers (Holdings):",
        value=st.session_state["portfolio_tickers"],
        height=100,
        placeholder="e.g. SCHD, VFLO, DIVI, JPST, JAAA",
        key="portfolio_input_field"
    )
    
    # Save edits to file dynamically
    if portfolio_input != st.session_state["portfolio_tickers"]:
        st.session_state["portfolio_tickers"] = portfolio_input
        save_ticker_file(PORTFOLIO_FILE, portfolio_input)
        
    btn_run_portfolio = st.button(
        "Run Portfolio Screen 💼", 
        type="primary", 
        disabled=not is_points_valid or not portfolio_input.strip(),
        use_container_width=True
    )

with col_watch:
    watchlist_input = st.text_area(
        "2. Watching Tickers (Watchlist):",
        value=st.session_state["watchlist_tickers"],
        height=100,
        placeholder="e.g. VEA, SCYB, EMXC",
        key="watchlist_input_field"
    )
    
    # Save edits to file dynamically
    if watchlist_input != st.session_state["watchlist_tickers"]:
        st.session_state["watchlist_tickers"] = watchlist_input
        save_ticker_file(WATCHLIST_FILE, watchlist_input)

    btn_run_watchlist = st.button(
        "Run Watchlist Screen 👀", 
        type="primary", 
        disabled=not is_points_valid or not watchlist_input.strip(),
        use_container_width=True
    )

# Determine execution trigger
active_tickers = []
screen_title = ""

if btn_run_portfolio:
    active_tickers = [t.strip().upper() for t in portfolio_input.replace("\n", ",").split(",") if t.strip()]
    screen_title = "Portfolio Holdings Scoring Matrix"
elif btn_run_watchlist:
    active_tickers = [t.strip().upper() for t in watchlist_input.replace("\n", ",").split(",") if t.strip()]
    screen_title = "Watchlist Scoring Matrix"

# Execution Routine
if active_tickers:
    results = []
    progress_bar = st.progress(0)
    
    for idx, ticker in enumerate(active_tickers):
        df = fetch_etf_history(ticker)
        eval_res = evaluate_rules(df, benchmark_df, RULE_PARAMS)
        
        if eval_res is not None:
            action_sig, _, _ = derive_action_signal(eval_res["Score"])
            results.append({
                "Ticker": ticker,
                "Action": action_sig,
                "Score": eval_res["Score"],
                "Price_Raw": eval_res['Close'],
                "Price": f"${eval_res['Close']:.2f}",
                "Trend": "✅ Pass" if eval_res["Pass_MA"] else "❌ Fail",
                "Return": "✅ Pass" if eval_res["Pass_Perf"] else "❌ Fail",
                "Flow": "✅ Pass" if eval_res["Pass_Flow"] else "❌ Fail",
                "Rel Strength": "✅ Pass" if eval_res["Pass_RS"] else "❌ Fail",
                "Vol Exp": "✅ Pass" if eval_res["Pass_VolExp"] else "❌ Fail",
                "Drawdown": "✅ Pass" if eval_res["Pass_DD"] else "❌ Fail",
                "52W High": "✅ Pass" if eval_res["Pass_52W"] else "❌ Fail",
                "RSI Band": "✅ Pass" if eval_res["Pass_RSI"] else "❌ Fail",
                "Sharpe": "✅ Pass" if eval_res["Pass_Sharpe"] else "❌ Fail",
                "ATR Squeeze": "✅ Pass" if eval_res["Pass_ATR"] else "❌ Fail",
            })
        
        progress_bar.progress((idx + 1) / len(active_tickers))

    progress_bar.empty()

    if results:
        res_df = pd.DataFrame(results)
        res_df = res_df.sort_values(by="Score", ascending=False).reset_index(drop=True)
        st.session_state["last_screener_df"] = res_df
        st.session_state["last_screener_title"] = screen_title
    else:
        st.warning("Could not retrieve valid historical data for any provided tickers.")


# Display interactive results table
if "last_screener_df" in st.session_state and not st.session_state["last_screener_df"].empty:
    st.subheader(st.session_state.get("last_screener_title", "Scoring Matrix Results"))
    st.caption("💡 Select any row to pop open its detailed Scorecard modal window.")

    screener_df = st.session_state["last_screener_df"]
    
    event = st.dataframe(
        screener_df.drop(columns=["Price_Raw"]),
        hide_index=True,
        column_order=[
            "Ticker", "Action", "Score", "Price", 
            "Trend", "Return", "Flow", "Rel Strength", "Vol Exp", 
            "Drawdown", "52W High", "RSI Band", "Sharpe", "ATR Squeeze"
        ],
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker"),
            "Action": st.column_config.TextColumn("Signal", help="🟢 BUY (≥70), 🟡 HOLD (45-69), 🔴 SELL (<45)"),
            "Score": st.column_config.NumberColumn("Score", format="%d pts")
        },
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row"
    )

    # Open Modal Window on Table Click Selection
    if event and event.selection and event.selection.rows:
        selected_index = event.selection.rows[0]
        selected_ticker = screener_df.iloc[selected_index]["Ticker"]
        show_scorecard_modal(selected_ticker, benchmark_df, RULE_PARAMS)
