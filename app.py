app.py
======
Modular ETF Rule Configurator & Scoring Engine (10 Rules)
Features:
- Streamlined 3-column Points Configurator [Rule #, Rule Name, My Weight].
- Delta tracking (+/- pts) in Batch Universe Screener & Single Symbol Scorecard.
- Historical snapshots saved quietly under the hood to calculate run comparisons.
"""

import os
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

SNAPSHOT_FILE = "weekly_runs.csv"

# Page setup
st.set_page_config(
    page_title="Custom ETF Screener & Rule Engine",
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
# SESSION STATE & HISTORICAL STORAGE ENGINE
# ==============================================================================

if "user_tickers" not in st.session_state:
    st.session_state["user_tickers"] = "VFLO, SCHD, SCYB, JPST, JAAA, VEA, DIVI, EMXC, SMH, XLK, QQQ, SPY"

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

def save_run_snapshot(results: list):
    """Saves or updates today's run snapshot quietly in local storage."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    records = []
    for r in results:
        records.append({
            "Run_Date": today_str,
            "Ticker": r["Ticker"],
            "Total_Score": r["Total Score"],
            "Price": r["Price_Raw"]
        })
    
    new_df = pd.DataFrame(records)
    if os.path.exists(SNAPSHOT_FILE):
        try:
            existing_df = pd.read_csv(SNAPSHOT_FILE)
            existing_df = existing_df[existing_df["Run_Date"] != today_str]
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_csv(SNAPSHOT_FILE, index=False)
        except Exception:
            new_df.to_csv(SNAPSHOT_FILE, index=False)
    else:
        new_df.to_csv(SNAPSHOT_FILE, index=False)

def get_previous_run_data() -> pd.DataFrame:
    """Retrieves the most recent prior run before today."""
    if not os.path.exists(SNAPSHOT_FILE):
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(SNAPSHOT_FILE)
        today_str = datetime.now().strftime("%Y-%m-%d")
        prior_df = df[df["Run_Date"] != today_str]
        
        if prior_df.empty:
            return pd.DataFrame()
        
        latest_prior_date = prior_df["Run_Date"].max()
        return prior_df[prior_df["Run_Date"] == latest_prior_date].set_index("Ticker")
    except Exception:
        return pd.DataFrame()


# ==============================================================================
# DATA FETCHING ENGINE
# ==============================================================================

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_etf_history(ticker: str) -> pd.DataFrame:
    """Fetches 1 year of daily price history safely using yfinance."""
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
    comm_ma = f"20 EMA (${fast_val:.2f}) > 50 EMA (${slow_val:.2f})" if rule_ma_passed else f"20 EMA (${fast_val:.2f}) ≤ 50 EMA (${slow_val:.2f})"

    # 2. Performance
    lookback_days = min(params["perf_days"], len(close) - 1)
    past_close = float(close.iloc[-lookback_days])
    period_return_pct = ((latest_close - past_close) / past_close) * 100
    rule_perf_passed = period_return_pct >= params["min_return_pct"]
    comm_perf = f"{lookback_days}d Return: {period_return_pct:+.2f}% (Min: {params['min_return_pct']}%)"

    # 3. Flow
    hist_vol = volume.tail(22)
    hist_close = close.tail(22)
    price_diff = hist_close.diff()
    directional_vol = np.where(price_diff >= 0, hist_vol, -hist_vol)
    net_vol = np.nan_to_num(directional_vol).sum()
    avg_vol = hist_vol.mean()
    flow_score = 50 if avg_vol == 0 else int(min(100, max(0, 50 + (net_vol / (avg_vol * 10)) * 50)))
    rule_flow_passed = flow_score >= params["min_flow_score"]
    comm_flow = f"Flow Index: {flow_score}/100 (Min: {params['min_flow_score']})"

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
    comm_rs = f"Alpha vs SPY: {alpha_pct:+.2f}% (Min: {params['min_alpha_pct']}%)"

    # 5. Vol Exp
    vol_ratio = 1.0
    rule_vol_exp_passed = False
    if len(volume) >= 50:
        vol_5d = volume.tail(5).mean()
        vol_50d = volume.tail(50).mean()
        vol_ratio = (vol_5d / vol_50d) if vol_50d > 0 else 1.0
        rule_vol_exp_passed = vol_ratio >= params["min_vol_ratio"]
    comm_vol = f"5d/50d Vol Ratio: {vol_ratio:.2f}x (Min: {params['min_vol_ratio']}x)"

    # 6. Drawdown
    max_dd_pct = 0.0
    rule_dd_passed = False
    if len(close) >= 60:
        tail_60 = close.tail(60)
        rolling_max = tail_60.cummax()
        drawdown = (tail_60 - rolling_max) / rolling_max
        max_dd_pct = abs(float(drawdown.min())) * 100
        rule_dd_passed = max_dd_pct <= params["max_drawdown_pct"]
    comm_dd = f"60d Max Drawdown: {max_dd_pct:.2f}% (Max Limit: {params['max_drawdown_pct']}%)"

    # 7. 52W High
    dist_52w_high_pct = 0.0
    rule_52w_passed = False
    if len(close) >= 120:
        high_52w = float(close.tail(252).max()) if len(close) >= 252 else float(close.max())
        dist_52w_high_pct = ((high_52w - latest_close) / high_52w) * 100
        rule_52w_passed = dist_52w_high_pct <= params["max_dist_52w_pct"]
    comm_52w = f"Dist from 52W High: {dist_52w_high_pct:.2f}% (Max: {params['max_dist_52w_pct']}%)"

    # 8. RSI
    rsi_val = calculate_rsi(close, period=14)
    rule_rsi_passed = (rsi_val >= params["min_rsi"]) and (rsi_val <= params["max_rsi"])
    comm_rsi = f"14d RSI: {rsi_val:.1f} (Target Band: {params['min_rsi']}-{params['max_rsi']})"

    # 9. Sharpe
    daily_returns = close.pct_change().dropna()
    ann_return = daily_returns.mean() * 252
    ann_std = daily_returns.std() * np.sqrt(252)
    sharpe_ratio = (ann_return / ann_std) if ann_std > 0 else 0.0
    rule_sharpe_passed = sharpe_ratio >= params["min_sharpe"]
    comm_sharpe = f"Ann. Sharpe Ratio: {sharpe_ratio:.2f} (Min: {params['min_sharpe']})"

    # 10. ATR
    atr_pct = calculate_atr_ratio(df, period=14)
    rule_atr_passed = atr_pct <= params["max_atr_pct"]
    comm_atr = f"ATR % of Price: {atr_pct:.2f}% (Max: {params['max_atr_pct']}%)"

    # Calculate Total
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
# MAIN APPLICATION INTERFACE
# ==============================================================================

st.title("🎯 Custom ETF Screener & Scoring Engine")

benchmark_df = fetch_etf_history("SPY")

# Navigation Tabs (Historical tracker UI removed, functionality preserved under hood)
tab_config, tab_screen, tab_single = st.tabs([
    "⚙️ Points Configurator",
    "📊 Batch Universe Screener",
    "🔍 Single Symbol Scorecard"
])


# ==============================================================================
# TAB 1: POINTS CONFIGURATOR
# ==============================================================================
with tab_config:
    st.subheader("Rule Weight Allocation")
    
    edited_df = st.data_editor(
        st.session_state["config_df_v2"],
        hide_index=True,
        use_container_width=False,
        width=500,
        column_config={
            "Rule #": st.column_config.TextColumn("Rule #", disabled=True),
            "Rule Name": st.column_config.TextColumn("Rule Name", disabled=True),
            "My Weight": st.column_config.NumberColumn("My Weight", min_value=0, max_value=100, step=1, format="%d")
        },
        key="rule_weights_editor_v2"
    )

    st.session_state["config_df_v2"] = edited_df

    total_raw_points = int(edited_df["My Weight"].sum())
    is_points_valid = (total_raw_points == 100)

    st.markdown(f"### **Total:** `{total_raw_points}`")

    if is_points_valid:
        st.success("✅ **Total weight equals 100 points.**")
    else:
        diff = 100 - total_raw_points
        action_str = f"Add {diff} pts" if diff > 0 else f"Subtract {abs(diff)} pts"
        st.error(f"⚠️ Current total is **{total_raw_points} pts**. Adjust weights to reach **100** ({action_str}).")


# Extract weights directly
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
# TAB 2: BATCH UNIVERSE SCREENER
# ==============================================================================
with tab_screen:
    st.header("Custom Universe Screening")

    if not is_points_valid:
        st.error(f"⚠️ Point allocation total is currently {total_raw_points} pts. Please balance weights to 100.")

    user_input = st.text_area(
        "Enter ETF Tickers (comma or space separated):",
        value=st.session_state["user_tickers"],
        height=100,
        key="ticker_input_field"
    )

    st.session_state["user_tickers"] = user_input
    tickers_list = [t.strip().upper() for t in user_input.replace("\n", ",").split(",") if t.strip()]

    if st.button("Run Universe Screen", type="primary", disabled=not is_points_valid):
        results = []
        progress_bar = st.progress(0)
        
        for idx, ticker in enumerate(tickers_list):
            df = fetch_etf_history(ticker)
            eval_res = evaluate_rules(df, benchmark_df, RULE_PARAMS)
            
            if eval_res is not None:
                results.append({
                    "Ticker": ticker,
                    "Total Score": eval_res["Score"],
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
            
            progress_bar.progress((idx + 1) / len(tickers_list))

        progress_bar.empty()

        if results:
            # Under-the-hood snapshot saving and comparison
            save_run_snapshot(results)
            prior_run = get_previous_run_data()

            res_df = pd.DataFrame(results)
            changes = []
            for _, row in res_df.iterrows():
                t = row["Ticker"]
                if not prior_run.empty and t in prior_run.index:
                    prev_s = prior_run.loc[t, "Total_Score"]
                    diff = row["Total Score"] - prev_s
                    changes.append(f"{diff:+d} pts" if diff != 0 else "0 pts")
                else:
                    changes.append("New")
            
            res_df.insert(2, "vs Prior Run", changes)
            res_df = res_df.sort_values(by="Total Score", ascending=False).reset_index(drop=True)

            st.dataframe(
                res_df.drop(columns=["Price_Raw"]),
                hide_index=True,
                column_config={
                    "Total Score": st.column_config.ProgressColumn(
                        "Total Score", format="%d pts", min_value=0, max_value=100
                    ),
                    "vs Prior Run": st.column_config.TextColumn("vs Prior Run")
                },
                use_container_width=True
            )
        else:
            st.warning("Could not retrieve valid historical data for any provided tickers.")


# ==============================================================================
# TAB 3: SINGLE SYMBOL SCORECARD
# ==============================================================================
with tab_single:
    st.header("Single ETF Rule Breakdown")

    lookup_ticker = st.text_input("Enter Ticker Symbol:", value="", placeholder="e.g. EMXC, VFLO, SCHD").strip().upper()

    if lookup_ticker:
        if not is_points_valid:
            st.warning("⚠️ Points allocation total must equal exactly 100 points.")
        else:
            with st.spinner(f"Fetching and analyzing {lookup_ticker}..."):
                df = fetch_etf_history(lookup_ticker)
                res = evaluate_rules(df, benchmark_df, RULE_PARAMS)

            if res is not None:
                prior_run = get_previous_run_data()
                delta_str = None
                if not prior_run.empty and lookup_ticker in prior_run.index:
                    prev_score = prior_run.loc[lookup_ticker, "Total_Score"]
                    diff = res["Score"] - prev_score
                    delta_str = f"{diff:+d} pts vs previous run"

                st.metric(
                    label=f"Composite Score for {lookup_ticker}",
                    value=f"{res['Score']} / 100 Points",
                    delta=delta_str
                )

                # Metrics with specific quantitative commentary
                st.markdown("---")
                c1, c2, c3 = st.columns(3)
                with c1:
                    status_ma = "✅ PASS" if res["Pass_MA"] else "❌ FAIL"
                    pts_ma = RULE_PARAMS['weight_ma'] if res['Pass_MA'] else 0
                    st.metric("1. Trend Status", status_ma, delta=f"{pts_ma} / {RULE_PARAMS['weight_ma']} pts")
                    st.caption(res["Comm_MA"])

                with c2:
                    status_perf = "✅ PASS" if res["Pass_Perf"] else "❌ FAIL"
                    pts_perf = RULE_PARAMS['weight_perf'] if res['Pass_Perf'] else 0
                    st.metric("2. Return Status", status_perf, delta=f"{pts_perf} / {RULE_PARAMS['weight_perf']} pts")
                    st.caption(res["Comm_Perf"])

                with c3:
                    status_flow = "✅ PASS" if res["Pass_Flow"] else "❌ FAIL"
                    pts_flow = RULE_PARAMS['weight_flow'] if res['Pass_Flow'] else 0
                    st.metric("3. Flow Status", status_flow, delta=f"{pts_flow} / {RULE_PARAMS['weight_flow']} pts")
                    st.caption(res["Comm_Flow"])

                st.markdown("---")
                c4, c5, c6 = st.columns(3)
                with c4:
                    status_rs = "✅ PASS" if res["Pass_RS"] else "❌ FAIL"
                    pts_rs = RULE_PARAMS['weight_rs'] if res['Pass_RS'] else 0
                    st.metric("4. Rel Strength", status_rs, delta=f"{pts_rs} / {RULE_PARAMS['weight_rs']} pts")
                    st.caption(res["Comm_RS"])

                with c5:
                    status_vol = "✅ PASS" if res["Pass_VolExp"] else "❌ FAIL"
                    pts_vol = RULE_PARAMS['weight_vol_exp'] if res['Pass_VolExp'] else 0
                    st.metric("5. Volume Ratio", status_vol, delta=f"{pts_vol} / {RULE_PARAMS['weight_vol_exp']} pts")
                    st.caption(res["Comm_VolExp"])

                with c6:
                    status_dd = "✅ PASS" if res["Pass_DD"] else "❌ FAIL"
                    pts_dd = RULE_PARAMS['weight_dd'] if res['Pass_DD'] else 0
                    st.metric("6. Max Drawdown", status_dd, delta=f"{pts_dd} / {RULE_PARAMS['weight_dd']} pts")
                    st.caption(res["Comm_DD"])

                st.markdown("---")
                c7, c8, c9 = st.columns(3)
                with c7:
                    status_52w = "✅ PASS" if res["Pass_52W"] else "❌ FAIL"
                    pts_52w = RULE_PARAMS['weight_52w'] if res['Pass_52W'] else 0
                    st.metric("7. 52W Proximity", status_52w, delta=f"{pts_52w} / {RULE_PARAMS['weight_52w']} pts")
                    st.caption(res["Comm_52W"])

                with c8:
                    status_rsi = "✅ PASS" if res["Pass_RSI"] else "❌ FAIL"
                    pts_rsi = RULE_PARAMS['weight_rsi'] if res['Pass_RSI'] else 0
                    st.metric("8. RSI Band", status_rsi, delta=f"{pts_rsi} / {RULE_PARAMS['weight_rsi']} pts")
                    st.caption(res["Comm_RSI"])

                with c9:
                    status_sharpe = "✅ PASS" if res["Pass_Sharpe"] else "❌ FAIL"
                    pts_sharpe = RULE_PARAMS['weight_sharpe'] if res['Pass_Sharpe'] else 0
                    st.metric("9. Sharpe Ratio", status_sharpe, delta=f"{pts_sharpe} / {RULE_PARAMS['weight_sharpe']} pts")
                    st.caption(res["Comm_Sharpe"])

                st.markdown("---")
                c10, _ = st.columns([1, 2])
                with c10:
                    status_atr = "✅ PASS" if res["Pass_ATR"] else "❌ FAIL"
                    pts_atr = RULE_PARAMS['weight_atr'] if res['Pass_ATR'] else 0
                    st.metric("10. ATR Volatility", status_atr, delta=f"{pts_atr} / {RULE_PARAMS['weight_atr']} pts")
                    st.caption(res["Comm_ATR"])
            else:
                st.error(f"Could not retrieve sufficient historical data for '{lookup_ticker}'.")
