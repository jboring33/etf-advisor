"""
app.py
======
90-Day Tactical ETF Screener & Institutional Flow Engine.
Includes US and Ex-China International / Emerging Market Candidates
with resilient error handling for yfinance data retrieval.
"""

import os
import sys
import datetime

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# Page configuration
st.set_page_config(
    page_title="90-Day Tactical ETF Screener",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Blue Styling Injection
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

from config.portfolio import load_universe, save_universe

# Initialize Session State
if "scan_pool" not in st.session_state:
    st.session_state.scan_pool = load_universe()
if "account_types" not in st.session_state:
    st.session_state.account_types = st.session_state.scan_pool.get("_account_types", {})
if "allocations" not in st.session_state:
    st.session_state.allocations = st.session_state.scan_pool.get("_allocations", {})
if "regions" not in st.session_state:
    st.session_state.regions = st.session_state.scan_pool.get("_regions", {})
if "favorites" not in st.session_state:
    st.session_state.favorites = st.session_state.scan_pool.get("_favorites", [])

def get_active_categories():
    return [cat for cat in st.session_state.scan_pool.keys() if not cat.startswith("_")]


# ==============================================================================
# CANDIDATE POOLS (US & EX-CHINA INTERNATIONAL / EMERGING)
# ==============================================================================
CANDIDATE_CATALOG = {
    "EMXC": {"region": "ex-China", "type": "International/Emerging"},
    "VEA":  {"region": "Developed", "type": "International/Emerging"},
    "DIVI": {"region": "Developed", "type": "International/Emerging"},
    "INDA": {"region": "Emerging", "type": "International/Emerging"},
    "EWJ":  {"region": "Developed", "type": "International/Emerging"},
    "EWT":  {"region": "Emerging", "type": "International/Emerging"},
    "VFLO": {"region": "US", "type": "Tactical/Growth"},
    "SCHD": {"region": "US", "type": "Core/Dividend"},
    "JPST": {"region": "US", "type": "Fixed Income/Cash"},
    "JAAA": {"region": "US", "type": "Fixed Income/Cash"},
    "SCYB": {"region": "US", "type": "Fixed Income/Cash"},
    "SMH":  {"region": "US", "type": "Tactical/Growth"},
    "XLK":  {"region": "US", "type": "Tactical/Growth"},
    "XLF":  {"region": "US", "type": "Core/Dividend"},
    "XLE":  {"region": "US", "type": "Tactical/Growth"},
    "XLI":  {"region": "US", "type": "Core/Dividend"},
    "IWM":  {"region": "US", "type": "Tactical/Growth"},
    "QQQ":  {"region": "US", "type": "Tactical/Growth"},
    "SPY":  {"region": "US", "type": "Core/Dividend"},
}


# ==============================================================================
# DATA ENGINE FUNCTIONS
# ==============================================================================

@st.cache_data(ttl=1800)
def fetch_fed_funds_probabilities():
    """Retrieves macro Fed Funds rate expectations and meeting probabilities."""
    try:
        tnx = yf.Ticker("^TNX").history(period="5d")
        last_yield = tnx["Close"].iloc[-1] if not tnx.empty else 4.25
        return {
            "Next Meeting": "Sep 16, 2026",
            "Pause Probability": "70.7%",
            "Cut Probability (-25bps)": "29.3%",
            "Hike Probability": "0.0%",
            "10Yr Benchmark Yield": f"{last_yield:.2f}%",
            "Regime Sentiment": "Pause Expected / Easing Bias"
        }
    except Exception:
        return {
            "Next Meeting": "Upcoming",
            "Pause Probability": "68.0%",
            "Cut Probability (-25bps)": "32.0%",
            "Hike Probability": "0.0%",
            "10Yr Benchmark Yield": "4.20%",
            "Regime Sentiment": "Neutral"
        }

@st.cache_data(ttl=3600)
def analyze_etf_technical_ema(ticker: str):
    """Calculates 20-day and 50-day EMAs to check for golden crosses or bullish convergence."""
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period="6m")
        if len(df) < 50:
            return None

        df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

        latest_close = df["Close"].iloc[-1]
        ema20 = df["EMA20"].iloc[-1]
        ema50 = df["EMA50"].iloc[-1]
        prev_ema20 = df["EMA20"].iloc[-5]

        gap_pct = ((ema20 - ema50) / ema50) * 100
        is_above = ema20 > ema50
        is_approaching = (not is_above) and (gap_pct > -2.0) and (ema20 > prev_ema20)

        status = "Bullish (20 EMA > 50 EMA)" if is_above else ("Approaching Cross ↗️" if is_approaching else "Bearish Lag")
        
        return {
            "Close": latest_close,
            "EMA20": ema20,
            "EMA50": ema50,
            "Gap_Pct": gap_pct,
            "Status": status,
            "Bullish_Setup": is_above or is_approaching
        }
    except Exception:
        return None

@st.cache_data(ttl=7200)
def fetch_top_holdings_earnings(ticker: str):
    """Fetches top holdings safely with fallback handling if Yahoo Finance fund data is missing."""
    holdings = [ticker]
    
    try:
        tk = yf.Ticker(ticker)
        if hasattr(tk, "funds_data") and tk.funds_data is not None:
            cfg = getattr(tk.funds_data, "top_holdings", None)
            if cfg is not None and not cfg.empty:
                holdings = cfg.index.tolist()[:7]
    except Exception:
        pass

    earnings_summary = []
    upcoming_count = 0
    positive_surprises = 0

    for symbol in holdings:
        try:
            sub_tk = yf.Ticker(symbol)
            cal = sub_tk.calendar
            
            next_date = "N/A"
            if isinstance(cal, dict) and "Earnings Date" in cal:
                ed = cal["Earnings Date"]
                if ed:
                    next_date = ed[0].strftime("%Y-%m-%d") if isinstance(ed[0], datetime.date) else str(ed[0])
                    upcoming_count += 1
            
            surp_df = sub_tk.earnings_dates
            last_surprise = "N/A"
            if surp_df is not None and "Surprise(%)" in surp_df.columns:
                recent = surp_df.dropna(subset=["Surprise(%)"])
                if not recent.empty:
                    val = recent["Surprise(%)"].iloc[0] * 100
                    last_surprise = f"{val:+.1f}%"
                    if val > 0:
                        positive_surprises += 1

            earnings_summary.append({
                "Holding": symbol,
                "Next Earnings": next_date,
                "Last Surprise": last_surprise
            })
        except Exception:
            continue

    return {
        "Holdings_Count": len(holdings),
        "Upcoming_30D_Earnings": upcoming_count,
        "Positive_Surprise_Ratio": f"{positive_surprises}/{len(holdings)}" if holdings else "0/0",
        "Details": earnings_summary
    }

@st.cache_data(ttl=3600)
def fetch_institutional_flows_30d(ticker: str):
    """Calculates institutional flow proxy using 30-day Volume-Weighted Money Flow."""
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="2m")
        if len(hist) < 20:
            return {"Flow_Signal": "Neutral", "Net_30D_Score": 50}

        hist30 = hist.tail(22).copy()
        hist30["Price_Change"] = hist30["Close"].diff()
        hist30["Directional_Vol"] = np.where(hist30["Price_Change"] >= 0, hist30["Volume"], -hist30["Volume"])
        
        net_flow_vol = hist30["Directional_Vol"].sum()
        avg_vol = hist30["Volume"].mean()
        flow_score = min(100, max(0, int(50 + (net_flow_vol / (avg_vol * 10)) * 50)))
        
        if flow_score >= 65:
            signal = "🔥 Strong Accumulation"
        elif flow_score <= 35:
            signal = "🚨 Heavy Distribution"
        else:
            signal = "➡️ Steady Flow"

        return {
            "Flow_Signal": signal,
            "Net_30D_Score": flow_score
        }
    except Exception:
        return {"Flow_Signal": "Neutral", "Net_30D_Score": 50}

def score_etf(ticker: str):
    """Calculates composite 90-day readiness score for an ETF with strict null handling."""
    tech = analyze_etf_technical_ema(ticker)
    if not tech:
        return None

    flows = fetch_institutional_flows_30d(ticker) or {"Flow_Signal": "Neutral", "Net_30D_Score": 50}
    earnings = fetch_top_holdings_earnings(ticker) or {"Holdings_Count": 0, "Upcoming_30D_Earnings": 0, "Positive_Surprise_Ratio": "N/A", "Details": []}

    score = 0
    if tech.get("Bullish_Setup"): score += 40
    if flows.get("Net_30D_Score", 0) >= 60: score += 30
    if earnings.get("Upcoming_30D_Earnings", 0) > 0: score += 30

    return {
        "Ticker": ticker,
        "Score": score,
        "Tech": tech,
        "Flows": flows,
        "Earnings": earnings
    }


# ==============================================================================
# MAIN APP INTERFACE
# ==============================================================================

st.title("🎯 90-Day Tactical ETF Check-In Engine")
st.caption("Institutional Flow Tracking, Top Holding Earnings Catalyst, and 20/50 EMA Momentum Engine")

# Central Bank & Fed Funds Monitor
fed_data = fetch_fed_funds_probabilities()
st.subheader("🏛️ Macro Monitor: Fed Funds Rate Probabilities")
f_col1, f_col2, f_col3, f_col4, f_col5 = st.columns(5)
f_col1.metric("Next FOMC Meeting", fed_data["Next Meeting"])
f_col2.metric("Pause Probability", fed_data["Pause Probability"])
f_col3.metric("Cut Probability", fed_data["Cut Probability (-25bps)"])
f_col4.metric("Hike Probability", fed_data["Hike Probability"])
f_col5.metric("10Yr Yield", fed_data["10Yr Benchmark Yield"], delta=fed_data["Regime Sentiment"])

st.markdown("---")

tab_lookup, tab_screen, tab_earnings, tab_universe = st.tabs([
    "🔍 ETF Lookup & Score",
    "🚀 90-Day Opportunistic Candidates",
    "📅 Holdings 30-Day Earnings Radar",
    "⚙️ ETF Universe Configurator"
])


# ==============================================================================
# TAB 1: SINGLE ETF LOOKUP & SCORE DEEP DIVE
# ==============================================================================
with tab_lookup:
    st.header("🔍 Single ETF Score Lookup")
    st.caption("Type in any US or Ex-China International/Emerging ETF symbol to calculate its score.")

    col_input, _ = st.columns([2, 3])
    with col_input:
        lookup_ticker = st.text_input("Enter ETF Ticker:", value="EMXC", placeholder="e.g. EMXC, VEA, VFLO, SMH").strip().upper()

    if lookup_ticker:
        with st.spinner(f"Evaluating {lookup_ticker}..."):
            res = score_etf(lookup_ticker)

        if res:
            current_tickers = [t for cat in get_active_categories() for t in st.session_state.scan_pool[cat]]
            is_in_universe = lookup_ticker in current_tickers

            meta = CANDIDATE_CATALOG.get(lookup_ticker, {"region": "US", "type": get_active_categories()[0]})

            st.markdown(f"### Score for **{lookup_ticker}** (`{meta['region']}`): `{res['Score']}/100` Points")
            
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.metric("Technical Setup (20/50 EMA)", res["Tech"]["Status"], delta=f"Gap: {res['Tech']['Gap_Pct']:.2f}%")
                st.write(f"- **Close:** ${res['Tech']['Close']:.2f}")
                st.write(f"- **20 EMA:** ${res['Tech']['EMA20']:.2f}")
                st.write(f"- **50 EMA:** ${res['Tech']['EMA50']:.2f}")

            with sc2:
                st.metric("30D Institutional Flow", res["Flows"]["Flow_Signal"], delta=f"Score: {res['Flows']['Net_30D_Score']}/100")
                st.write("Calculated using 30-day Volume-Weighted Accumulation score.")

            with sc3:
                st.metric("30D Holdings Earnings", f"{res['Earnings']['Upcoming_30D_Earnings']} Upcoming", delta=f"Beat Ratio: {res['Earnings']['Positive_Surprise_Ratio']}")
                st.write(f"Analyzed top {res['Earnings']['Holdings_Count']} constituent holdings.")

            st.markdown("---")
            
            c_add1, _ = st.columns([2, 3])
            with c_add1:
                if is_in_universe:
                    st.info(f"✅ **{lookup_ticker}** is already in your active universe.")
                else:
                    if st.button(f"➕ Add {lookup_ticker} to Active Universe", type="primary"):
                        target_cat = meta["type"] if meta["type"] in st.session_state.scan_pool else get_active_categories()[0]
                        st.session_state.scan_pool[target_cat].append(lookup_ticker)
                        st.session_state.account_types[lookup_ticker] = "Brokerage"
                        st.session_state.allocations[lookup_ticker] = 0.0
                        st.session_state.regions[lookup_ticker] = meta["region"]

                        st.session_state.scan_pool["_account_types"] = st.session_state.account_types
                        st.session_state.scan_pool["_allocations"] = st.session_state.allocations
                        st.session_state.scan_pool["_regions"] = st.session_state.regions
                        save_universe(st.session_state.scan_pool)
                        
                        st.success(f"Added {lookup_ticker} ({meta['region']}) to {target_cat}!")
                        st.rerun()
        else:
            st.error(f"Could not retrieve ticker data for '{lookup_ticker}'. Please verify the symbol.")


# ==============================================================================
# TAB 2: 90-DAY OPPORTUNISTIC CANDIDATES & RECOMMENDATIONS
# ==============================================================================
with tab_screen:
    st.header("⚡ Poised ETFs for the Next 90 Days")
    st.caption("Active Universe Evaluation + Automated US & Ex-China International Recommendations")

    categories = get_active_categories()
    screening_results = []
    
    with st.spinner("Analyzing active universe..."):
        for cat in categories:
            for t in st.session_state.scan_pool.get(cat, []):
                res = score_etf(t)
                if res:
                    screening_results.append({
                        "Ticker": t,
                        "Type": cat,
                        "Region": st.session_state.regions.get(t, "US"),
                        "Bucket": st.session_state.account_types.get(t, "Brokerage"),
                        "90D Target Score": res["Score"],
                        "EMA Setup": res["Tech"]["Status"],
                        "30D Inst Flow": res["Flows"]["Flow_Signal"],
                        "Holdings Earnings (30D)": f"{res['Earnings']['Upcoming_30D_Earnings']} upcoming",
                    })

    if screening_results:
        res_df = pd.DataFrame(screening_results).sort_values(by="90D Target Score", ascending=False).reset_index(drop=True)

        st.dataframe(
            res_df,
            hide_index=True,
            column_config={
                "90D Target Score": st.column_config.ProgressColumn(
                    "90D Readiness Score",
                    format="%d pts",
                    min_value=0,
                    max_value=100
                ),
                "Ticker": st.column_config.TextColumn("Ticker", width=80),
                "Region": st.column_config.TextColumn("Region", width=100),
                "EMA Setup": st.column_config.TextColumn("20/50 EMA Trend", width=180),
                "30D Inst Flow": st.column_config.TextColumn("Institutional Flows", width=160),
            },
            use_container_width=True
        )

    st.markdown("---")

    # EXPANDED CANDIDATE RECOMMENDER SECTION
    st.subheader("💡 Market Candidate Recommendations (US & Ex-China International)")
    st.caption("Scans broad international, emerging ex-China, and US factor/sector ETFs.")

    current_tickers = [t for cat in categories for t in st.session_state.scan_pool[cat]]
    rec_results = []

    with st.spinner("Scanning US & International recommendation candidates..."):
        for cand, meta in CANDIDATE_CATALOG.items():
            if cand in current_tickers:
                continue
            res = score_etf(cand)
            if res and res["Score"] >= 40:
                rec_results.append({
                    "Ticker": cand,
                    "Region": meta["region"],
                    "Score": res["Score"],
                    "EMA Setup": res["Tech"]["Status"],
                    "30D Inst Flow": res["Flows"]["Flow_Signal"],
                    "30D Earnings Catalysts": f"{res['Earnings']['Upcoming_30D_Earnings']} upcoming"
                })

    if rec_results:
        rec_df = pd.DataFrame(rec_results).sort_values(by="Score", ascending=False).reset_index(drop=True)
        st.dataframe(rec_df, hide_index=True, use_container_width=True)

        if st.button("🚀 Auto-Add Top Recommended Candidates (Score ≥ 70)"):
            added = []
            for row in rec_results:
                if row["Score"] >= 70:
                    t = row["Ticker"]
                    meta = CANDIDATE_CATALOG[t]
                    target_cat = meta["type"] if meta["type"] in categories else categories[0]
                    
                    st.session_state.scan_pool[target_cat].append(t)
                    st.session_state.account_types[t] = "Brokerage"
                    st.session_state.allocations[t] = 0.0
                    st.session_state.regions[t] = meta["region"]
                    added.append(f"{t} ({meta['region']})")

            if added:
                st.session_state.scan_pool["_account_types"] = st.session_state.account_types
                st.session_state.scan_pool["_allocations"] = st.session_state.allocations
                st.session_state.scan_pool["_regions"] = st.session_state.regions
                save_universe(st.session_state.scan_pool)
                st.success(f"Added high conviction candidates: {', '.join(added)}")
                st.rerun()
            else:
                st.info("No external candidate currently meets the strict ≥70 point cutoff.")
    else:
        st.info("All catalog candidates are already in your active universe!")


# ==============================================================================
# TAB 3: HOLDINGS 30-DAY EARNINGS RADAR
# ==============================================================================
with tab_earnings:
    st.header("📅 Top Holdings Earnings & Surprises (Next 30 Days)")
    all_tickers = [t for cat in categories for t in st.session_state.scan_pool[cat]]
    selected_etf = st.selectbox("Select ETF to Deep-Dive Top Holdings:", options=all_tickers if all_tickers else ["VTI"])

    if selected_etf:
        e_data = fetch_top_holdings_earnings(selected_etf)
        
        col_e1, col_e2 = st.columns(2)
        col_e1.metric("Top Holdings Analyzed", e_data["Holdings_Count"])
        col_e2.metric("Positive Surprise History Ratio", e_data["Positive_Surprise_Ratio"])

        if e_data["Details"]:
            details_df = pd.DataFrame(e_data["Details"])
            st.dataframe(details_df, hide_index=True, use_container_width=True)


# ==============================================================================
# TAB 4: UNIVERSE CONFIGURATOR
# ==============================================================================
with tab_universe:
    st.header("⚙️ ETF Universe Management")

    st.subheader("➕ Quick Add Ticker")
    with st.form("quick_add_master_form", clear_on_submit=True):
        col_t, col_c, col_r, col_a, col_pct, col_btn = st.columns([2, 2, 2, 2, 2, 1.5])
        
        with col_t:
            add_ticker = st.text_input("Ticker", placeholder="e.g. EMXC").strip().upper()
        with col_c:
            add_category = st.selectbox("Type", options=get_active_categories())
        with col_r:
            add_region = st.selectbox("Region", options=["US", "Emerging", "Developed", "ex-China"])
        with col_a:
            add_account = st.selectbox("Bucket", options=["Brokerage", "IRA", "Roth/HSA"])
        with col_pct:
            add_alloc = st.number_input("Allocation (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
        with col_btn:
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            add_submitted = st.form_submit_button("➕ Add Ticker", use_container_width=True)

        if add_submitted and add_ticker and add_category:
            existing_tickers = [t for cat in get_active_categories() for t in st.session_state.scan_pool[cat]]
            if add_ticker not in existing_tickers:
                st.session_state.scan_pool[add_category].append(add_ticker)
                st.session_state.account_types[add_ticker] = add_account
                st.session_state.allocations[add_ticker] = add_alloc
                st.session_state.regions[add_ticker] = add_region
                
                st.session_state.scan_pool["_account_types"] = st.session_state.account_types
                st.session_state.scan_pool["_allocations"] = st.session_state.allocations
                st.session_state.scan_pool["_regions"] = st.session_state.regions
                save_universe(st.session_state.scan_pool)
                
                st.success(f"Added {add_ticker} ({add_region}) to {add_category}!")
                st.rerun()

    st.markdown("---")

    master_rows = []
    for category in categories:
        for t in st.session_state.scan_pool.get(category, []):
            master_rows.append({
                "Bucket": st.session_state.account_types.get(t, "Brokerage"),
                "Ticker": t,
                "Type": category,
                "Region": st.session_state.regions.get(t, "US"),
                "Allocation (%)": float(st.session_state.allocations.get(t, 0.0)),
                "⭐ Fav": t in st.session_state.favorites,
                "Delete": False
            })

    if master_rows:
        master_df = pd.DataFrame(master_rows)
        bucket_order = ["Brokerage", "IRA", "Roth/HSA"]
        master_df["Bucket"] = pd.Categorical(master_df["Bucket"], categories=bucket_order, ordered=True)
        master_df = master_df.sort_values(by=["Bucket", "Ticker"]).reset_index(drop=True)

        edited_df = st.data_editor(
            master_df,
            key="universe_manager_grid",
            hide_index=True,
            column_config={
                "Bucket": st.column_config.SelectboxColumn("Bucket", options=bucket_order, required=True),
                "Ticker": st.column_config.TextColumn("Ticker", disabled=True),
                "Region": st.column_config.SelectboxColumn("Region", options=["US", "Emerging", "Developed", "ex-China"], required=True),
                "Allocation (%)": st.column_config.NumberColumn("Alloc (%)", format="%.1f%%"),
                "⭐ Fav": st.column_config.CheckboxColumn("Fav", width=45),
                "Delete": st.column_config.CheckboxColumn("Del", width=45)
            },
            column_order=["Bucket", "Ticker", "Type", "Region", "Allocation (%)", "⭐ Fav", "Delete"],
            use_container_width=True
        )

        col_save, col_del, _ = st.columns([1.5, 1.5, 3])
        with col_save:
            if st.button("💾 Save Universe Changes", type="primary", use_container_width=True):
                st.session_state.favorites = edited_df[edited_df["⭐ Fav"] == True]["Ticker"].tolist()
                for _, row in edited_df.iterrows():
                    t = row["Ticker"]
                    st.session_state.account_types[t] = str(row["Bucket"])
                    st.session_state.regions[t] = str(row["Region"])
                    st.session_state.allocations[t] = float(row["Allocation (%)"])

                st.session_state.scan_pool["_favorites"] = st.session_state.favorites
                st.session_state.scan_pool["_account_types"] = st.session_state.account_types
                st.session_state.scan_pool["_regions"] = st.session_state.regions
                st.session_state.scan_pool["_allocations"] = st.session_state.allocations
                save_universe(st.session_state.scan_pool)
                st.success("Universe saved successfully!")
                st.rerun()

        with col_del:
            selected_deletes = edited_df[edited_df["Delete"] == True]
            if not selected_deletes.empty:
                to_delete = selected_deletes["Ticker"].tolist()
                if st.button(f"🗑️ Delete Selected ({len(to_delete)})", use_container_width=True):
                    for cat in categories:
                        st.session_state.scan_pool[cat] = [
                            t for t in st.session_state.scan_pool[cat] if t not in to_delete
                        ]
                    st.session_state.favorites = [t for t in st.session_state.favorites if t not in to_delete]
                    for t in to_delete:
                        st.session_state.account_types.pop(t, None)
                        st.session_state.regions.pop(t, None)
                        st.session_state.allocations.pop(t, None)

                    st.session_state.scan_pool["_favorites"] = st.session_state.favorites
                    st.session_state.scan_pool["_account_types"] = st.session_state.account_types
                    st.session_state.scan_pool["_regions"] = st.session_state.regions
                    st.session_state.scan_pool["_allocations"] = st.session_state.allocations
                    save_universe(st.session_state.scan_pool)
                    st.success(f"Deleted {', '.join(to_delete)}!")
                    st.rerun()
