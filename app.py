import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns, DiscreteAllocation
from pypfopt.discrete_allocation import get_latest_prices
import matplotlib.pyplot as plt
import plotly.express as px
import io

# Optional: sector info for some Indian stocks (extend as needed)
INDIAN_STOCKS = {
    "RELIANCE.NS": "Energy",
    "TCS.NS": "IT",
    "HDFCBANK.NS": "Banking",
    "INFY.NS": "IT",
    "ICICIBANK.NS": "Banking",
    "HINDUNILVR.NS": "FMCG",
    "SBIN.NS": "Banking",
    "BHARTIARTL.NS": "Telecom",
    "KOTAKBANK.NS": "Banking",
    "LT.NS": "Engineering"
}
US_STOCKS = {
    "AAPL": "Tech",
    "MSFT": "Tech",
    "GOOGL": "Tech",
    "AMZN": "Consumer",
    "TSLA": "Auto",
    "SPY": "ETF",
    "VOO": "ETF",
    "QQQ": "ETF",
    "VTI": "ETF",
    "ARKK": "ETF"
}

# --- Helper Functions ---
def get_tooltip(label, tooltip):
    return f"{label} ℹ️" if tooltip else label

def fetch_data(tickers, period="3y"):
    data = yf.download(tickers, period=period, auto_adjust=True)
    # If only one ticker, yfinance returns a Series, convert to DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    # If multi-index columns, select 'Close' or 'Adj Close' if present
    if isinstance(data.columns, pd.MultiIndex):
        if 'Adj Close' in data.columns.get_level_values(0):
            data = data['Adj Close']
        elif 'Close' in data.columns.get_level_values(0):
            data = data['Close']
    return data

# Additional analytics
def calculate_drawdown(prices):
    cum_returns = (1 + prices.pct_change().fillna(0)).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    return drawdown.min().min()

def calculate_cumulative_returns(prices):
    return (prices.iloc[-1] / prices.iloc[0] - 1).to_dict()

def optimize_portfolio(prices, risk_level):
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)
    ef = EfficientFrontier(mu, S)
    if risk_level == 'Low':
        ef.min_volatility()
    elif risk_level == 'Medium':
        ef.efficient_risk(target_risk=0.15)
    else:
        ef.max_sharpe()
    weights = ef.clean_weights()
    perf = ef.portfolio_performance(verbose=False)
    return weights, perf, ef

def get_portfolio_metrics(ef):
    ret, vol, sharpe = ef.portfolio_performance()
    return {
        'Expected Annual Return': f"{ret*100:.2f}%",
        'Annual Volatility': f"{vol*100:.2f}%",
        'Sharpe Ratio': f"{sharpe:.2f}"
    }

def parse_csv(file):
    df = pd.read_csv(file)
    tickers = df['Ticker'].tolist()
    weights = df['Weight'].tolist() if 'Weight' in df.columns else None
    return tickers, weights

# --- Streamlit UI ---
st.set_page_config(page_title="Robo-Advisor: Portfolio Optimization Tool", layout="wide")
st.title("🤖 Robo-Advisor: Portfolio Optimization Tool")

with st.sidebar:
    st.header("User Inputs")
    market = st.selectbox(
        get_tooltip("Market Region", True),
        ["India", "US"],
        index=0,
        help="Choose the market region for your portfolio. Indian stocks use .NS suffix (e.g., RELIANCE.NS)."
    )
    risk_level = st.selectbox(
        get_tooltip("Risk Tolerance", True),
        ["Low", "Medium", "High"],
        help="Risk Tolerance: Your willingness to accept fluctuations in your investment value. Higher risk can mean higher returns but also higher losses."
    )
    investment = st.number_input(
        get_tooltip("Initial Investment (₹/$)", True),
        min_value=1000, value=10000, step=1000,
        help="Initial Investment: The amount of money you want to invest at the start."
    )
    input_method = st.radio(
        get_tooltip("Select Input Method", True),
        ["Dropdown", "Upload CSV"],
        help="Choose to select tickers from a list or upload your own CSV file."
    )
    if input_method == "Dropdown":
        if market == "India":
            all_tickers = list(INDIAN_STOCKS.keys())
            default_tickers = all_tickers[:3]
        else:
            all_tickers = list(US_STOCKS.keys())
            default_tickers = all_tickers[:3]
        tickers = st.multiselect(
            get_tooltip("Select Stocks/ETFs", True),
            all_tickers, default=default_tickers,
            help="Choose stocks or ETFs to include in your portfolio."
        )
        weights = None
    else:
        uploaded_file = st.file_uploader(
            get_tooltip("Upload CSV of Tickers/Weights", True),
            type=["csv"],
            help="Upload a CSV with columns: Ticker, [Weight]. Weight is optional."
        )
        tickers, weights = ([], None)
        if uploaded_file:
            tickers, weights = parse_csv(uploaded_file)

if (input_method == "Dropdown" and tickers) or (input_method == "Upload CSV" and tickers):
    with st.spinner("Fetching data and optimizing portfolio..."):
        prices = fetch_data(tickers)
        if prices.isnull().values.any():
            st.warning("Some tickers may not have enough data. Please check your selection.")
        weights_opt, perf, ef = optimize_portfolio(prices, risk_level)
        metrics = get_portfolio_metrics(ef)
        latest_prices = get_latest_prices(prices)
        da = DiscreteAllocation(weights_opt, latest_prices, total_portfolio_value=investment)
        allocation, leftover = da.lp_portfolio()

        # --- Additional Analytics ---
        drawdown = calculate_drawdown(prices)
        cum_returns = calculate_cumulative_returns(prices)
        returns_table = prices.pct_change().fillna(0).resample('Y').apply(lambda x: (1 + x).prod() - 1)

    # --- Results ---
    st.subheader("Suggested Portfolio Allocation")
    alloc_df = pd.DataFrame(list(weights_opt.items()), columns=["Ticker", "Weight"])
    fig_pie = px.pie(alloc_df, names="Ticker", values="Weight", title="Portfolio Allocation")
    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Expected Return vs Risk")
    st.markdown("**Expected Return:** The average return you might expect per year.\n\n**Risk (Volatility):** How much the value of your portfolio might fluctuate.")
    scatter_df = pd.DataFrame({
        "Expected Return": [perf[0]],
        "Risk (Volatility)": [perf[1]]
    })
    fig_scatter = px.scatter(scatter_df, x="Risk (Volatility)", y="Expected Return", text=["Optimized Portfolio"], size=[20])
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Key Portfolio Metrics")
    for k, v in metrics.items():
        st.markdown(f"**{k}:** {v}")
    st.markdown(f"**Max Drawdown:** {drawdown:.2%}  ", help="Max Drawdown: The largest observed loss from a peak to a trough. Lower is better.")

    st.subheader("Cumulative Returns")
    st.dataframe(pd.DataFrame.from_dict(cum_returns, orient='index', columns=['Cumulative Return']).style.format({'Cumulative Return': '{:.2%}'}))

    st.subheader("Annual Returns Table")
    st.dataframe(returns_table.style.format('{:.2%}'))

    st.subheader("Discrete Allocation (Number of Shares)")
    alloc_table = pd.DataFrame(list(allocation.items()), columns=["Ticker", "Shares"])
    st.dataframe(alloc_table)
    st.markdown(f"**Funds Remaining:** {'₹' if market=='India' else '$'}{leftover:.2f}")

    # --- Sector Breakdown ---
    st.subheader("Sector Breakdown")
    if market == "India":
        sector_map = INDIAN_STOCKS
    else:
        sector_map = US_STOCKS
    alloc_df['Sector'] = alloc_df['Ticker'].map(sector_map)
    sector_df = alloc_df.groupby('Sector')['Weight'].sum().reset_index()
    fig_sector = px.pie(sector_df, names="Sector", values="Weight", title="Sector Allocation")
    st.plotly_chart(fig_sector, use_container_width=True)

    # --- Export CSV ---
    csv = alloc_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Export Optimized Portfolio as CSV",
        data=csv,
        file_name="optimized_portfolio.csv",
        mime="text/csv"
    )

    # --- Stock Info Links ---
    st.subheader("More Information on Stocks")
    for t in tickers:
        if market == "India":
            st.markdown(f"[{t} on NSE](https://www.nseindia.com/get-quotes/equity?symbol={t.replace('.NS','')})")
        else:
            st.markdown(f"[{t} on Yahoo Finance](https://finance.yahoo.com/quote/{t})")
else:
    st.info("Please select or upload at least one ticker to proceed.")

st.markdown("---")
st.caption("This tool is for educational purposes only. Not financial advice.")
