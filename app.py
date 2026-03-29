import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
import pickle
import onnxruntime as ort
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="Stock Movement Predictor",
    page_icon="",
    layout="wide"
)


@st.cache_resource
def load_artifacts():
    sess = ort.InferenceSession("stock_ann_model.onnx")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    return sess, scaler, feature_columns

model, scaler, feature_columns = load_artifacts()

def compute_features(ticker, period_years=2):
    end   = datetime.today()
    start = end - timedelta(days=365 * period_years)
    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False
    )
    if df.empty or len(df) < 50:
        return None
    df.columns = [col[0] for col in df.columns]
    df.columns = ["close", "high", "low", "open", "volume"]

    df["rsi"]          = ta.rsi(df["close"], length=14)
    macd               = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macd"]         = macd["MACD_12_26_9"]
    df["macd_signal"]  = macd["MACDs_12_26_9"]
    df["ema9"]         = ta.ema(df["close"], length=9)
    df["ema21"]        = ta.ema(df["close"], length=21)
    df["ema_ratio"]    = df["ema9"] / df["ema21"]
    bbands             = ta.bbands(df["close"], length=20, std=2)
    df["bb_percent"]   = bbands["BBP_20_2.0_2.0"]
    df["atr"]          = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(10).mean()
    stoch              = ta.stoch(df["high"], df["low"], df["close"])
    df["stoch_k"]      = stoch["STOCHk_14_3_3"]
    df["stoch_d"]      = stoch["STOCHd_14_3_3"]
    df["roc"]          = ta.roc(df["close"], length=10)
    df["willr"]        = ta.willr(df["high"], df["low"], df["close"], length=14)
    df["return_1d"]    = df["close"].pct_change(1)
    df["return_3d"]    = df["close"].pct_change(3)
    df["return_5d"]    = df["close"].pct_change(5)
    df = df.dropna()
    return df


def run_backtest(df, signals, initial_capital):
    bt = pd.DataFrame(index=df.index)
    bt["close"]           = df["close"].values
    bt["signal"]          = signals
    bt["market_return"]   = bt["close"].pct_change().fillna(0)
    bt["strategy_return"] = bt["market_return"] * bt["signal"].shift(1).fillna(0)
    bt["market_value"]    = initial_capital * (1 + bt["market_return"]).cumprod()
    bt["strategy_value"]  = initial_capital * (1 + bt["strategy_return"]).cumprod()

    sharpe   = (bt["strategy_return"].mean() /
                bt["strategy_return"].std()) * (252 ** 0.5)
    roll_max = bt["strategy_value"].cummax()
    max_dd   = ((bt["strategy_value"] - roll_max) / roll_max).min()
    wins     = (bt["strategy_return"] > 0).sum()
    total    = (bt["strategy_return"] != 0).sum()
    win_rate = wins / total if total > 0 else 0

    return bt, sharpe, max_dd, win_rate


def show_results(ticker, df, threshold, initial_capital, currency, line_color):
    """
    Renders the full prediction UI block.
    Identical format for both US and Indian markets.
    Only currency symbol and line color differ.
    """

    X        = df[feature_columns]
    X_scaled = scaler.transform(X).astype(np.float32)
    proba    = model.run(None, {"keras_tensor_9": X_scaled})[0].flatten()
    signals  = (proba > threshold).astype(int)

    latest_proba  = proba[-1]
    latest_signal = signals[-1]
    latest_price  = df["close"].iloc[-1]
    latest_date   = df.index[-1].strftime("%b %d, %Y")

    st.subheader(f"Prediction for {ticker} — next trading day")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        direction = "UP" if latest_signal == 1 else "DOWN"
        st.metric("Prediction", direction)
    with c2:
        conf = latest_proba if latest_signal == 1 else 1 - latest_proba
        st.metric("Confidence", f"{conf:.1%}")
    with c3:
        st.metric("Latest Close", f"{currency}{latest_price:.2f}")
    with c4:
        st.metric("As of", latest_date)

    st.progress(
        float(latest_proba),
        text=f"Model probability: {latest_proba:.1%}"
    )
    st.divider()

    bt, sharpe, max_dd, win_rate = run_backtest(df, signals, initial_capital)

    mkt_ret   = (bt["market_value"].iloc[-1]   / initial_capital) - 1
    strat_ret = (bt["strategy_value"].iloc[-1] / initial_capital) - 1

    st.subheader("Backtest Results")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Sharpe Ratio",    f"{sharpe:.2f}")
    m2.metric("Max Drawdown",    f"{max_dd:.2%}")
    m3.metric("Win Rate",        f"{win_rate:.2%}")
    m4.metric("Strategy Return", f"{strat_ret:.2%}")
    m5.metric("Market Return",   f"{mkt_ret:.2%}")

    final_strat = bt["strategy_value"].iloc[-1]
    final_mkt   = bt["market_value"].iloc[-1]
    st.markdown(
        f"**{currency}{initial_capital:,.0f} invested → "
        f"{currency}{final_strat:,.2f} (strategy) vs "
        f"{currency}{final_mkt:,.2f} (buy and hold)**"
    )
    st.divider()

    st.subheader("Equity Curve")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(bt.index, bt["market_value"],
             label="Buy and Hold",
             color="#B0BEC5", linewidth=2, linestyle="--")
    ax1.plot(bt.index, bt["strategy_value"],
             label="ANN Strategy",
             color=line_color, linewidth=2.5)
    ax1.axhline(y=initial_capital, color="gray",
                linestyle=":", linewidth=1, alpha=0.7)
    ax1.fill_between(bt.index, bt["strategy_value"], initial_capital,
                     where=bt["strategy_value"] >= initial_capital,
                     alpha=0.1, color="green")
    ax1.fill_between(bt.index, bt["strategy_value"], initial_capital,
                     where=bt["strategy_value"] < initial_capital,
                     alpha=0.1, color="red")
    ax1.set_title(f"{ticker} — ANN Strategy vs Buy and Hold")
    ax1.set_ylabel(f"Portfolio Value ({currency})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    roll_max = bt["strategy_value"].cummax()
    dd       = (bt["strategy_value"] - roll_max) / roll_max * 100
    ax2.fill_between(bt.index, dd, 0, color="#EF5350", alpha=0.4)
    ax2.plot(bt.index, dd, color="#EF5350", linewidth=1)
    ax2.set_title("Drawdown %")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    st.divider()

    st.subheader("Latest Technical Indicators")

    latest = df.iloc[-1]
    i1, i2, i3 = st.columns(3)

    with i1:
        st.metric("RSI (14)",     f"{latest['rsi']:.1f}")
        st.metric("MACD",         f"{latest['macd']:.3f}")
        st.metric("MACD Signal",  f"{latest['macd_signal']:.3f}")
        st.metric("EMA Ratio",    f"{latest['ema_ratio']:.4f}")
    with i2:
        st.metric("Bollinger %B", f"{latest['bb_percent']:.3f}")
        st.metric("ATR",          f"{latest['atr']:.2f}")
        st.metric("Volume Ratio", f"{latest['volume_ratio']:.2f}")
        st.metric("Williams %R",  f"{latest['willr']:.1f}")
    with i3:
        st.metric("Stoch K",      f"{latest['stoch_k']:.1f}")
        st.metric("Stoch D",      f"{latest['stoch_d']:.1f}")
        st.metric("ROC (10)",     f"{latest['roc']:.2f}%")
        st.metric("1D Return",    f"{latest['return_1d']:.2%}")

    st.caption(
        "Disclaimer: This is a portfolio project for educational "
        "purposes only. Do not use for real trading decisions."
    )


INDIAN_STOCKS = {
    "Reliance Industries"      : "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS",
    "Infosys"                  : "INFY.NS",
    "HDFC Bank"                : "HDFCBANK.NS",
    "Wipro"                    : "WIPRO.NS",
    "Tata Motors"              : "TATAMOTORS.NS",
    "Adani Enterprises"        : "ADANIENT.NS",
    "State Bank of India"      : "SBIN.NS",
    "Bajaj Finance"            : "BAJFINANCE.NS",
    "Asian Paints"             : "ASIANPAINT.NS",
    "Maruti Suzuki"            : "MARUTI.NS",
    "ITC"                      : "ITC.NS",
    "Larsen & Toubro"          : "LT.NS",
    "HCL Technologies"         : "HCLTECH.NS",
    "Sun Pharma"               : "SUNPHARMA.NS",
    "Zomato"                   : "ZOMATO.NS",
    "Nifty 50 Index"           : "^NSEI",
}

# ─────────────────────────────────────────
# App header
# ─────────────────────────────────────────
st.title("Stock Price Movement Predictor")
st.markdown("*ANN-based binary classifier — predicts UP or DOWN for next trading day*")
st.divider()

# ─────────────────────────────────────────
# Unified Sidebar
# ─────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    # Market toggle
    market = st.radio(
        "Select market",
        options=["🇺🇸 US Market", "🇮🇳 Indian Market"],
        horizontal=True
    )
    st.divider()

    if market == "🇺🇸 US Market":
        ticker = st.text_input(
            "Stock Ticker",
            value="AAPL",
            help="Enter any US ticker e.g. AAPL, TSLA, GOOGL, MSFT"
        ).upper().strip()
        st.caption(f"Searching: `{ticker}` on NYSE / NASDAQ")
        currency   = "$"
        line_color = "#2196F3"

    else:
        selected_company = st.selectbox(
            "Select stock",
            options=list(INDIAN_STOCKS.keys()),
            index=0
        )
        ticker = INDIAN_STOCKS[selected_company]
        st.caption(f"NSE ticker: `{ticker}`")

        custom = st.text_input(
            "Or type a custom ticker",
            placeholder="e.g. ZOMATO.NS",
            help="Add .NS for NSE or .BO for BSE"
        )
        if custom:
            ticker = custom.upper().strip()
            st.caption(f"Using custom: `{ticker}`")

        currency   = "₹"
        line_color = "#4CAF50"

    st.divider()

    threshold = st.slider(
        "Prediction threshold",
        min_value=0.30,
        max_value=0.70,
        value=0.40,
        step=0.05,
        help="Lower = more UP predictions. Higher = more selective."
    )

    cap_label = "Backtest capital ($)" if market == "🇺🇸 US Market" else "Backtest capital (₹)"
    cap_min   = 1000    if market == "🇺🇸 US Market" else 10000
    cap_max   = 1000000 if market == "🇺🇸 US Market" else 10000000
    cap_val   = 10000   if market == "🇺🇸 US Market" else 100000
    cap_step  = 1000    if market == "🇺🇸 US Market" else 10000

    initial_capital = st.number_input(
        cap_label,
        min_value=cap_min,
        max_value=cap_max,
        value=cap_val,
        step=cap_step
    )

    predict_btn = st.button(
        "Run Prediction",
        type="primary",
        use_container_width=True
    )

    st.divider()
    st.markdown("**Model info**")
    st.markdown("- Trained on: AAPL, GOOGL, MSFT, TSLA")
    st.markdown("- Features: 13 technical indicators")
    st.markdown("- Architecture: ANN 128→64→32→1")
    st.markdown("- Baseline F1: 49.32%")
    st.markdown("- ANN F1: 68.95%")
    st.markdown("- Backtest Sharpe: 1.51")

if predict_btn:
    with st.spinner(f"Downloading {ticker} data and computing indicators..."):
        df = compute_features(ticker, period_years=2)

    if df is None:
        st.error(
            f"Could not download data for `{ticker}`. "
            f"Please check the ticker symbol and try again."
        )
    else:
        show_results(
            ticker          = ticker,
            df              = df,
            threshold       = threshold,
            initial_capital = initial_capital,
            currency        = currency,
            line_color      = line_color
        )

else:
    st.info("Select a market and stock in the sidebar, then click **Run Prediction**.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🇺🇸 US Market
        - Type any NYSE or NASDAQ ticker
        - Examples: AAPL, TSLA, GOOGL, MSFT, AMZN
        - Capital and prices shown in USD ($)
        - Strategy line shown in blue
        """)
    with col2:
        st.markdown("""
        ### 🇮🇳 Indian Market
        - Select from 17 popular NSE stocks
        - Or type any custom .NS or .BO ticker
        - Capital and prices shown in INR (₹)
        - Strategy line shown in green
        """)

    st.divider()
    st.markdown("""
    ### How it works
    1. Select your market in the sidebar
    2. Choose or type a stock ticker
    3. Adjust prediction threshold and capital
    4. Click **Run Prediction**
    5. The app downloads 2 years of price data
    6. Computes 13 technical indicators
    7. Feeds them into the trained ANN
    8. Shows prediction, backtest results and equity curve

    ### About the model
    | Detail | Value |
    |---|---|
    | Training stocks | AAPL, GOOGL, MSFT, TSLA |
    | Total samples | 4,900 |
    | Features | 13 technical indicators |
    | Architecture | ANN 128 → 64 → 32 → 1 |
    | ANN F1 Score | 68.95% |
    | Baseline F1 | 49.32% (Logistic Regression) |
    | Sharpe Ratio | 1.51 |
    | Max Drawdown | -13.34% |

    ### Disclaimer
    *This is a portfolio project for educational purposes only.
    Do not use this for real trading decisions.*
    """)