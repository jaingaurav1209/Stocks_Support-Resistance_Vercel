import streamlit as st
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator

# =================== Streamlit Config ===================
st.set_page_config(page_title="📈 Stock Strategies Hub", layout="wide")
st.markdown("""
<style>
.main {background-color: #f8f9fa;}
h1,h2,h3 { color: #2c3e50;}
.stButton>button {background-color: #2ecc71; color: white; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# =================== Helper Functions ===================
def get_country_suffix(country):
    country = country.lower()
    if country == "india": return ".NS"
    elif country == "australia": return ".AX"
    return ""

def get_close_series(df):
    close_series = df['Close']
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:, 0]
    return close_series

# =================== Strategies ===================
def buy_and_hold_strategy(symbol, years, country):
    suffix = get_country_suffix(country)
    if not symbol.endswith((".NS", ".AX")):
        symbol += suffix
    df = yf.download(symbol, period=f"{years}y", interval="1d", auto_adjust=True)
    if df.empty: return None
    df['Market Return'] = df['Close'].pct_change()
    df['Cumulative Market Return'] = (1 + df['Market Return']).cumprod() - 1
    return df

def moving_average_crossover_strategy(symbol, years, short_window, long_window, country):
    suffix = get_country_suffix(country)
    if not symbol.endswith((".NS", ".AX")):
        symbol += suffix
    df = yf.download(symbol, period=f"{years}y", interval="1d", auto_adjust=True)
    if df.empty: return None
    df['MA_short'] = df['Close'].rolling(window=short_window).mean()
    df['MA_long'] = df['Close'].rolling(window=long_window).mean()
    df['Signal'] = 0
    df.loc[df['MA_short'] > df['MA_long'], 'Signal'] = 1
    df.loc[df['MA_short'] < df['MA_long'], 'Signal'] = -1
    df['Market Return'] = df['Close'].pct_change()
    df['Strategy Return'] = df['Market Return'] * df['Signal'].shift(1)
    df['Cumulative Market Return'] = (1 + df['Market Return']).cumprod() - 1
    df['Cumulative Strategy Return'] = (1 + df['Strategy Return']).cumprod() - 1
    return df

def rsi_ma_stoploss_strategy(symbol, years, invest, short_ma, long_ma,
                             rsi_lower, rsi_upper, stoploss_pct, country):
    suffix = get_country_suffix(country)
    if not symbol.endswith((".NS", ".AX")):
        symbol += suffix
    df = yf.download(symbol, period=f"{years}y", interval="1d", auto_adjust=True)
    if df.empty: return None, []
    close_series = get_close_series(df)
    df['RSI'] = RSIIndicator(close_series, 14).rsi()
    df['SMA_short'] = SMAIndicator(close_series, short_ma).sma_indicator()
    df['SMA_long'] = SMAIndicator(close_series, long_ma).sma_indicator()
    df['Signal'] = 0
    df.loc[(df['RSI'] > rsi_lower) & (df['SMA_short'] > df['SMA_long']), 'Signal'] = 1
    position = 0; entry_price = 0; trades = 0
    positions_list = []; trade_log = []
    for i in range(len(df)):
        price = close_series.iloc[i]; date = df.index[i]
        if position == 0 and df['Signal'].iloc[i] == 1:
            position, entry_price, trades = 1, price, trades+1
            trade_log.append([date.date(), "Buy", price])
        elif position == 1:
            if price <= entry_price * (1 - stoploss_pct):
                position, trades = 0, trades+1
                trade_log.append([date.date(), "Sell (Stoploss)", price])
            elif df['SMA_short'].iloc[i] < df['SMA_long'].iloc[i] or df['RSI'].iloc[i] < rsi_upper:
                position, trades = 0, trades+1
                trade_log.append([date.date(), "Sell (Trend Reversal)", price])
        positions_list.append(position)
    df['Position'] = positions_list
    df['Market Return'] = close_series.pct_change()
    df['Strategy Return'] = df['Market Return'] * df['Position'].shift(1).fillna(0)
    df['Portfolio Value'] = invest * (1 + df['Strategy Return']).cumprod()
    return df, trade_log

def rsi_ma_stoploss_backtest(symbol, years, invest, short_ma, long_ma,
                             rsi_lower, rsi_upper, stoploss_pct, country):
    suffix = get_country_suffix(country)
    if not symbol.endswith((".NS", ".AX")):
        symbol += suffix
    df = yf.download(symbol, period=f"{years}y", interval="1d", auto_adjust=True)
    if df.empty: return None
    close_series = get_close_series(df)
    df['RSI'] = RSIIndicator(close_series, 14).rsi()
    df['SMA_short'] = SMAIndicator(close_series, short_ma).sma_indicator()
    df['SMA_long'] = SMAIndicator(close_series, long_ma).sma_indicator()
    df['Signal'] = 0
    df.loc[(df['RSI'] > rsi_lower) & (df['SMA_short'] > df['SMA_long']), 'Signal'] = 1
    position = 0; entry_price = 0; trades = 0; positions_list = []
    for i in range(len(df)):
        price = close_series.iloc[i]
        if position == 0 and df['Signal'].iloc[i] == 1:
            position, entry_price, trades = 1, price, trades+1
        elif position == 1:
            if price <= entry_price * (1 - stoploss_pct):
                position, trades = 0, trades+1
            elif df['SMA_short'].iloc[i] < df['SMA_long'].iloc[i] or df['RSI'].iloc[i] < rsi_upper:
                position, trades = 0, trades+1
        positions_list.append(position)
    df['Position'] = positions_list
    df['Market Return'] = close_series.pct_change()
    df['Strategy Return'] = df['Market Return'] * df['Position'].shift(1).fillna(0)
    df['Portfolio Value'] = invest * (1 + df['Strategy Return']).cumprod()
    last_action_idx = df.index[(df['Position'] != df['Position'].shift(1).fillna(0))].max()
    last_action_price = close_series.loc[last_action_idx] if pd.notnull(last_action_idx) else "NA"
    return {
        'Ticker': symbol,
        'Final Portfolio Value': round(df['Portfolio Value'].iloc[-1], 2),
        'Total Return (%)': round((df['Portfolio Value'].iloc[-1] / invest - 1) * 100, 2),
        'Trades Executed': trades,
        'Last Action Date': str(last_action_idx.date()) if pd.notnull(last_action_idx) else "No trades",
        'Last Action Price': round(last_action_price, 2) if last_action_price != "NA" else "NA"
    }

# =================== Streamlit UI ===================
st.title("📊 Stock Strategies Hub")

choice = st.sidebar.selectbox("Select Strategy", 
    ["Buy & Hold", "Moving Average Crossover", "RSI+SMA+Stoploss (Single)", "RSI+SMA+Stoploss (Multi)"])

if choice == "Buy & Hold":
    col1, col2, col3 = st.columns(3)
    with col1:
        symbol = st.text_input("Symbol", "RELIANCE")
    with col2:
        country = st.selectbox("Country", ["India", "Australia", "US"])
    with col3:
        years = st.number_input("Years", 1, 20, 3)  # ✅ Fixed indentation
    if st.button("Run Strategy"):
        df = buy_and_hold_strategy(symbol, years, country)
        if df is not None:
            st.line_chart(df['Cumulative Market Return'])

elif choice == "Moving Average Crossover":
    col1, col2, col3 = st.columns(3)
    with col1:
        symbol = st.text_input("Symbol", "RELIANCE")
    with col2:
        country = st.selectbox("Country", ["India", "Australia", "US"])
    with col3:
        years = st.number_input("Years", 1, 20, 3)
    s_win = st.number_input("Short MA", 5, 100, 20)
    l_win = st.number_input("Long MA", 10, 200, 50)
    if st.button("Run Strategy"):
        df = moving_average_crossover_strategy(symbol, years, s_win, l_win, country)
        if df is not None:
            st.line_chart(df[['Cumulative Market Return', 'Cumulative Strategy Return']])

elif choice == "RSI+SMA+Stoploss (Single)":
    symbol = st.text_input("Symbol", "RELIANCE")
    country = st.selectbox("Country", ["India", "Australia", "US"])
    years = st.number_input("Years", 1, 20, 3)
    invest = st.number_input("Investment", 1000, 10000000, 100000)
    if st.button("Run Strategy"):
        df, log = rsi_ma_stoploss_strategy(symbol, years, invest, 20, 50, 30, 70, 0.01, country)
        if df is not None:
            st.write(pd.DataFrame(log, columns=["Date", "Action", "Price"]))
            st.line_chart(df['Portfolio Value'])

elif choice == "RSI+SMA+Stoploss (Multi)":
    symbols = st.text_input("Symbols comma-separated", "RELIANCE,TCS")
    country = st.selectbox("Country", ["India", "Australia", "US"])
    years = st.number_input("Years", 1, 20, 3)
    invest = st.number_input("Investment", 1000, 10000000, 100000)
    if st.button("Run Backtest"):
        results = []
        for s in symbols.split(","):
            res = rsi_ma_stoploss_backtest(s.strip().upper(), years, invest, 20, 50, 30, 70, 0.01, country)
            if res:
                results.append(res)
        if results:
            st.dataframe(pd.DataFrame(results))
