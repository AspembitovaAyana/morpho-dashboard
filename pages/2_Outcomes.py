import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

st.header("Bad Debt")

st.image("assets/BadDebt.png", width=700)

st.link_button(
    "Open interactive dashboard in Dune",
    "https://dune.com/queries/4308329"
)

st.header("Liquidations")

st.image("assets/Liquidations.png", width=700)

st.link_button(
    "Open interactive dashboard in Dune",
    "https://dune.com/queries/6689016"
)

st.subheader("Market Configs preview")

DATA_PATH = "data/api/market_configs.csv"  # <-- change to your actual filename

@st.cache_data
def load_contagion_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])


    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df

try:
    dfc = load_contagion_csv(DATA_PATH)
except FileNotFoundError:
    st.error(f"CSV not found: {DATA_PATH}. Update DATA_PATH to your filename in data/api/")
    st.stop()

with st.expander("Filters", expanded=False):
    if "chain_network" in dfc.columns:
        chains = sorted(dfc["chain_network"].dropna().astype(str).unique().tolist())
        sel_chains = st.multiselect("chain_network", chains, default=chains)
        dfc = dfc[dfc["chain_network"].astype(str).isin(sel_chains)]

    if "curator" in dfc.columns:
        curators = sorted(dfc["curator"].dropna().astype(str).unique().tolist())
        sel_curators = st.multiselect("curator", curators, default=curators[:10] if len(curators) > 10 else curators)
        dfc = dfc[dfc["curator"].astype(str).isin(sel_curators)]

    if "date" in dfc.columns and dfc["date"].notna().any():
        dmin = dfc["date"].min().date()
        dmax = dfc["date"].max().date()
        dr = st.date_input("date range", value=(dmin, dmax), min_value=dmin, max_value=dmax)
        if isinstance(dr, tuple) and len(dr) == 2:
            start_dt = pd.to_datetime(dr[0])
            end_dt = pd.to_datetime(dr[1]) + pd.Timedelta(days=1)
            dfc = dfc[(dfc["date"] >= start_dt) & (dfc["date"] < end_dt)]

# Preview controls
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    n = st.number_input("Rows", min_value=10, max_value=2000, value=50, step=10)
with c2:
    show_types = st.checkbox("Show dtypes", value=False)

st.dataframe(dfc.head(int(n)), use_container_width=True)

if show_types:
    st.caption("Column dtypes")
    st.code(str(dfc.dtypes), language="text")

st.caption(f"Rows: {len(dfc):,} | Columns: {len(dfc.columns)}")
