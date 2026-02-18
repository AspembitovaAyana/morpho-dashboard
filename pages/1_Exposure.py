import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

st.set_page_config(layout="wide", page_title="Exposure — Chains")

DATA_PATH = "data/api/chains_TS_long.csv"

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["exposure_supplyUsd"] = pd.to_numeric(df["exposure_supplyUsd"], errors="coerce").fillna(0.0)

    df["chain_network"] = df["chain_network"].astype(str)
    df["collateral_symbol"] = df["collateral_symbol"].astype(str)

    df = df.dropna(subset=["date", "chain_network", "collateral_symbol"])
    return df


def apply_mpl_dark(ax):
    ax.set_facecolor("none")
    ax.figure.set_facecolor("none")

    ax.tick_params(colors="#C9D1D9", labelsize=9)
    ax.title.set_color("#E6E6E6")

    for spine in ax.spines.values():
        spine.set_color("#2B3240")

    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.20)


def chain_total_ts(df_chain: pd.DataFrame, value_col="exposure_supplyUsd") -> pd.DataFrame:
    return (
        df_chain.groupby("date", as_index=False)[value_col]
        .sum()
        .sort_values("date")
    )


def plot_chain_total(df_chain: pd.DataFrame, chain: str, value_col="exposure_supplyUsd", use_roll=False, roll_days=7):
    g = chain_total_ts(df_chain, value_col=value_col)
    if g.empty:
        return

    y = g[value_col].to_numpy()
    if use_roll:
        y = pd.Series(y).rolling(roll_days, min_periods=1).mean().to_numpy()

    fig, ax = plt.subplots(figsize=(6.4, 2.35), dpi=150)

    ax.plot(g["date"].to_numpy(), y)
    ax.fill_between(g["date"].to_numpy(), y, np.zeros(len(y)), alpha=0.10)

    ax.set_title(chain, loc="left", fontsize=10, pad=6)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    apply_mpl_dark(ax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    st.pyplot(fig, use_container_width=True, clear_figure=True)


def plot_chain_by_asset(df_chain: pd.DataFrame, chain: str, value_col="exposure_supplyUsd"):
    # one line per collateral_symbol within this chain
    if df_chain.empty:
        return

    fig, ax = plt.subplots(figsize=(6.4, 2.8), dpi=150)

    for sym, g in df_chain.groupby("collateral_symbol"):
        gg = (
            g.groupby("date", as_index=False)[value_col]
            .sum()
            .sort_values("date")
        )
        ax.plot(gg["date"].to_numpy(), gg[value_col].to_numpy(), label=sym, alpha=0.9)

    ax.set_title(f"{chain} — by asset", loc="left", fontsize=10, pad=6)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    apply_mpl_dark(ax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=False, fontsize=8, ncol=2)

    st.pyplot(fig, use_container_width=True, clear_figure=True)


st.title("Exposure by Chain")
st.caption("Small multiples: total supply exposure (USD) per chain (sum across collaterals).")

df = load_data(DATA_PATH)

st.sidebar.header("Filters")

chains_all = sorted(df["chain_network"].unique().tolist())
selected_chains = st.sidebar.multiselect(
    "Chains",
    chains_all,
    default=chains_all[:3] if len(chains_all) > 3 else chains_all
)

collats_all = sorted(df["collateral_symbol"].unique().tolist())
selected_collats = st.sidebar.multiselect(
    "Collateral",
    collats_all,
    default=collats_all
)

min_d, max_d = df["date"].min(), df["date"].max()
date_range = st.sidebar.date_input(
    "Date range",
    value=(min_d.date(), max_d.date()),
    min_value=min_d.date(),
    max_value=max_d.date(),
)

start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)

with st.sidebar.expander("Smoothing (optional)", expanded=False):
    use_roll = st.checkbox("Show rolling average (total only)", value=False)
    roll_days = st.slider("Window (days)", 2, 30, 7)

cards_per_row = st.sidebar.selectbox("Cards per row", [2, 3, 4], index=1)

f = df[
    df["chain_network"].isin(selected_chains)
    & df["collateral_symbol"].isin(selected_collats)
    & (df["date"] >= start_date)
    & (df["date"] < end_date)
].copy()

if f.empty:
    st.warning("No data after filtering.")
    st.stop()

with st.expander("Data preview", expanded=False):
    st.dataframe(f.sort_values(["chain_network", "date"]).head(50), use_container_width=True)

st.divider()

chains_to_render = sorted(f["chain_network"].unique().tolist())

for i in range(0, len(chains_to_render), cards_per_row):
    row = chains_to_render[i : i + cards_per_row]
    cols = st.columns(cards_per_row)
    for col, chain in zip(cols, row):
        with col:
            with st.container(border=True):
                df_chain = f[f["chain_network"] == chain].copy()

                # Top chart = total
                plot_chain_total(
                    df_chain=df_chain,
                    chain=chain,
                    use_roll=use_roll,
                    roll_days=roll_days
                )

               
                with st.expander("By asset (collateral)", expanded=False):
                    plot_chain_by_asset(df_chain=df_chain, chain=chain)

# =========================================================
# Exposure by Markets 
# =========================================================

st.header("Exposure by Market")
st.caption("Total supply exposure (USD) per market (sum across collaterals).")

MARKETS_PATH = "data/api/markets_ts_long.csv"  # <-- change if needed

EXPOSED_MARKETS = [
    "0xbd1ad3b968f5f0552dbd8cf1989a62881407c5cccf9e49fb3657c8731caf0c1f",
    "0x0f9563442d64ab3bd3bcb27058db0b0d4046a4c46f0acd811dacae9551d2b129",
    "0x9e90aec7d768403dacc9dd0d8320307fda3f980eed4df43e3e52168a1c667709",
    "0x8d009383866dffaac5fe25af684e93f8dd5a98fed1991c298624ecc3a860f39f",
]


@st.cache_data
def load_markets_data(path: str) -> pd.DataFrame:
    dfm = pd.read_csv(path)

    if "Unnamed: 0" in dfm.columns:
        dfm = dfm.drop(columns=["Unnamed: 0"])

    required = ["date", "market_uniqueKey", "supplyAssetsUsd"]
    missing = [c for c in required if c not in dfm.columns]
    if missing:
        raise ValueError(f"Markets file missing columns: {missing}. Columns: {dfm.columns.tolist()}")

    dfm["date"] = pd.to_datetime(dfm["date"], errors="coerce")
    dfm = dfm.dropna(subset=["date"])

    dfm["market_id"] = dfm["market_uniqueKey"].astype(str)
    dfm["exposure_supplyUsd"] = pd.to_numeric(dfm["supplyAssetsUsd"], errors="coerce").fillna(0.0)

    for c in ["chain_network", "collateral_symbol", "chain_id"]:
        if c in dfm.columns:
            dfm[c] = dfm[c].astype(str)

    return dfm


def plot_market_total_card(ts: pd.DataFrame, title: str) -> None:
    ts = ts.sort_values("date")
    if ts.empty:
        st.info("No data")
        return

    fig, ax = plt.subplots(figsize=(6.2, 2.25), dpi=150)
    ax.plot(ts["date"].to_numpy(), ts["exposure_supplyUsd"].to_numpy())
    ax.fill_between(ts["date"].to_numpy(), ts["exposure_supplyUsd"].to_numpy(), 0, alpha=0.10)

    ax.set_title(title, loc="left", fontsize=10, pad=6)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    try:
        apply_mpl_dark(ax)
    except NameError:
        ax.grid(True, linestyle="--", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    st.pyplot(fig, use_container_width=True, clear_figure=True)


dfm = load_markets_data(MARKETS_PATH)

try:
    dfm = dfm[(dfm["date"] >= start_date) & (dfm["date"] < end_date)].copy()
except NameError:
    pass

dfm = dfm[dfm["market_id"].isin(EXPOSED_MARKETS)].copy()

if dfm.empty:
    st.warning("No market exposure data for the selected filters/date range.")
else:
    market_totals = (
        dfm.groupby(["market_id", "date"], as_index=False)["exposure_supplyUsd"]
        .sum()
        .sort_values(["market_id", "date"])
    )

    try:
        cards_per_row = cards_per_row
    except NameError:
        cards_per_row = 3

    markets_to_render = [m for m in EXPOSED_MARKETS if m in set(market_totals["market_id"])]

    for i in range(0, len(markets_to_render), cards_per_row):
        row = markets_to_render[i:i + cards_per_row]
        cols = st.columns(cards_per_row)

        for col, mid in zip(cols, row):
            with col:
                with st.container(border=True):
                    ts = market_totals[market_totals["market_id"] == mid]
                    label = f"{mid[:6]}…{mid[-4:]}"
                    plot_market_total_card(ts, label)

                    if "collateral_symbol" in dfm.columns:
                        with st.expander("Collateral breakdown", expanded=False):
                            fig2, ax2 = plt.subplots(figsize=(8.5, 3.3), dpi=150)

                            sub = dfm[dfm["market_id"] == mid].copy()
                            for sym, gg in sub.groupby("collateral_symbol"):
                                gg = gg.groupby("date", as_index=False)["exposure_supplyUsd"].sum().sort_values("date")
                                ax2.plot(gg["date"].to_numpy(), gg["exposure_supplyUsd"].to_numpy(), label=sym, alpha=0.85)

                            ax2.set_title("By collateral", loc="left", fontsize=10, pad=6)
                            ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
                            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

                            try:
                                apply_mpl_dark(ax2)
                            except NameError:
                                ax2.grid(True, linestyle="--", alpha=0.2)
                                ax2.spines["top"].set_visible(False)
                                ax2.spines["right"].set_visible(False)

                            ax2.legend(frameon=False, ncol=3, fontsize=9)
                            st.pyplot(fig2, use_container_width=True, clear_figure=True)


# =========================================================
# Exposure by Vault (from TS_vaults_long1.csv)
# =========================================================

VAULTS_PATH = "data/api/TS_vaults_long1.csv"  # <-- update path if needed


@st.cache_data
def load_vaults(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    required = ["date", "vault_address", "vault_name", "market_uniqueKey", "vault_supplyUsd"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in vault file: {missing}. Found: {df.columns.tolist()}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    df["vault_address"] = df["vault_address"].astype(str)
    df["vault_name"] = df["vault_name"].astype(str)
    df["market_id"] = df["market_uniqueKey"].astype(str)

    df["vault_supplyUsd"] = pd.to_numeric(df["vault_supplyUsd"], errors="coerce").fillna(0.0)

    return df


def short_hex(x: str) -> str:
    x = str(x)
    return f"{x[:6]}…{x[-4:]}" if len(x) > 14 else x


def plot_vault_card(vdf: pd.DataFrame, vault_addr: str, title: str = None):
    sub = vdf[vdf["vault_address"] == vault_addr].copy()
    if sub.empty:
        st.info("No data")
        return

    total = (
        sub.groupby("date", as_index=False)["vault_supplyUsd"]
           .sum()
           .sort_values("date")
    )

    fig, ax = plt.subplots(figsize=(6.2, 2.25), dpi=150)
    ax.plot(total["date"].to_numpy(), total["vault_supplyUsd"].to_numpy(), alpha=0.95)
    ax.fill_between(total["date"].to_numpy(), total["vault_supplyUsd"].to_numpy(), 0, alpha=0.10)

    ax.set_title(title or short_hex(vault_addr), loc="left", fontsize=10, pad=6)
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.20)

    st.pyplot(fig, use_container_width=True, clear_figure=True)


def market_vault_shares(vdf: pd.DataFrame) -> pd.DataFrame:
    d = vdf.copy()
    totals = (
        d.groupby(["date", "market_id"], as_index=False)["vault_supplyUsd"]
         .sum()
         .rename(columns={"vault_supplyUsd": "market_total_supplyUsd"})
    )
    d = d.merge(totals, on=["date", "market_id"], how="left")
    d["vault_share_pct"] = np.where(
        d["market_total_supplyUsd"] > 0,
        100.0 * d["vault_supplyUsd"] / d["market_total_supplyUsd"],
        0.0
    )
    return d


# =========================================================
# EXPOSURE BY VAULTS 
# =========================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


VAULTS_PATH = "data/api/TS_vaults_long1.csv"


@st.cache_data
def load_vaults(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    required = ["date", "vault_address", "vault_name", "market_uniqueKey", "vault_supplyUsd"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    df["vault_address"] = df["vault_address"].astype(str)
    df["vault_name"] = df["vault_name"].astype(str)
    df["market_id"] = df["market_uniqueKey"].astype(str)
    df["vault_supplyUsd"] = pd.to_numeric(df["vault_supplyUsd"], errors="coerce").fillna(0)

    return df


def short_hex(x: str) -> str:
    return f"{x[:6]}…{x[-4:]}"


def market_vault_shares(vdf: pd.DataFrame) -> pd.DataFrame:
    d = vdf.copy()

    totals = (
        d.groupby(["date", "market_id"], as_index=False)["vault_supplyUsd"]
         .sum()
         .rename(columns={"vault_supplyUsd": "market_total"})
    )

    d = d.merge(totals, on=["date", "market_id"], how="left")
    d["share_pct"] = np.where(d["market_total"] > 0,
                              100 * d["vault_supplyUsd"] / d["market_total"], 0)

    return d


def plot_vault_card(df: pd.DataFrame, addr: str, name: str):
    sub = df[df["vault_address"] == addr]

    total = (
        sub.groupby("date")["vault_supplyUsd"]
        .sum()
        .reset_index()
        .sort_values("date")
    )

    x = total["date"].to_numpy()
    y = total["vault_supplyUsd"].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 2.3), dpi=140)

    ax.plot(x, y, alpha=0.9)
    ax.fill_between(x, y, np.zeros(len(y)), alpha=0.1)

    ax.set_title(name, loc="left", fontsize=10)

    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    ax.grid(True, linestyle="--", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    st.pyplot(fig, use_container_width=True, clear_figure=True)



# -------------------------
# UI START
# -------------------------

st.divider()
st.header("Exposure by Vaults")

vaults_df = load_vaults(VAULTS_PATH)

# -------------------------
# LOCAL DATE FILTERS (self-contained)
# -------------------------
min_d = vaults_df["date"].min().date()
max_d = vaults_df["date"].max().date()

date_range = st.date_input(
    "Date range (vaults)",
    value=(min_d, max_d),
    min_value=min_d,
    max_value=max_d
)

start_dt = pd.to_datetime(date_range[0])
end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)

vf = vaults_df[
    (vaults_df["date"] >= start_dt) &
    (vaults_df["date"] < end_dt)
].copy()

if vf.empty:
    st.warning("No vault data in selected range")
    st.stop()

# -------------------------
# Preview table
# -------------------------
with st.expander("Preview vault dataset"):
    st.dataframe(vf.head(200), use_container_width=True)

# -------------------------
# Vault selector
# -------------------------
vault_map = (
    vf[["vault_address", "vault_name"]]
    .drop_duplicates()
    .sort_values("vault_name")
)

vault_dict = dict(zip(vault_map["vault_address"], vault_map["vault_name"]))

selected_vaults = st.multiselect(
    "Select vaults",
    vault_map["vault_address"].tolist(),
    default=vault_map["vault_address"].tolist()[:6],
    format_func=lambda x: vault_dict[x]
)

cards_per_row = st.slider("Vault cards per row", 2, 5, 3)

# -------------------------
# Vault cards
# -------------------------
for i in range(0, len(selected_vaults), cards_per_row):
    row = selected_vaults[i:i + cards_per_row]
    cols = st.columns(cards_per_row)

    for col, addr in zip(cols, row):
        with col:
            with st.container(border=True):
                plot_vault_card(vf, addr, vault_dict[addr])

# ======================================================
# MARKET ALLOCATION SECTION
# ======================================================

st.subheader("Vault Allocation inside Markets")

shares = market_vault_shares(vf)

markets = sorted(shares["market_id"].unique())

market_sel = st.selectbox("Select market", markets, format_func=short_hex)

sm = shares[shares["market_id"] == market_sel]

mode = st.radio("View", ["Latest", "Peak", "Average"], horizontal=True)

if mode == "Latest":
    last = sm["date"].max()
    t = sm[sm["date"] == last].copy()
    t["vault"] = t["vault_name"]
    st.dataframe(t[["vault", "vault_supplyUsd", "share_pct"]]
                 .sort_values("vault_supplyUsd", ascending=False))

elif mode == "Peak":
    g = sm.groupby("vault_name")["share_pct"].max().reset_index()
    st.dataframe(g.sort_values("share_pct", ascending=False))

else:
    g = sm.groupby("vault_name")["share_pct"].mean().reset_index()
    st.dataframe(g.sort_values("share_pct", ascending=False))


# =========================================================
# RELATIVE EXPOSURE VS CHAIN TVL
# =========================================================

st.divider()
st.header("Relative Exposure vs Chain TVL")

# -----------------------------
# 1️⃣ MANUAL INPUT: Chain TVL
# -----------------------------
CHAIN_TVL = {
    "Arbitrum": 410_838_783,   
    "ethereum": 6_232_876_939,
    "Plume": 167_389_913,
}

st.caption("Chain TVL values are manually provided (snapshot).")

# -----------------------------
# 2️⃣ Compute exposure TS by chain
# -----------------------------
chain_ts = (
    df.groupby(["date", "chain_network"], as_index=False)["exposure_supplyUsd"]
      .sum()
      .rename(columns={"exposure_supplyUsd": "exposure_usd"})
)

# map TVL
chain_ts["chain_tvl"] = chain_ts["chain_network"].map(CHAIN_TVL)

# drop chains without TVL provided
chain_ts = chain_ts.dropna(subset=["chain_tvl"])

chain_ts["share_pct"] = 100 * chain_ts["exposure_usd"] / chain_ts["chain_tvl"]

# -----------------------------
# 3️⃣ Latest snapshot table
# -----------------------------
latest_date = chain_ts["date"].max()

latest = chain_ts[chain_ts["date"] == latest_date].copy()
latest = latest.sort_values("share_pct", ascending=False)

st.subheader("Latest exposure vs chain TVL")

st.dataframe(
    latest[["chain_network", "exposure_usd", "chain_tvl", "share_pct"]]
      .rename(columns={
          "chain_network": "Chain",
          "exposure_usd": "Exposure USD",
          "chain_tvl": "Chain TVL",
          "share_pct": "% of Chain TVL"
      }),
    use_container_width=True
)

# -----------------------------
# 4️⃣ Time-series plot
# -----------------------------
st.subheader("Exposure as % of Chain TVL over time")

fig, ax = plt.subplots(figsize=(11, 4), dpi=150)

for chain, g in chain_ts.groupby("chain_network"):
    g = g.sort_values("date")
    ax.plot(
        g["date"].to_numpy(),
        g["share_pct"].to_numpy(),
        label=chain,
        alpha=0.9
    ) 

ax.set_ylabel("% of Chain TVL")
ax.set_xlabel("")
ax.set_title("Relative Exposure vs Chain TVL", loc="left")

ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

ax.grid(True, linestyle="--", alpha=0.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(frameon=False)

st.pyplot(fig, use_container_width=True, clear_figure=True)
