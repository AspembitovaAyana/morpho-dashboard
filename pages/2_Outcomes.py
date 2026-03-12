import streamlit as st
import pandas as pd

# ─────────────────────────────────────────────
# BAD DEBT
# ─────────────────────────────────────────────
st.header("Bad Debt")

st.image("assets/BadDebt.png", width=700)

st.link_button(
    "Open interactive dashboard in Dune",
    "https://dune.com/queries/4308329"
)

# ─────────────────────────────────────────────
# LIQUIDATIONS
# ─────────────────────────────────────────────
st.header("Liquidations")

st.image("assets/Liquidations.png", width=700)

st.link_button(
    "Open interactive dashboard in Dune",
    "https://dune.com/queries/6689016"
)

# ─────────────────────────────────────────────
# WHY NO LIQUIDATIONS 
# ─────────────────────────────────────────────
st.header("Why No Liquidations? Oracle Divergence Analysis")

st.markdown("""
Morpho liquidations trigger when a position's **LTV exceeds the LLTV threshold** — which requires
the oracle to reflect true market value. During the November 2025 depeg, few compounding
factors prevented liquidations from occurring.
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
**Layer 1 — ERC4626 Oracle Frozen**  
The xUSD/deUSD oracle called `vault.convertToAssets()` — the vault's internal accounting NAV.
When Stream paused withdrawals, this value was never written down, keeping the oracle
pinned at ~$1.266 throughout the depeg.
""")
with col2:
    st.markdown("""
**Layer 2 — No Staleness Check**  
`MorphoChainlinkOracleV2` has no staleness enforcement by design.
A reverting staleness check would freeze the entire market — so the oracle silently
reported stale prices with no on-chain signal.
""")

st.divider()
# ─────────────────────────────────────────────
# STEP 1 — MARKET CONFIGS
# ─────────────────────────────────────────────
st.subheader("Step 1 — Market Configuration Review")

st.markdown("""
On-chain market configs were pulled for all affected markets to check LLTV, oracle
addresses, and collateral/loan token pairs — confirming the exact parameters in place
at the time of the depeg.
""")

@st.cache_data
def load_market_configs():
    df = pd.read_csv("data/api/market_configs.csv")
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

try:
    dfc = load_market_configs()

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
                dfc = dfc[
                    (dfc["date"] >= pd.to_datetime(dr[0])) &
                    (dfc["date"] < pd.to_datetime(dr[1]) + pd.Timedelta(days=1))
                ]

    c1, c2 = st.columns([1, 1])
    with c1:
        n = st.number_input("Rows", min_value=10, max_value=2000, value=50, step=10)
    with c2:
        show_types = st.checkbox("Show dtypes", value=False)

    st.dataframe(dfc.head(int(n)), use_container_width=True)

    if show_types:
        st.caption("Column dtypes")
        st.code(str(dfc.dtypes), language="text")

    st.caption(f"Rows: {len(dfc):,} | Columns: {len(dfc.columns)}")

except FileNotFoundError:
    st.error("market_configs.csv not found in data/api/")

st.divider()

# ─────────────────────────────────────────────
# STEP 2 — ORACLE CONTRACT CODE
# ─────────────────────────────────────────────
st.subheader("Step 2 — Oracle Contract Inspection")

st.markdown("""
With the oracle addresses from the config, the `MorphoChainlinkOracleV2` source was inspected.
The key finding: staleness is never checked — a known, deliberate design decision.
""")

with st.expander("ChainlinkDataFeedLib — staleness not enforced"):
    st.code("""
function getPrice(AggregatorV3Interface feed) internal view returns (uint256) {
    if (address(feed) == address(0)) return 1e36;
    (, int256 answer,,,) = feed.latestRoundData();
    // updatedAt is returned but never validated
    require(answer > 0, "negative price");
    return uint256(answer);
}
""", language="solidity")

st.info("""
**Why no staleness check?** A check that reverts on stale data would freeze the entire market —
no borrows, no repayments, no liquidations. The design trades oracle accuracy for market liveness.
The consequence: during the depeg, the oracle reported $1.266 indefinitely with no on-chain signal.
""")

st.divider()

# ─────────────────────────────────────────────
# STEP 3 — ORACLE PRICE DIVERGENCE
# ─────────────────────────────────────────────
st.subheader("Step 3 — On-chain Oracle Price vs Market Price")

st.markdown("""
Historical oracle prices were fetched directly from Alchemy archive nodes and compared against CoinGecko market prices over the depeg window
(Nov 01 – Nov 8, 2025).
""")

tab1, tab2 = st.tabs(["xUSD / USDC (Arbitrum)", "deUSD / USDC (Ethereum)"])

with tab1:
    st.markdown("""
**Oracle address:** `0x1837efFC34Bb5a96EFdA00d53560799bE3a4226E`  
The oracle used `xUSD_vault.convertToAssets()` — the vault's internal accounting rate.
When Stream paused withdrawals on Nov 3, the vault NAV was never written down,
keeping the oracle pinned at **~$1.266** while market price collapsed.
    """)
    st.image("assets/xusd_oracle_divergence.png", use_container_width=True)

with tab2:
    st.markdown("""
**Oracle address:** `0x65F9f6d537C2D628D1c2663896436817440eDB72`  
deUSD/sdeUSD was exposed to xUSD via Elixir's backing basket.
The deUSD oracle showed a slow upward drift (~accumulating yield) but never reflected
the sharp market discount that emerged once Elixir's xUSD exposure became known.
    """)
    st.image("assets/deusd_oracle_divergence.png", use_container_width=True)

with st.expander("Oracle Implementation Logic"):
    st.code("""
// ChainlinkDataFeedLib — staleness NOT checked 
function getPrice(AggregatorV3Interface feed) internal view returns (uint256) {
    if (address(feed) == address(0)) return 1e36;
    (, int256 answer,,,) = feed.latestRoundData();
    // updatedAt is returned but never validated
    require(answer > 0, "negative price");
    return uint256(answer);
}
""", language="solidity")

st.info("""
**Key implications:**
- The oracle relies on vault NAV / Chainlink feeds rather than live market prices
- Staleness is not enforced — stale prices produce no revert or signal
- If withdrawals are paused, `convertToAssets()` never marks down
- Result: oracle stays pinned at pre-depeg levels indefinitely, preventing liquidations
""")

