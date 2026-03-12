import streamlit as st
import pandas as pd

st.title("Incident Overview: xUSD + deUSD")
st.caption("A concise timeline + mechanism diagram linking asset failures to Morpho outcomes.")

with st.container(border=True):
    st.markdown("""
**Core story (one line):** Stream’s loss disclosure + withdrawal freeze triggered xUSD depeg and a liquidity unwind across lending markets; Elixir’s deUSD then collapsed/sunset due to Stream exposure.
""")

st.divider()

st.subheader("Timeline (UTC dates)")

events = [
    ("Nov 3–4, 2025", "Stream discloses ~$93M loss and pauses withdrawals/deposits → xUSD begins sharp depeg."),
    ("Nov 4–7, 2025", "Contagion: lending pools drain / utilization spikes; liquidations struggle (oracle + liquidity + keeper incentives)."),
    ("Nov 6–7, 2025", "Elixir moves to sunset/wind down deUSD amid Stream exposure; deUSD price collapses."),
    ("Nov 7, 2025", "Public impact snapshots: Morpho exposure discussed (e.g., ~700k bad debt in a vault context)."),
    ("Nov 12–13, 2025", "Morpho vault operations remove/zero deUSD market exposure; ~3.6% vault-level bad debt referenced by curator comms."),
    ("Late Nov–Dec 2025", "Post-incident writeups: liquidation/oracle lessons; legal follow-ups emerge.")
]

for d, txt in events:
    st.markdown(f"**{d}** — {txt}")

st.divider()

st.header("Bad Debt — Reconciliation Across Sources")

st.markdown("""
Numbers reported across sources appear inconsistent because they measure **different things at different scopes**.
The table below reconciles them.
""")

data = {
    "Number": ["$93M", "$160M", "$283–285M", "$628K", "$68M", "$25.4M", "3.6%"],
    "What it measures": [
        "Loss by Stream's external fund manager — root cause of the event",
        "User deposits frozen in Stream at time of halt",
        "Total loans across all DeFi secured by Stream-related collateral",
        "Realized bad debt in Morpho's public xUSD/USDC Arbitrum vault",
        "Elixir's exposure via private, non-whitelisted Morpho Plume vault",
        "MEV Capital's Morpho position — worst-case if oracle had updated",
        "Bad debt as % of MEV Capital ETH USDC vault TVL (sdeUSD/USDC)",
    ],
    "Scope": [
        "Stream Finance (off-chain)",
        "Stream Finance",
        "Cross-protocol (Morpho, Euler, Silo…)",
        "One Morpho market (Arbitrum)",
        "One Morpho market (Plume)",
        "MEV Capital vault",
        "MEV Capital vault",
    ],
    "Source": [
        "[CoinDesk, Nov 4](https://www.coindesk.com/markets/2025/11/04/stream-finance-faces-usd93-million-loss-launches-legal-investigation)",
        "[CoinDesk, Nov 4](https://www.coindesk.com/markets/2025/11/04/stream-finance-faces-usd93-million-loss-launches-legal-investigation)",
        "[YAM via CoinDesk](https://www.coindesk.com/markets/2025/11/04/stream-finance-faces-usd93-million-loss-launches-legal-investigation) · [DL News, Dec 12](https://www.dlnews.com/articles/defi/stream-finance-founders-sue-partner-over-alleged-93m-loss/)",
        "[QuillAudits](https://x.com/QuillAudits_AI/status/1986377632926273796) · [Tiger Research, Nov 14](https://reports.tiger-research.com/p/collapse-of-the-defi-jenga-the-stream-eng)",
        "[BlockEden, Nov 9](https://blockeden.xyz/blog/2025/11/08/m-defi-contagion/)",
        "[Followin analysis](https://followin.io/en/trendingTopic/2647)",
        "[MEV Capital statement](https://x.com/MEVCapital/status/1988581694476071222)",
    ],
}

df_bd = pd.DataFrame(data)
st.dataframe(
    df_bd,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Source": st.column_config.LinkColumn("Source", display_text="🔗 Source"),
    },
)

st.info("""
**Bottom line for Morpho specifically:** only **$628K** was realized bad debt in the public 
xUSD/USDC Arbitrum market. The $68M Plume vault was private and non-whitelisted. 
The $25.4M worst-case never materialized because the oracle never updated.
Morpho's isolation architecture contained the damage to a single market.
""")
