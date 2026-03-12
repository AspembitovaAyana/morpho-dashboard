import streamlit as st
import pandas as pd

st.title("Incident Overview: xUSD + deUSD")
st.caption("A concise timeline + mechanism diagram linking asset failures to Morpho outcomes.")

# --- High-level framing
with st.container(border=True):
    st.markdown("""
**Core story (one line):** Stream’s loss disclosure + withdrawal freeze triggered xUSD depeg and a liquidity unwind across lending markets; Elixir’s deUSD then collapsed/sunset due to Stream exposure; Morpho vaults reduced exposure, with bad debt realized in specific vault/market paths.
""")

st.divider()

# --- Timeline (edit wording as you like)
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

# --- Mechanism diagram (Mermaid)
st.subheader("Mechanism diagram")

st.markdown("Paste this diagram into your write-up; it shows the causal chain clearly.")

st.code(r"""
flowchart TD
  A[Stream: loss disclosure + withdrawals paused] --> B[xUSD depegs]
  B --> C[Lending markets unwind: pools drained, utilization spikes]
  C --> D[Liquidations struggle: oracle lag/bounds + thin liquidity + keeper economics]
  D --> E[Bad debt crystallizes in specific markets/vault paths]

  A --> F[Elixir exposure to Stream impairs backing]
  F --> G[deUSD depegs / sunset announced]
  G --> H[Morpho curators remove/zero allocations to affected markets]
  H --> I[Vault-level bad-debt realization / withdrawal queue effects]
""", language="text")

st.divider()

# --- Where to look in your dashboard
st.subheader("Where this dashboard answers what")
c1, c2, c3 = st.columns(3)
with c1:
    with st.container(border=True):
        st.markdown("### Exposure")
        st.markdown("- By chain\n- By market\n- By collateral\n- By vault/curator")
with c2:
    with st.container(border=True):
        st.markdown("### Behavior")
        st.markdown("- Curator exits/reallocations\n- Timing vs depeg window\n- Liquidity stress signals")
with c3:
    with st.container(border=True):
        st.markdown("### Outcomes (Dune)")
        st.markdown("- Liquidations\n- Bad debt\n- Cross-check evidence")

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

st.header("How Morpho Works")
 
st.markdown("""
Morpho Blue is a permissionless lending protocol built around **isolated markets**.
Each market is defined by a single collateral/loan pair, an oracle, and an LLTV parameter.
Markets are grouped into vaults managed by curators who decide capital allocation and risk parameters.
""")
 
st.graphviz_chart("""
digraph architecture {
    rankdir=LR
    node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11]
    edge [fontsize=10]
 
    Depositor [label="Depositor\n(e.g. USDC)", fillcolor="#dbeafe"]
    Vault     [label="MetaMorpho Vault\n(e.g. MEV Capital USDC)", fillcolor="#e0f2fe"]
    Curator   [label="Curator\n(Risk Manager)", fillcolor="#fef9c3"]
    Market1   [label="Market A\nxUSD / USDC", fillcolor="#fee2e2"]
    Market2   [label="Market B\nwstETH / USDC", fillcolor="#dcfce7"]
    Market3   [label="Market C\nsdeUSD / USDC", fillcolor="#fee2e2"]
    Borrower  [label="Borrower", fillcolor="#f3e8ff"]
 
    Depositor -> Vault    [label="deposits"]
    Vault -> Curator      [label="managed by"]
    Curator -> Market1    [label="allocates"]
    Curator -> Market2    [label="allocates"]
    Curator -> Market3    [label="allocates"]
    Market1 -> Borrower   [label="lends USDC"]
    Market2 -> Borrower   [label="lends USDC"]
    Market3 -> Borrower   [label="lends USDC"]
 
    Isolation [label="Bad debt in Market A\ndoes NOT affect Market B or C",
               shape=note, fillcolor="#fef3c7", fontsize=10]
    Market1 -> Isolation [style=dashed, color="#ef4444"]
}
""")
 
st.markdown("""
**Key parameters per market:**  
- **LLTV** — maximum LTV before liquidation triggers  
- **Oracle** — prices the collateral asset  
- **Collateral / Loan token** — defines the market pair  
 
Liquidation is triggered when `borrowAssets / (collateralAssets × oracle_price) > LLTV`.
The oracle is therefore the critical link between market price and protocol risk controls.
""")
