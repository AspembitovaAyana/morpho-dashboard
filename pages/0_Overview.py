import streamlit as st

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

# --- Links section
st.subheader("Primary references")
st.markdown("""
- Stream loss disclosure & xUSD depeg reporting (examples): CoinDesk / Decrypt  
- deUSD wind-down / claims portal reporting: The Block / other coverage  
- Ecosystem impact snapshot: Arbitrum forum update  
- Dune recap: Dune Digest 037  
""")
