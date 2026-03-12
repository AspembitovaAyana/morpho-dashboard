import streamlit as st

st.markdown("""
Morpho's isolated market design prevented *structural* contagion — bad debt in one market
cannot spill into another. But this analysis reveals a second channel: **reputational contagion**.
Depositor withdrawals spread across vaults managed by the same curator, regardless of whether
those vaults had any direct exposure to xUSD or deUSD.
""")

# ─────────────────────────────────────────────
# ECOSYSTEM AUM
# ─────────────────────────────────────────────
st.header("Ecosystem AUM — Curator Landscape")

st.markdown("""
Total Morpho AUM did not collapse after the event — structural isolation worked. 
But the **relative share** tells a different story: curators perceived as lower-risk 
(Steakhouse Finance, Gauntlet) grew their share significantly post-November, while 
curators associated with the incident shrank permanently. 
**Reputation became the differentiating factor.**
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Absolute AUM")
    st.markdown("No systemic collapse. Total AUM recovered within weeks — isolation design contained structural damage to a single market.")
    st.image("assets/aum_curator_abs.png", use_container_width=True)
    st.link_button("Open in Dune", "https://dune.com/queries/6702717")

with col2:
    st.subheader("Relative AUM share")
    st.markdown("Steakhouse Finance and Gauntlet gained market share. Curators associated with the incident lost depositor trust and never recovered their relative position.")
    st.image("assets/aum_curator_rel.png", use_container_width=True)
    st.link_button("Open in Dune", "https://dune.com/queries/6702717")

st.divider()

# ─────────────────────────────────────────────
# MEV CAPITAL CASE STUDY
# ─────────────────────────────────────────────
st.header("MEV Capital — Case Study in Reputational Contagion")

st.markdown("""
The clearest evidence of reputational contagion comes from MEV Capital's two USDC vaults.
**Neither vault had direct xUSD or deUSD exposure** — both held USDC as collateral.
Yet both experienced sharp AUM declines tied to MEV Capital's association with the incident
through their other affected vaults.
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Ethereum vault — USDC Prime")
    st.markdown("""
`0xf1fd8ac...` — MEV Capital USDC Prime (Ethereum)  
**Collateral: USDC. No xUSD or deUSD exposure.**

This vault was **fully exited by August 2025** — roughly 3 months before the November event.
The drawdown happened well before any public signal of xUSD stress, which raises a key question:

> *Did MEV Capital reduce risk proactively, or were depositors already exiting for unrelated reasons?*

Given the small scale (~$150k peak), this may reflect a low-priority vault that was wound down
as MEV Capital consolidated its Arbitrum strategy — not necessarily early risk awareness.
    """)
    st.image("assets/mev_eth_usdc.png", use_container_width=True)
    st.link_button("Open in Dune", "https://dune.com/queries/6702137")

with col2:
    st.subheader("Arbitrum vault — USDC")
    st.markdown("""
`0xa60643c...` — MEV Capital USDC (Arbitrum)  
**Collateral: USDC. No xUSD or deUSD exposure.**

This vault grew rapidly from September 2025, peaked at **~$50M in November**, 
then collapsed to near zero within weeks — despite having **zero direct exposure** 
to the depegged assets.

This is the clearest evidence of reputational contagion: depositors withdrew from a 
structurally safe vault solely because of MEV Capital's association with the xUSD incident 
through their other vaults.
    """)
    st.image("assets/mev_usdc_arb.png", use_container_width=True)
    st.link_button("Open in Dune", "https://dune.com/queries/6702137")

st.divider()

# ─────────────────────────────────────────────
# TAKEAWAY
# ─────────────────────────────────────────────
st.header("What This Tells Us About Risk Isolation")

col1, col2 = st.columns(2)

with col1:
    st.success("""
**Structural isolation worked**  
Bad debt stayed contained to the xUSD/USDC Arbitrum market.
300+ other Morpho vaults were entirely unaffected at the contract level.
Total ecosystem AUM recovered within weeks.
    """)

with col2:
    st.warning("""
**Reputational contagion is real and uncontained**  
Depositor withdrawals followed curator identity, not vault exposure.
MEV Capital lost AUM across vaults with no direct exposure.
Market share shifted permanently toward Steakhouse Finance and Gauntlet.
    """)

st.markdown("""
**Implication for risk monitoring:** On-chain bad debt metrics alone are insufficient. 
A complete risk framework needs to track **curator-level AUM flows as a leading indicator** — 
depositor withdrawals from a curator's non-exposed vaults signal reputational stress 
before bad debt is formally realized.
""")
