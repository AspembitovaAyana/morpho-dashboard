import streamlit as st

st.set_page_config(page_title="Morpho Stress Event", layout="wide")

# =========================================================
# HEADER
# =========================================================

st.title("Morpho Stress Event Dashboard")

st.markdown(
"""
This dashboard analyzes the **xUSD / deUSD depeg stress event** on the Morpho protocol, 
focusing on exposure concentration, curator behavior, and resulting outcomes.

The goal is to provide a structured view of:

• **Where risk was concentrated** across chains, markets, and vaults  
• **How curators responded** during the stress event  
• **What outcomes occurred**, including bad debt and liquidations  

"""
)

st.divider()

# =========================================================
# INCIDENT SUMMARY
# =========================================================

st.header("Incident Overview")

st.markdown(
"""
During the stress event, the stable assets **xUSD** and **deUSD** experienced significant depegs, 
leading to elevated collateral risk across several Morpho markets.

Key systemic questions addressed in this dashboard:

1. Which markets and vaults were exposed to the affected assets?
2. How did vault curators react during the incident?
3. Why did liquidations occur slowly in some markets?
4. What was the magnitude of resulting bad debt?

"""
)

st.divider()

# =========================================================
# HOW TO USE DASHBOARD
# =========================================================

st.header("How to Navigate")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
    """
    **Exposure Page**
    - Exposure by chain, market, and vault
    - Relative exposure vs chain TVL
    - Vault allocation within markets
    """
    )

    st.markdown(
    """
    **Incident Overview Page**
    - Timeline of the stress event
    - Key mechanisms behind the depeg
    """
    )

with col2:
    st.markdown(
    """
    **Outcomes Page**
    - Bad debt evolution
    - Liquidation dynamics
    - Links to interactive Dune dashboards
    """
    )

    st.markdown(
    """
    **Contagion Overview Page**
    - Evolution of AUM across curators
    - Affected vaults with no direct exposure to xUSD/deUSD
    """
    )

st.divider()

# =========================================================
# DATA NOTE
# =========================================================

st.caption(
"""
Data sources include Morpho protocol datasets, on-chain analytics, and Dune Analytics queries.  
This dashboard is intended for risk analysis and incident post-mortem review.
"""
)
