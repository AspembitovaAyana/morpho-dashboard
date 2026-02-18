import streamlit as st

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
