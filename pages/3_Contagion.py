import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

st.header("AUM per Curator (absolute values)")

st.image("assets/aum_curator_abs.png", width=700)

st.link_button(
    "Open interactive dashboard in Dune",
    "https://dune.com/queries/6702717"
)

st.header("AUM per Curator (relative values)")

st.image("assets/aum_curator_rel.png", width=700)

st.link_button(
    "Open interactive dashboard in Dune",
    "https://dune.com/queries/6702717"
)

st.header("MEV Capital USDC Ethereum")

st.image("assets/mev_eth_usdc.png", width=700)

st.link_button(
    "Open interactive dashboard in Dune",
    "https://dune.com/queries/6702137"
)

st.header("MEV Capital USDC Arbitrum")

st.image("assets/mev_usdc_arb.png", width=700)

st.link_button(
    "Open interactive dashboard in Dune",
    "https://dune.com/queries/6702137"
)

