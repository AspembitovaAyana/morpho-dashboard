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