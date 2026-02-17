import pandas as pd
import streamlit as st

st.set_page_config(page_title="Morpho Stress Event", layout="wide")

st.title("Morpho Stress Event Dashboard")
st.caption("Upload your exposure CSV to start")

st.sidebar.header("Upload data")

# Upload file
api_file = st.sidebar.file_uploader("Upload exposure CSV", type=["csv"])

# Read file
if api_file is not None:
    exposure = pd.read_csv(api_file)
else:
    exposure = None

# Display
if exposure is None:
    st.info("Please upload your exposure CSV from the sidebar.")
else:
    st.write("Columns detected:")
    st.write(list(exposure.columns))
    st.dataframe(exposure.head(50))
