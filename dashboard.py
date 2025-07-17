import streamlit as st
import pandas as pd
import plotly.express as px
import glob
import os

st.set_page_config(page_title="Battery Analyzer Dashboard", layout="wide")

st.title("Battery Analyzer Dashboard")

data_dir = st.sidebar.text_input("Processed data directory", value="processed_data")

if not os.path.isdir(data_dir):
    st.warning("Input a valid directory containing processed CSV files.")
    st.stop()

files = glob.glob(os.path.join(data_dir, "*_cycles.csv"))
if not files:
    st.warning("No cycle summary files found in directory.")
    st.stop()

file_choice = st.selectbox("Select cell", options=[os.path.basename(f).replace("_cycles.csv", "") for f in files])

base_path = os.path.join(data_dir, file_choice)
summary_path = base_path + "_cycles.csv"
ica_path = base_path + "_ica.csv"
dcir_path = base_path + "_dcir.csv"
raw_path = base_path + ".csv"

st.header("Cycle Summary and Capacity Fade")
summary_df = pd.read_csv(summary_path)
fig = px.line(summary_df, x="full_cycle_num", y=["charge_capacity", "discharge_capacity"], title="Capacity vs Cycle")
st.plotly_chart(fig, use_container_width=True)

if os.path.isfile(ica_path):
    st.header("Incremental Capacity (dQ/dV)")
    ica_df = pd.read_csv(ica_path)
    fig2 = px.line(ica_df, x="voltage", y="dQdV", title="ICA Curve")
    st.plotly_chart(fig2, use_container_width=True)

if os.path.isfile(dcir_path):
    st.header("DCIR over Test")
    dcir_df = pd.read_csv(dcir_path)
    fig3 = px.scatter(dcir_df, x="time", y="dc_internal_resistance_mohm", title="DC Internal Resistance Events")
    st.plotly_chart(fig3, use_container_width=True)

if os.path.isfile(raw_path):
    if st.checkbox("Show raw processed data"):
        st.write(pd.read_csv(raw_path).head()) 