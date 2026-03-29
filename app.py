import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(__file__))

from src.preprocess import load_stock_data, create_time_series_windows
from src.sax_transform import apply_sax
from src.distributed_search import split_into_nodes, distributed_search

DATA_PATH = "dataset/Microsoft_Stock.csv"
WINDOW_SIZE = 20

st.set_page_config(page_title="Distributed Time-Series Search", layout="wide")
st.title("Distributed Similarity Search over Time-Series Data")
st.markdown("Find similar stock price patterns across distributed nodes using SAX encoding.")

# Load and prepare data once (cached so it doesn't reload on every interaction)
@st.cache_data
def load_all_data():
    data = load_stock_data(DATA_PATH)
    time_series = create_time_series_windows(data, window_size=WINDOW_SIZE)
    sax_data = apply_sax(time_series)
    return data, time_series, sax_data

data, time_series, sax_data = load_all_data()

# Sidebar
st.sidebar.header("Settings")
num_nodes = st.sidebar.slider("Number of Nodes", min_value=2, max_value=6, value=3)
query_index = st.sidebar.slider(
    "Query Series Index",
    min_value=0,
    max_value=len(sax_data) - 1,
    value=0,
    help="Pick which time series to use as your search query"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Team:** Gavin · Eli · Ronny · Dharshan ")
st.sidebar.markdown("**Steps completed:** 2–9")

# Show selected query
st.subheader("Query Pattern")
query = sax_data[query_index]
st.code(f"SAX pattern: {' '.join(query)}")

col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(time_series[query_index], color="steelblue", linewidth=2, marker="o", markersize=3)
    ax.set_title("Query Time Series (raw prices)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# Run search
if st.button("Search Across Nodes"):
    with st.spinner("Searching..."):
        nodes = split_into_nodes(sax_data, num_nodes=num_nodes)

        import time
        start = time.time()
        results = distributed_search(nodes, query)
        elapsed = time.time() - start

    st.success(f"Search completed in **{elapsed:.4f} seconds** across **{num_nodes} nodes**")

    # Node sizes
    st.subheader("Node Distribution")
    node_cols = st.columns(num_nodes)
    nodes_for_display = split_into_nodes(sax_data, num_nodes=num_nodes)
    for i, (col, node) in enumerate(zip(node_cols, nodes_for_display)):
        col.metric(label=f"Node {i+1}", value=f"{len(node)} series")

    # Results table
    st.subheader("Top Similar Patterns Found")
    for rank, (idx, dist) in enumerate(results, start=1):
        st.markdown(f"**#{rank}** — Series `{idx + 1}` &nbsp;&nbsp; Distance: `{float(dist):.3f}`")

    # Plot: query vs top match
    st.subheader("Query vs Best Match")
    if results:
        best_idx = results[0][0]
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(time_series[query_index], label="Your Query", linewidth=2, color="steelblue")
        ax2.plot(time_series[best_idx], label=f"Best Match (series #{best_idx+1})", linewidth=2, color="coral", linestyle="--")
        ax2.set_title("Query vs Best Match")
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Price")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

    # Distance bar chart
    st.subheader("Match Distances (lower = more similar)")
    fig3, ax3 = plt.subplots(figsize=(8, 3))
    labels = [f"Series #{r[0]+1}" for r in results]
    distances = [float(r[1]) for r in results]
    bars = ax3.bar(labels, distances, color="steelblue", alpha=0.8)
    ax3.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
    ax3.set_ylabel("Distance")
    ax3.grid(True, axis="y", alpha=0.3)
    st.pyplot(fig3)