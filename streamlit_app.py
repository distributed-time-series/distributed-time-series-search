import streamlit as st

st.set_page_config(page_title="Distributed Time-Series Search", layout="wide")

st.title("Distributed Time-Series Similarity Search")

st.markdown("""
This application demonstrates **distributed similarity search** on time-series data.

Pipeline:
1. Load time-series dataset
2. Convert data using **SAX (Symbolic Aggregate Approximation)**
3. Perform **similarity search with Euclidean Distance + LSH**
4. Simulate distributed nodes using **multiprocessing**
5. Compare performance on **small vs large datasets**
""")

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select page below",
    [
        "Project Overview",
        "Dataset",
        "Project Overview",
        "Dataset",
        "SAX Transformation",
        "Similarity Search",
        "Distributed Search",
        "Evaluation",
        "Visualization"

    ]


)


if page == "Project Overview":
    st.header("Project Overview")
    st.write("This project demonstrates distributed similarity search for time-series data.")

elif page == "Dataset":
    st.header("Dataset")
    st.write("Load and explore the small and large datasets.")

elif page == "SAX Transformation":
    st.header("SAX Transformation")
    st.write("Convert time series to symbolic representation.")

elif page == "Similarity Search":
    st.header("Similarity Search")
    st.write("Find nearest time series using Euclidean distance.")

elif page == "Distributed Search":
    st.header("Distributed Search")
    st.write("Simulate multiple nodes using multiprocessing.")

elif page == "Evaluation":
    st.header("Performance Evaluation")
    st.write("Compare runtime between small and large datasets.")

elif page == "Visualization":
    st.header("Visualization")
    st.write("Display time-series plots and similarity results.")