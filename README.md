# Distributed Time-Series Similarity Search

A distributed system for performing fast similarity search on time-series data using SAX transformation, Locality Sensitive Hashing (LSH), and multiprocessing.

This project demonstrates how large time-series datasets can be searched efficiently by combining symbolic representations and distributed computation.\\\

## Technologies

- Python
- NumPy
- Pandas
- pyts (SAX transformation)
- datasketch (Locality Sensitive Hashing)
- Multiprocessing
- Matplotlib
- Streamlit

## Project Architecture

The system follows this pipeline:

Dataset → Preprocessing → SAX Transformation → Distributed Search → Similarity Matching → Evaluation → Visualization


## Goals

- Compare search performance between small and large datasets
- Reduce search time using symbolic time-series representations
- Simulate distributed search across multiple nodes
- Visualize similarity results and runtime performance

------------------------------------------------------------ 

## Installation

Clone the repository

git clone https://github.com/distributed-time-series/distributed-time-series-search.git

cd distributed-time-series-search

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Run Streamlit Demo
streamlit run streamlit_app.py
