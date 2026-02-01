import streamlit as st
import pandas as pd
from joblib import load

# Page config
st.set_page_config(
    page_title="Diamond Price Prediction",
    page_icon="💎",
    layout="centered"
)

st.title("💎 Diamond Price Prediction App")
st.write("Predict diamond prices using a trained KNN model.")

# Load model
@st.cache_resource
def load_model():
    return load("knn_pipeline.joblib")

model = load_model()

# Sidebar inputs
st.sidebar.header("Input Diamond Features")

carat = st.sidebar.number_input("Carat", min_value=0.1, max_value=5.0, value=1.0)
cut = st.sidebar.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
color = st.sidebar.selectbox("Color", ["D", "E", "F", "G", "H", "I", "J"])
clarity = st.sidebar.selectbox(
    "Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
)
depth = st.sidebar.number_input("Depth", min_value=40.0, max_value=80.0, value=61.5)
table = st.sidebar.number_input("Table", min_value=40.0, max_value=80.0, value=57.0)
x = st.sidebar.number_input("X (length)", min_value=0.0, value=6.5)
y = st.sidebar.number_input("Y (width)", min_value=0.0, value=6.5)
z = st.sidebar.number_input("Z (depth)", min_value=0.0, value=4.0)

# Create input DataFrame
input_data = pd.DataFrame({
    "carat": [carat],
    "cut": [cut],
    "color": [color],
    "clarity": [clarity],
    "depth": [depth],
    "table": [table],
    "x": [x],
    "y": [y],
    "z": [z],
})

st.subheader("Input Data")
st.dataframe(input_data)

# Prediction
if st.button("Predict Price 💰"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"Estimated Diamond Price: ₹ {int(prediction):}")
    except Exception as e:
        st.error("Prediction failed. Check model compatibility.")
        st.exception(e)
