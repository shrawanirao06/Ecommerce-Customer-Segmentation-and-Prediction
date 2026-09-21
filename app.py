import streamlit as st
import joblib
import pandas as pd
import numpy as np


# -----------------------------------
# Load the trained models
# -----------------------------------

prediction_model = joblib.load(
    r"C:\Users\SHRAWANI\Downloads\Ecommerce_Customer_Segmentation\gradient_boosting_customer_prediction_model.pkl"
)

segmentation_model = joblib.load(
    r"C:\Users\SHRAWANI\Downloads\Ecommerce_Customer_Segmentation\customer_segmentation_kmeans_model.pkl"
)

# Load the RFM scaler
rfm_scaler = joblib.load(
    r"C:\Users\SHRAWANI\Downloads\Ecommerce_Customer_Segmentation\rfm_scaler.pkl"
)


# -----------------------------------
# Page Title
# -----------------------------------

st.title("E-commerce Customer Segmentation and Prediction")

st.write(
    "Enter customer purchase information to identify the customer segment "
    "and predict whether the customer is likely to make a future purchase."
)


# -----------------------------------
# Input Fields
# -----------------------------------

recency = st.number_input(
    "Recency (days)",
    min_value=0,
    value=30
)

frequency = st.number_input(
    "Frequency (number of purchases)",
    min_value=1,
    value=5
)

monetary = st.number_input(
    "Monetary Value",
    min_value=0.0,
    value=1000.0
)

average_order_value = st.number_input(
    "Average Order Value",
    min_value=0.0,
    value=200.0
)


# -----------------------------------
# Prediction Button
# -----------------------------------

if st.button("Predict"):

    # -----------------------------------
    # Customer Segmentation
    # -----------------------------------

    rfm_input = pd.DataFrame({
        "Recency": [recency],
        "Frequency": [frequency],
        "Monetary": [monetary]
    })

    # Apply log transformation
    rfm_log_input = np.log1p(rfm_input)

    # Apply the same scaler used during training
    rfm_scaled_input = rfm_scaler.transform(rfm_log_input)

    # Predict customer cluster
    cluster = segmentation_model.predict(rfm_scaled_input)[0]

    # Display customer segment
    st.subheader("Customer Segment")

    if cluster == 1:
        st.success("High-Value / Highly-Engaged Customer")
    else:
        st.info("Low-Value / Less-Engaged Customer")


    # -----------------------------------
    # Future Purchase Prediction
    # -----------------------------------

    prediction_input = pd.DataFrame({
        "Recency": [recency],
        "Frequency": [frequency],
        "Monetary": [monetary],
        "AverageOrderValue": [average_order_value]
    })

    # Predict future purchase
    prediction = prediction_model.predict(prediction_input)[0]

    # Display prediction
    st.subheader("Future Purchase Prediction")

    if prediction == 1:
        st.warning("Prediction: No Future Purchase")
    else:
        st.success("Prediction: Future Purchase")

