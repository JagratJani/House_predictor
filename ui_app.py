# ui_app.py

import requests
import streamlit as st

# URL of your FastAPI endpoint
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

st.title("🏠 California House Price Predictor")
st.write("Move the sliders / inputs to set house features and click **Predict**.")

st.markdown("---")

# Sidebar or main inputs for features
st.subheader("Input Features")

col1, col2 = st.columns(2)

with col1:
    MedInc = st.slider(
        "Median Income (10k USD units)",
        min_value=0.5,
        max_value=15.0,
        value=5.0,
        step=0.1,
        help="Median income in the block group (from dataset, typically ~1.5 to 15)"
    )

    HouseAge = st.slider(
        "Median House Age (years)",
        min_value=1.0,
        max_value=60.0,
        value=25.0,
        step=1.0,
    )

    AveRooms = st.slider(
        "Average Rooms",
        min_value=1.0,
        max_value=15.0,
        value=6.0,
        step=0.1,
    )

    AveBedrms = st.slider(
        "Average Bedrooms",
        min_value=0.5,
        max_value=5.0,
        value=1.0,
        step=0.1,
    )

with col2:
    Population = st.slider(
        "Population",
        min_value=100,
        max_value=5000,
        value=1000,
        step=50,
    )

    AveOccup = st.slider(
        "Average Occupancy",
        min_value=1.0,
        max_value=6.0,
        value=3.0,
        step=0.1,
    )

    Latitude = st.slider(
        "Latitude",
        min_value=32.0,
        max_value=42.0,
        value=34.2,
        step=0.1,
    )

    Longitude = st.slider(
        "Longitude",
        min_value=-124.0,
        max_value=-114.0,
        value=-118.5,
        step=0.1,
    )

st.markdown("---")

if st.button("🔮 Predict Price"):
    # Build the JSON payload
    payload = {
        "MedInc": MedInc,
        "HouseAge": HouseAge,
        "AveRooms": AveRooms,
        "AveBedrms": AveBedrms,
        "Population": Population,
        "AveOccup": AveOccup,
        "Latitude": Latitude,
        "Longitude": Longitude,
    }

    try:
        with st.spinner("Calling model API..."):
            response = requests.post(API_URL, json=payload, timeout=10)
        if response.status_code == 200:
            result = response.json()
            price_100k = result["predicted_value_100k"]
            price_dollars = result["predicted_value_dollars"]

            st.success("Prediction successful!")
            st.metric("Predicted Value (x 100K USD)", f"{price_100k:,.3f}")
            st.metric("Approx Price (USD)", f"${price_dollars:,.0f}")

            st.write("💡 *Note:* This is a model estimate based on the California Housing dataset.")
        else:
            st.error(f"API Error {response.status_code}: {response.text}")
    except Exception as e:
        st.error(f"Failed to reach API: {e}")

else:
    st.info("Set the features and click **Predict Price** above.")
