import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("house_price_model.joblib")

st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("🏠 House Price Prediction App")
st.write("Enter house details to predict the price")

# Numerical inputs
bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, step=1)
bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, step=1)
sqft = st.number_input("Square Feet (sqft)", min_value=200, max_value=20000, step=100)
lot_size = st.number_input("Lot Size", min_value=0, max_value=50000, step=100)
age = st.number_input("House Age (years)", min_value=0, max_value=200, step=1)
year_built = st.number_input("Year Built", min_value=1800, max_value=2025, step=1)
garage = st.number_input("Garage Spaces", min_value=0, max_value=5, step=1)
condition = st.slider("House Condition (1 = Poor, 10 = Excellent)", 1, 10)
school_rating = st.slider("School Rating (1–10)", 1, 10)

# Binary inputs
has_pool = st.selectbox("Has Pool?", [0, 1])
has_fireplace = st.selectbox("Has Fireplace?", [0, 1])
has_basement = st.selectbox("Has Basement?", [0, 1])

# Categorical inputs
location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
house_type = st.selectbox("House Type", ["Apartment", "Villa", "Independent House"])

# Create input DataFrame
input_data = pd.DataFrame({
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms],
    "sqft": [sqft],
    "lot_size": [lot_size],
    "age": [age],
    "year_built": [year_built],
    "garage": [garage],
    "location": [location],
    "house_type": [house_type],
    "condition": [condition],
    "has_pool": [has_pool],
    "has_fireplace": [has_fireplace],
    "has_basement": [has_basement],
    "school_rating": [school_rating]
})

if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated House Price: ₹{prediction:,.2f}")
