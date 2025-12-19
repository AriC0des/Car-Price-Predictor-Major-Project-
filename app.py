import streamlit as st
import pickle
import numpy as np
# # Load the saved model and scaler
with open('random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Set Streamlit Page Title
st.set_page_config(page_title="Used Car Price Predictor 🚗", layout="centered")

# App Title
st.title('🚗 Used Car Price Prediction App')

st.write('Fill the car details below to predict the selling price!')

# User Inputs
year = st.number_input('Year of Manufacture', min_value=1990, max_value=2024, value=2015)
km_driven = st.number_input('Kilometers Driven', min_value=0, value=30000)
mileage = st.number_input('Mileage (kmpl)', min_value=0.0, value=18.0)
engine = st.number_input('Engine (CC)', min_value=500, max_value=5000, value=1200)
max_power = st.number_input('Max Power (bhp)', min_value=30.0, max_value=400.0, value=80.0)

fuel_type = st.selectbox('Fuel Type', ['Diesel', 'Petrol', 'LPG'])
seller_type = st.selectbox('Seller Type', ['Dealer', 'Individual', 'Trustmark Dealer'])
transmission = st.selectbox('Transmission Type', ['Manual', 'Automatic'])
owner = st.selectbox('Owner Type', ['First', 'Second', 'Third', 'Fourth & Above', 'Test Drive Car'])

# Encode Categorical Variables
fuel_Diesel = 1 if fuel_type == 'Diesel' else 0
fuel_LPG = 1 if fuel_type == 'LPG' else 0
fuel_Petrol = 1 if fuel_type == 'Petrol' else 0

seller_Dealer = 1 if seller_type == 'Dealer' else 0
seller_Individual = 1 if seller_type == 'Individual' else 0
seller_Trustmark = 1 if seller_type == 'Trustmark Dealer' else 0

transmission_Manual = 1 if transmission == 'Manual' else 0
transmission_Automatic = 1 if transmission == 'Automatic' else 0

owner_First = 1 if owner == 'First' else 0
owner_Second = 1 if owner == 'Second' else 0
owner_Third = 1 if owner == 'Third' else 0
owner_Fourth_Above = 1 if owner == 'Fourth & Above' else 0
owner_TestDrive = 1 if owner == 'Test Drive Car' else 0

# Create feature array (order should match training)
features = np.array([[year, km_driven, mileage, engine, max_power,
                      fuel_Diesel, fuel_LPG, fuel_Petrol,
                      seller_Dealer, seller_Individual, seller_Trustmark,
                      transmission_Automatic, transmission_Manual,
                      owner_First, owner_Second]])

# Scale features
scaled_features = scaler.transform(features)

# Predict Button
if st.button('Predict Selling Price'):
    prediction = model.predict(scaled_features)
    st.success(f"💰 Predicted Selling Price: ₹ {prediction[0]:,.2f}")
