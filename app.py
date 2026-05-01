import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Load the model and the scaler
model = joblib.load('flight_price_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Flight Price Predictor", page_icon="✈️")
st.title("✈️ Flight Price Predictor")

# Mappings (defined here for use in the app)
time_mapping = {'Early_Morning':0,'Morning':1,'Afternoon':2,'Evening':3,'Night':4,'Late_Night':5}

# 2. User Inputs
col1, col2 = st.columns(2)
with col1:
    airline = st.selectbox("Airline", ["Vistara", "Air India", "Indigo", "AirAsia", "GO FIRST", "SpiceJet"])
    source = st.selectbox("Source City", ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"])
    dest = st.selectbox("Destination City", ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"])
    flight_class = st.selectbox("Class", ["Economy", "Business"])

with col2:
    duration = st.number_input("Duration (Hours)", value=2.0)
    days_left = st.number_input("Days Left", value=15)
    stops = st.selectbox("Stops", ["zero", "one", "two_or_more"])
    dep_time = st.selectbox("Departure Time", ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"])
    arr_time = st.selectbox("Arrival Time", ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"])

# 3. Prediction Logic
if st.button("Calculate Price"):
    # Create matching columns
    input_df = pd.DataFrame(columns=['departure_time', 'stops', 'arrival_time', 'class', 'duration', 'days_left',
                                     'airline_AirAsia', 'airline_Air_India', 'airline_GO_FIRST', 'airline_Indigo',
                                     'airline_SpiceJet', 'airline_Vistara', 'source_city_Bangalore', 'source_city_Chennai',
                                     'source_city_Delhi', 'source_city_Hyderabad', 'source_city_Kolkata', 'source_city_Mumbai',
                                     'destination_city_Bangalore', 'destination_city_Chennai', 'destination_city_Delhi',
                                     'destination_city_Hyderabad', 'destination_city_Kolkata', 'destination_city_Mumbai'])
    input_df.loc[0] = 0

    # Map values
    input_df['duration'] = duration
    input_df['days_left'] = days_left
    input_df['class'] = 1 if flight_class == "Business" else 0
    input_df['stops'] = 0 if stops == "zero" else (1 if stops == "one" else 2)
    input_df['departure_time'] = time_mapping[dep_time]
    input_df['arrival_time'] = time_mapping[arr_time]

    # Set one-hot encoded columns
    input_df[f'airline_{airline.replace(" ", "_")}'] = 1
    input_df[f'source_city_{source}'] = 1
    input_df[f'destination_city_{dest}'] = 1

    # IMPORTANT: Scale the numerical data just like in training
    input_df[['duration', 'days_left']] = scaler.transform(input_df[['duration', 'days_left']])

    # Predict
    price = model.predict(input_df)[0]
    st.metric("Estimated Price", f"₹{int(price)}")
