import streamlit as st
import joblib
import pandas as pd

# Load the trained model
logi = joblib.load('logi.sav')

st.title('Delivery Delay Prediction App')
st.write('Enter the details below to predict if there will be a delivery delay.')

# Input fields for features
delivery_distance = st.slider('Delivery Distance (km)', 0.0, 100.0, 20.0)
traffic_congestion = st.slider('Traffic Congestion (1-5)', 1, 5, 3)
weather_condition = st.slider('Weather Condition (1-5)', 1, 5, 3)
delivery_slot = st.slider('Delivery Slot (1-3)', 1, 3, 2)
driver_experience = st.slider('Driver Experience (years)', 0, 20, 5)
num_stops = st.slider('Number of Stops', 1, 10, 5)
vehicle_age = st.slider('Vehicle Age (years)', 0, 15, 3)
road_condition_score = st.slider('Road Condition Score (1-5)', 1, 5, 3)
package_weight = st.slider('Package Weight (kg)', 0.0, 50.0, 10.0)
fuel_efficiency = st.slider('Fuel Efficiency (km/l)', 5.0, 25.0, 15.0)
warehouse_processing_time = st.slider('Warehouse Processing Time (minutes)', 0, 150, 60)

# Create a DataFrame for prediction
input_data = pd.DataFrame([{
    'Delivery_Distance': delivery_distance,
    'Traffic_Congestion': traffic_congestion,
    'Weather_Condition': weather_condition,
    'Delivery_Slot': delivery_slot,
    'Driver_Experience': driver_experience,
    'Num_Stops': num_stops,
    'Vehicle_Age': vehicle_age,
    'Road_Condition_Score': road_condition_score,
    'Package_Weight': package_weight,
    'Fuel_Efficiency': fuel_efficiency,
    'Warehouse_Processing_Time': warehouse_processing_time
}])

if st.button('Predict Delay'):
    prediction = logi.predict(input_data)
    prediction_proba = logi.predict_proba(input_data)

    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.error(f'There is a high chance of Delivery Delay!')
    else:
        st.success(f'Delivery is likely to be On Time.')
    
    st.write(f'Probability of No Delay: {prediction_proba[0][0]:.2f}')
    st.write(f'Probability of Delay: {prediction_proba[0][1]:.2f}')
