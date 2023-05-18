import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesClassifier
from prediction import get_prediction, ordinal_encoder
import zipfile


# Load the model
model1 = joblib.load(r'Model/rf_model.joblib')

# Convert the model to a format that is compatible with the Render server
joblib.dump(model1, r'Model/rf_model_render.joblib', protocol=4)

# Load the model
model = joblib.load(r'Model/rf_model_render.joblib')

st.set_page_config(page_title="Accident Severity Prediction App",
                   page_icon="🚧", layout="wide")


#creating option list for dropdown menu
options_day = ['Sunday', "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

options_age = ['18-30', '31-50', 'Over 51', 'Unknown', 'Under 18']


options_light_conditions = ['Daylight','Darkness - lights lit','Darkness - no lighting','Darkness - lights unlit']

options_vehicletype = ['Automobile', 'Public (> 45 seats)', 'Lorry (41?100Q)', 'nan',
                            'Public (13?45 seats)', 'Lorry (11?40Q)', 'Long lorry', 'Public (12 seats)',
                            'Taxi', 'Pick up upto 10Q', 'Stationwagen', 'Ridden horse', 'Other', 'Bajaj',
                            'Turbo', 'Motorcycle', 'Special vehicle', 'Bicycle']

options_accident_cause = ['Moving Backward', 'Overtaking', 'Changing lane to the left',
                            'Changing lane to the right', 'Overloading', 'Other',
                            'No priority to vehicle', 'No priority to pedestrian', 'No distancing',
                            'Getting off the vehicle improperly', 'Improper parking', 'Overspeed',
                            'Driving carelessly', 'Driving at high speed', 'Driving to the left',
                            'Unknown', 'Overturning', 'Turnover', 'Driving under the influence of drugs',
                            'Drunk driving']

options_junction_type = ['No junction', 'Y Shape', 'Crossing', 'O Shape', 'Other', 'Unknown', 'T Shape', 'X Shape' ,'nan']

features = ['hour','day_of_week','casualties','accident_cause','vehicles_involved','vehicle_type','driver_age','accident_area','driving_experience','lanes']


st.markdown("<h1 style='text-align: center;'>Accident Severity Prediction App 🚧</h1>", unsafe_allow_html=True)
def main():
    with st.form('prediction_form'):

        st.subheader("Enter the input for following features:")
        
        hour = st.slider("Pickup Hour: ", 0, 23, value=0, format="%d")
        minute = st.slider("Minute: ", 0, 60, value=0, format="%d")
        day_of_week = st.selectbox("Select Day of the Week: ", options=options_day)
        casualties = st.slider("casualties: ", 1, 8, value=0, format="%d")
        light_conditions = st.selectbox("Light Conditions: ", options=options_light_conditions)
        vehicles_involved = st.slider("vehicles_involved: ", 1, 7, value=0, format="%d")
        driver_age = st.selectbox("Select Driver Age: ", options=options_age)
        vehicle_type = st.selectbox("Vehicle type: ", options=options_vehicletype          )
        accident_cause = st.selectbox("Accident Cause: ",options=options_accident_cause)
        junction_type = st.selectbox("junction_type: ",options=options_junction_type)
        
        submit = st.form_submit_button("Predict")


    if submit:
        day_of_week = ordinal_encoder(day_of_week, options_day)
        driver_age =  ordinal_encoder(driver_age, options_age)
        vehicle_type = ordinal_encoder(vehicle_type, options_vehicletype )
        light_conditions = ordinal_encoder(light_conditions, options_light_conditions)
        junction_type = ordinal_encoder(junction_type, options_junction_type)
        accident_cause = ordinal_encoder(accident_cause, options_accident_cause)

        data = np.array([minute,hour,casualties,vehicles_involved,day_of_week,accident_cause,
                         driver_age,light_conditions,junction_type,vehicle_type]).reshape(1,-1)

        print(data)
        print(model.feature_names_in_)

        pred = get_prediction(data=data, model=model)

        st.write(f"The predicted severity is:  {pred[0]}")

if __name__ == '__main__':
    main()

