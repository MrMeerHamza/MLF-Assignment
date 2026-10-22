"""
Heart Disease Prediction System - Streamlit Web App
Assignment 02 (Task 2) - Muddasar Ali Ghafoor (SP24-BSE-062)

Loads the SVC model trained in Task2_Heart_Disease_Prediction.ipynb and
encodes the user's input exactly as in Step 8.3 of the notebook.
"""

import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

APP_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_model():
    with open(os.path.join(APP_DIR, "svc_trained_model.pkl"), "rb") as model_file:
        return pickle.load(model_file)


# Label Encoders trained on the same fixed category lists as Step 5.1 of the notebook
chest_pain_label_encoder = LabelEncoder().fit(np.ravel(["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]))
exercise_angina_label_encoder = LabelEncoder().fit(np.ravel(["Yes", "No"]))
thal_label_encoder = LabelEncoder().fit(np.ravel(["Normal", "Fixed Defect", "Reversible Defect"]))
major_vessels_ordinal_mapping = {"Zero": 0, "One": 1, "Two": 2, "Three": 3}

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")
st.title("❤️ Heart Disease Prediction")
st.write("Select the patient's details and click **Predict** to see whether the patient is likely to have heart disease.")

col1, col2 = st.columns(2)
with col1:
    chest_pain_input = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    major_vessels_input = st.selectbox("Major Vessels Coloured by Fluoroscopy", ["Zero", "One", "Two", "Three"])
with col2:
    exercise_angina_input = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    thal_input = st.selectbox("Thal (Thallium Stress Test)", ["Normal", "Fixed Defect", "Reversible Defect"])

if st.button("Predict", type="primary"):
    # Convert user input into a feature vector and label-encode it (same as notebook Step 8.2 / 8.3)
    unseen_data_features = pd.DataFrame({
        "ChestPainType": chest_pain_label_encoder.transform([chest_pain_input]),
        "ExerciseAngina": exercise_angina_label_encoder.transform([exercise_angina_input]),
        "MajorVessels": [major_vessels_ordinal_mapping[major_vessels_input]],
        "Thal": thal_label_encoder.transform([thal_input]),
    })

    predicted_heart_disease = load_model().predict(unseen_data_features)[0]

    if predicted_heart_disease == 1:
        st.error("Prediction: **HEART DISEASE**")
    else:
        st.success("Prediction: **NO HEART DISEASE**")

    with st.expander("Encoded feature vector sent to the model"):
        st.dataframe(unseen_data_features, hide_index=True)

st.caption("Model: Support Vector Classifier (scikit-learn) trained on 297 patients from the UCI Cleveland Heart Disease dataset "
           "with 4 input attributes. For learning purposes only, not a medical diagnosis.")
