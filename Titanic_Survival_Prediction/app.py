"""
Titanic Passenger Survival Prediction System - Streamlit Web App
Assignment 02 (Task 1) - Muddasar Ali Ghafoor (SP24-BSE-062)

Loads the SVC model trained in Task1_Titanic_Survival_Prediction.ipynb and
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
pclass_label_encoder = LabelEncoder().fit(np.ravel(["First", "Second", "Third"]))
gender_label_encoder = LabelEncoder().fit(np.ravel(["Male", "Female"]))
embarked_label_encoder = LabelEncoder().fit(np.ravel(["Southampton", "Cherbourg", "Queenstown"]))
sibling_ordinal_mapping = {"Zero": 0, "One": 1, "Two": 2, "Three": 3}

st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢")
st.title("🚢 Titanic Passenger Survival Prediction")
st.write("Select the passenger's details and click **Predict** to see whether the passenger would have survived.")

col1, col2 = st.columns(2)
with col1:
    pclass_input = st.selectbox("Passenger Class (PClass)", ["First", "Second", "Third"])
    sibling_input = st.selectbox("Siblings / Spouses Aboard (Sibling)", ["Zero", "One", "Two", "Three"])
with col2:
    gender_input = st.selectbox("Gender", ["Male", "Female"])
    embarked_input = st.selectbox("Port of Embarkation (Embarked)", ["Southampton", "Cherbourg", "Queenstown"])

if st.button("Predict", type="primary"):
    # Convert user input into a feature vector and label-encode it (same as notebook Step 8.2 / 8.3)
    unseen_data_features = pd.DataFrame({
        "PClass": pclass_label_encoder.transform([pclass_input]),
        "Gender": gender_label_encoder.transform([gender_input]),
        "Sibling": [sibling_ordinal_mapping[sibling_input]],
        "Embarked": embarked_label_encoder.transform([embarked_input]),
    })

    predicted_survival = load_model().predict(unseen_data_features)[0]

    if predicted_survival == 1:
        st.success("Prediction: **SURVIVED**")
    else:
        st.error("Prediction: **NOT SURVIVED**")

    with st.expander("Encoded feature vector sent to the model"):
        st.dataframe(unseen_data_features, hide_index=True)

st.caption("Model: Support Vector Classifier (scikit-learn) trained on 100 Titanic passengers with 4 input attributes.")
