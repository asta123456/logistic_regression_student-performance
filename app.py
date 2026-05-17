import streamlit as st
import numpy as np
import joblib

# =========================
# LOAD MODEL
# =========================

model = joblib.load('logistic_model.pkl')

# LOAD SCALER

scaler = joblib.load('scaler.pkl')

# =========================
# TITLE
# =========================

st.title("Student Pass Prediction")

st.write("Logistic Regression Classification")

# =========================
# INPUTS
# =========================

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

reading_score = st.number_input(
    "Reading Score",
    min_value=0,
    max_value=100,
    value=70
)

writing_score = st.number_input(
    "Writing Score",
    min_value=0,
    max_value=100,
    value=75
)

# =========================
# MANUAL ENCODING
# =========================

gender_map = {
    "Female": 0,
    "Male": 1
}

gender = gender_map[gender]

# =========================
# PREDICT BUTTON
# =========================

if st.button("Predict"):

    input_data = np.array([[
        gender,
        1,
        2,
        1,
        1,
        reading_score,
        writing_score
    ]])

    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Student Will Pass")
    else:
        st.error("Student May Fail")
