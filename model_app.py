import streamlit as st
import joblib

model = joblib.load(filename="regression.joblib")

size = st.number_input("House size")
nb_rooms = st.number_input("Number of rooms")
garden = st.number_input("Garden")

result = model.predict([[size, nb_rooms, garden]])
st.write(f"Predicted price: {result[0]}")