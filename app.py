import streamlit as st
from src.courtiq.models.predict import predict_from_last_n

st.set_page_config(page_title="CourtIQ", layout="centered")
st.title("CourtIQ Player Predictions")

player = st.text_input("Player name", "Kevin Durant")
n = st.slider("Last N games", 1, 10, 5)

if st.button("Predict"):
    result = predict_from_last_n(player_name=player, n=n)
    st.subheader("Prediction Result")
    st.json(result)