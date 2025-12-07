
import streamlit as st
import cv2
from hand_boundary import process_frame
import numpy as np

st.title("Hand Boundary Detection - Live Webcam")

run = st.checkbox('Start Webcam')

FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("Cannot access webcam")
        break

    frame = process_frame(frame)

    # Convert BGR to RGB for Streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

cap.release()
