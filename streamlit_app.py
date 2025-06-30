import streamlit as st
import tempfile
import os
from inference import process_video

st.set_page_config(page_title="Vehicle Tracker", layout="centered")
st.title("🚗 Vehicle Tracking & Counting")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_input:
        tmp_input.write(uploaded_file.read())
        input_path = tmp_input.name

    output_path = input_path.replace(".mp4", "_output.mp4")

    st.info("Processing started. This might take a minute depending on video size.")
    process_video(input_path, output_path)
    st.success("Processing complete!")

    st.video(output_path)
    with open(output_path, "rb") as f:
        st.download_button("Download Output Video", f, file_name="output.mp4")
