import cv2
import mediapipe as mp
import streamlit as st
import tempfile
import os
import gc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Initialize MediaPipe's pose detection model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Streamlit interface
st.title("Human Motion Recognition System - Chendong\n\nAcknowledgements: AOU, FJTCM")
st.markdown("""
<style>
    .main-title {
        font-size: 24px !important;
    }
    .sub-title {
        font-size: 14px !important;
    }
</style>
""", unsafe_allow_html=True)

# Upload video file
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

@st.cache(allow_output_mutation=True)
def load_model():
    return mp_pose.Pose()

pose = load_model()

@st.cache(allow_output_mutation=True)
def process_video(file_data):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(file_data)

    cap = cv2.VideoCapture(tfile.name)
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = 5  # Process every 5 frames
    landmarks = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Calculate video duration
        if frame_count == 1:
            video_duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
            if video_duration > 10:
                st.error("Video duration exceeds 10 seconds, please upload a shorter video.")
                cap.release()
                os.remove(tfile.name)
                return []

        if frame_count % frame_interval != 0:
            continue

        # Reduce resolution
        frame = cv2.resize(frame, (640, 480))

        # Convert the image from BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Perform pose detection
        results = pose.process(image)

        if results.pose_landmarks:
            landmarks.append([(lm.x, lm.y) for lm in results.pose_landmarks.landmark])

        # Force garbage collection
        gc.collect()

    cap.release()
    os.remove(tfile.name)
    return np.array(landmarks)

def plot_landmarks(landmarks, frame_idx, ax):
    ax.clear()
    connections = mp_pose.POSE_CONNECTIONS
    landmark_coords = landmarks[frame_idx]
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        start_coords = landmark_coords[start_idx]
        end_coords = landmark_coords[end_idx]
        ax.plot([start_coords[0], end_coords[0]], [start_coords[1], end_coords[1]], 'r-')
    ax.scatter(*zip(*landmark_coords), c='r', s=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect('equal')

if uploaded_file is not None:
    landmarks = process_video(uploaded_file.read())
    if len(landmarks) > 0:
        st.sidebar.title("Control Panel")
        frame_idx = st.sidebar.slider("Select Frame", 0, len(landmarks) - 1, 0)

        fig, ax = plt.subplots()
        plot_landmarks(landmarks, frame_idx, ax)
        st.pyplot(fig)
        
        # Download button for landmarks data
        column_names = [
            "Nose_x", "Nose_y", "LeftEyeInner_x", "LeftEyeInner_y", "LeftEye_x", "LeftEye_y",
            "LeftEyeOuter_x", "LeftEyeOuter_y", "RightEyeInner_x", "RightEyeInner_y",
            "RightEye_x", "RightEye_y", "RightEyeOuter_x", "RightEyeOuter_y", "LeftEar_x", 
            "LeftEar_y", "RightEar_x", "RightEar_y", "MouthLeft_x", "MouthLeft_y", 
            "MouthRight_x", "MouthRight_y", "LeftShoulder_x", "LeftShoulder_y", 
            "RightShoulder_x", "RightShoulder_y", "LeftElbow_x", "LeftElbow_y", 
            "RightElbow_x", "RightElbow_y", "LeftWrist_x", "LeftWrist_y", "RightWrist_x", 
            "RightWrist_y", "LeftPinky_x", "LeftPinky_y", "RightPinky_x", "RightPinky_y", 
            "LeftIndex_x", "LeftIndex_y", "RightIndex_x", "RightIndex_y", "LeftThumb_x", 
            "LeftThumb_y", "RightThumb_x", "RightThumb_y", "LeftHip_x", "LeftHip_y", 
            "RightHip_x", "RightHip_y", "LeftKnee_x", "LeftKnee_y", "RightKnee_x", 
            "RightKnee_y", "LeftAnkle_x", "LeftAnkle_y", "RightAnkle_x", "RightAnkle_y", 
            "LeftHeel_x", "LeftHeel_y", "RightHeel_x", "RightHeel_y", "LeftFootIndex_x", 
            "LeftFootIndex_y", "RightFootIndex_x", "RightFootIndex_y"
        ]
        landmarks_df = pd.DataFrame(landmarks.reshape(-1, 33 * 2), columns=column_names)
        csv = landmarks_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Landmarks Data as CSV",
            data=csv,
            file_name='landmarks.csv',
            mime='text/csv'
        )
