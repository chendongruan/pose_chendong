import cv2
import mediapipe as mp
import streamlit as st
import tempfile
import os
import gc
import matplotlib.pyplot as plt
import numpy as np

# Initialize MediaPipe's pose detection model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Streamlit interface
st.title("Human Motion Recognition System")

# Upload video file
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

landmarks = []

@st.cache(allow_output_mutation=True)
def load_model():
    return mp_pose.Pose()

pose = load_model()

def process_video(uploaded_file):
    global landmarks
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = 5  # Process every 5 frames

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
                return

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
    landmarks = np.array(landmarks)

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
    process_video(uploaded_file)
    if len(landmarks) > 0:
        st.sidebar.title("Control Panel")
        frame_idx = st.sidebar.slider("Select Frame", 0, len(landmarks) - 1, 0)

        fig, ax = plt.subplots()
        plot_landmarks(landmarks, frame_idx, ax)
        st.pyplot(fig)