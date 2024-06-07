import cv2
import mediapipe as mp
import streamlit as st
import tempfile
import os
import gc
import matplotlib.pyplot as plt

# 初始化MediaPipe的姿态检测模型
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Streamlit界面
st.title("人体动作识别系统")

# 上传视频文件
uploaded_file = st.file_uploader("选择一个视频文件", type=["mp4", "avi", "mov"])

landmarks = []

def process_video(uploaded_file):
    global landmarks
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = 5  # 每隔5帧处理一次

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1

        # 计算视频时长
        if frame_count == 1:
            video_duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
            if video_duration > 10:
                st.error("视频时长超过10秒，请上传一个更短的视频。")
                cap.release()
                os.remove(tfile.name)
                return

        if frame_count % frame_interval != 0:
            continue

        # 降低分辨率
        frame = cv2.resize(frame, (640, 480))

        # 将图像从BGR转换为RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # 进行姿态检测
        results = pose.process(image)

        if results.pose_landmarks:
            landmarks.append(results.pose_landmarks)

        # 强制垃圾回收
        gc.collect()

    cap.release()
    os.remove(tfile.name)

def plot_landmarks(landmark, ax):
    ax.clear()
    connections = mp_pose.POSE_CONNECTIONS
    landmark_coords = [(lm.x, lm.y) for lm in landmark.landmark]
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
    if landmarks:
        st.sidebar.title("控制面板")
        frame_idx = st.sidebar.slider("选择帧", 0, len(landmarks) - 1, 0)

        fig, ax = plt.subplots()
        plot_landmarks(landmarks[frame_idx], ax)
        st.pyplot(fig)
