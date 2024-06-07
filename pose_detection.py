import cv2
import mediapipe as mp
import streamlit as st
import tempfile
import os

# 初始化MediaPipe的姿态检测模型
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Streamlit界面
st.title("人体动作识别系统")

# 上传视频文件
uploaded_file = st.file_uploader("选择一个视频文件", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 将图像从BGR转换为RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # 进行姿态检测
        results = pose.process(image)

        # 将图像转换回BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 绘制姿态检测结果
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 将图像显示在Streamlit中
        stframe.image(image, channels='BGR')

    cap.release()
    os.remove(tfile.name)
