import streamlit as st
import cv2
import tempfile
import torch
import numpy as np
import pygame
from pathlib import Path

st.title("Theft Detection System 🔍🚨")

# Load YOLOv5 model
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

model = load_model()

# Initialize alarm
alarm_path = "Alarm/alarm.wav"
pygame.init()
pygame.mixer.init()
pygame.mixer.music.load(alarm_path)

# Upload video
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(video_path)
    run_detection = st.button("Start Detection")

    if run_detection:
        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()

        pts = [(100, 100), (500, 100), (500, 400), (100, 400)]  # Example ROI rectangle

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            detections = results.pandas().xyxy[0]

            polygon = np.array(pts, np.int32)
            cv2.polylines(frame, [polygon], True, (0, 255, 255), 2)

            alarm_triggered = False

            for _, det in detections.iterrows():
                if det['name'] == 'person':
                    x1, y1, x2, y2 = map(int, [det['xmin'], det['ymin'], det['xmax'], det['ymax']])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    if cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        alarm_triggered = True
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if alarm_triggered:
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play()
            else:
                pygame.mixer.music.stop()

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        cap.release()
