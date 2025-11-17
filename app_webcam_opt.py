import streamlit as st
from deepface import DeepFace
import cv2
from collections import deque, Counter
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import time

st.set_page_config(page_title="AI Emotion Detector (Webcam)", page_icon="📸", layout="centered")

st.title("📸 AI Emotion Detector - Webcam Live")
st.write("Open your webcam and leave that AI recognize your emotion in real-time!")

class EmotionDetector(VideoTransformerBase):
    def __init__(self):
        self.last_result = "N/A"
        self.results_buffer = deque(maxlen=5)  # Ultimi 5 risultati per media mobile
        self.last_analysis_time = 0
        self.analysis_interval = 0.2  # secondi tra un'analisi e l'altra

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        current_time = time.time()

        try:
            # Analizza solo se è passato l'intervallo di tempo
            if current_time - self.last_analysis_time > self.analysis_interval:
                result = DeepFace.analyze(img_path=img, actions=['emotion'], enforce_detection=False)
                dominant_emotion = result[0]['dominant_emotion']

                # Aggiungi al buffer e aggiorna la media mobile
                self.results_buffer.append(dominant_emotion)
                self.last_result = Counter(self.results_buffer).most_common(1)[0][0]

                self.last_analysis_time = current_time

            # Disegna il testo sopra al volto
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, f"Emotion: {self.last_result}", (50, 50),
                        font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        except Exception as e:
            print("Errore analisi:", e)

        return img

webrtc_streamer(
    key="emotion_detector",
    video_processor_factory=EmotionDetector,
    media_stream_constraints={"video": True, "audio": False},
)

st.markdown("👉 Come close to the webcam and try different facial expression: 😃 😢 😡 😲 😐 🤮 😧")

#streamlit run app_webcam_opt.py