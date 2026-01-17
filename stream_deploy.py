import streamlit as st
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import re
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av


st.image("TITLE.png")
st.header("Calculate With Your Hands!!")

st.write("Click 'START' to begin using hand gestures for calculation")

# Load model once
@st.cache_resource
def load_model():
    model_dict = pickle.load(open('./model.p', 'rb'))
    return model_dict['model']

model = load_model()

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Default and alternate labels
default_labels_dict = {
    0: 'waiting', 1: '0', 2: '1', 3: '2', 4: '3', 5: '4', 6: '5', 7: '6', 8: '7', 9: '8', 
    10: '9', 11: 'next function', 12: 'answer'
}

alt_labels_dict = {
    0: 'waiting', 2: '+', 3: '-', 4: '*', 5: '/', 6: '^'
}

# Session state for expression
if 'expression' not in st.session_state:
    st.session_state.expression = ""
if 'result' not in st.session_state:
    st.session_state.result = None

# Display area
expression_placeholder = st.empty()
result_placeholder = st.empty()

expression_placeholder.markdown(f"### Expression: `{st.session_state.expression}`")
if st.session_state.result is not None:
    result_placeholder.success(f"RESULT: {st.session_state.result}")


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)
        self.labels_dict = default_labels_dict
        self.last_prediction = None
        self.waiting_for_alt_gesture = False
        self.last_prediction_time = 0
        self.cooldown_time = 1.0

    def calc(self, expression):
        tokens = re.findall(r'\d+|\^|\/|\*|\+|\-', expression)
        precedence = ['^', '/', '*', '+', '-']
        for op in precedence:
            i = 0
            while i < len(tokens):
                if tokens[i] == op:
                    left = float(tokens[i - 1])
                    right = float(tokens[i + 1])
                    if op == '^':
                        result = left ** right
                    elif op == '/':
                        result = left / right
                    elif op == '*':
                        result = left * right
                    elif op == '+':
                        result = left + right
                    elif op == '-':
                        result = left - right
                    tokens[i-1:i+2] = [str(result)]
                    i = 0
                else:
                    i += 1
        return float(tokens[0])

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        H, W, _ = img.shape
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            data_aux = []
            x_ = []
            y_ = []

            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                if x_ and y_:
                    for landmark in hand_landmarks.landmark:
                        data_aux.append(landmark.x - min(x_))
                        data_aux.append(landmark.y - min(y_))

                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10
                    x2 = int(max(x_) * W) - 10
                    y2 = int(max(y_) * H) - 10

                    prediction = model.predict(np.array(data_aux).reshape(1, -1))
                    predicted_index = int(prediction[0])
                    predicted_character = self.labels_dict.get(predicted_index, 'Unknown')

                    current_time = time.time()

                    if predicted_character == 'next function' and not self.waiting_for_alt_gesture:
                        if current_time - self.last_prediction_time >= self.cooldown_time:
                            self.labels_dict = alt_labels_dict
                            self.waiting_for_alt_gesture = True
                            self.last_prediction = predicted_character
                            self.last_prediction_time = current_time

                    elif self.waiting_for_alt_gesture:
                        if predicted_index in alt_labels_dict:
                            predicted_character = alt_labels_dict[predicted_index]
                            if current_time - self.last_prediction_time >= self.cooldown_time:
                                if predicted_character != 'waiting':
                                    st.session_state.expression += str(predicted_character)
                                self.last_prediction = predicted_character
                                self.last_prediction_time = current_time
                                self.labels_dict = default_labels_dict
                                self.waiting_for_alt_gesture = False

                    elif not self.waiting_for_alt_gesture and predicted_character != self.last_prediction:
                        if current_time - self.last_prediction_time >= self.cooldown_time:
                            if predicted_character != 'waiting':
                                if predicted_character == 'answer':
                                    try:
                                        st.session_state.result = self.calc(st.session_state.expression)
                                    except:
                                        st.session_state.result = "Error"
                                else:
                                    st.session_state.expression += str(predicted_character)
                            self.last_prediction = predicted_character
                            self.last_prediction_time = current_time

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(img, predicted_character, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# WebRTC configuration for STUN servers
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="hand-gesture-calculator",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)

# Clear button
if st.button("Clear Expression"):
    st.session_state.expression = ""
    st.session_state.result = None
    st.rerun()
