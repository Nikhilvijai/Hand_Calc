import streamlit as st
import pickle
import cv2
import numpy as np
import time
import re

import mediapipe as mp

st.set_page_config(page_title="Hand Gesture Calculator", layout="centered")

st.image("TITLE.png")
st.header("Calculate With Your Hands!")

if "expression" not in st.session_state:
    st.session_state.expression = ""

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "last_time" not in st.session_state:
    st.session_state.last_time = 0

if "alt_mode" not in st.session_state:
    st.session_state.alt_mode = False

if "running" not in st.session_state:
    st.session_state.running = False

COOLDOWN_TIME = 1.0  


@st.cache_resource
def load_model():
    model_dict = pickle.load(open("./model.p", "rb"))
    return model_dict["model"]

model = load_model()


try:
    mp_hands = mp.solutions.hands
except AttributeError:
    from mediapipe.python.solutions import hands as mp_hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=True,
    min_detection_confidence=0.3
)


default_labels = {
    0: 'waiting', 1: '0', 2: '1', 3: '2', 4: '3', 5: '4',
    6: '5', 7: '6', 8: '7', 9: '8', 10: '9',
    11: 'next function', 12: 'answer'
}

alt_labels = {
    0: 'waiting', 2: '+', 3: '-', 4: '*', 5: '/', 6: '^'
}


def calculate_expression(expr):
    tokens = re.findall(r'\d+|\^|\/|\*|\+|\-', expr)
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

                tokens[i - 1:i + 2] = [str(result)]
                i = 0
            else:
                i += 1

    return float(tokens[0])


col1, col2 = st.columns(2)

with col1:
    if st.button("Start"):
        st.session_state.running = True

with col2:
    if st.button("Reset"):
        st.session_state.expression = ""
        st.session_state.running = False
        st.session_state.alt_mode = False

st.markdown(f"### Expression: `{st.session_state.expression}`")


camera = st.camera_input("Show your hand gesture")

if camera is not None and st.session_state.running:
    file_bytes = np.asarray(bytearray(camera.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    frame = cv2.flip(frame, 1)

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            data_aux = []
            x_, y_ = [], []

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            prediction = model.predict(np.array(data_aux).reshape(1, -1))
            idx = int(prediction[0])

            now = time.time()
            if now - st.session_state.last_time < COOLDOWN_TIME:
                st.image(frame, channels="BGR")
                st.stop()

           
            if not st.session_state.alt_mode:
                char = default_labels.get(idx, "waiting")
            else:
                char = alt_labels.get(idx, "waiting")

           
            if char == "next function":
                st.session_state.alt_mode = True

            elif char == "answer":
                try:
                    result = calculate_expression(st.session_state.expression)
                    st.success(f"RESULT: {result}")
                    st.session_state.running = False
                except Exception:
                    st.error("Invalid Expression")
                st.stop()

            elif char != "waiting":
                st.session_state.expression += char
                st.session_state.alt_mode = False

            st.session_state.last_prediction = char
            st.session_state.last_time = now

            cv2.putText(
                frame, char, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3
            )

    st.image(frame, channels="BGR")
