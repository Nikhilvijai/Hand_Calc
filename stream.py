import streamlit as st
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import re


st.image("TITLE.png")
st.header("Calculate With Your Hands!!")
FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

st.write("for calculation please press enter and provide required hand gesture......")
s = st.button(label="enter")


def active():
        sum = ""
        # loading time
        cooldown_time = 1.0  # seconds
        last_prediction_time = 0

        #calculation
        def calc():
            expression = sum
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
            st.success(f"RESULT: {float(tokens[0])}")

                
        # load the trained model
        model_dict = pickle.load(open('./model.p', 'rb'))
        model = model_dict['model']


        # initialize MediaPipe hands
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

        # default labels
        default_labels_dict = {
            0: 'waiting', 1: '0', 2: '1', 3: '2', 4: '3', 5: '4', 6: '5', 7: '6', 8: '7', 9: '8', 
            10: '9',11: 'next function',12:'answer'
        }

        # alternate labels 
        alt_labels_dict = {
            0: 'waiting', 2: '+', 3: '-', 4: '*',5:'/',6:'^'  # Only these values are valid
        }

        labels_dict = default_labels_dict 
        last_prediction = None  
        waiting_for_alt_gesture = False  

        placeholder = st.empty()

        while True:
            data_aux = []
            x_ = []
            y_ = []

            ret, frame = cap.read()
            if not ret:
                break  
            
            frame = cv2.flip(frame, 1)
            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                for hand_landmarks in results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        x_.append(landmark.x)
                        y_.append(landmark.y)

                    if not x_ or not y_:
                        continue  

                    for landmark in hand_landmarks.landmark:
                        data_aux.append(landmark.x - min(x_))
                        data_aux.append(landmark.y - min(y_))

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                # make a prediction
                prediction = model.predict(np.array(data_aux).reshape(1, -1))  
                predicted_index = int(prediction[0])
                predicted_character = labels_dict.get(predicted_index, 'Unknown')

                if predicted_character == 'next function' and not waiting_for_alt_gesture:
                    labels_dict = alt_labels_dict  
                    waiting_for_alt_gesture = True  
                    current_time = time.time()
                    cooldown_time = 1.0
                    if current_time - last_prediction_time < cooldown_time:
                        continue  
                    last_prediction = predicted_character
                    last_prediction_time = current_time 

                elif waiting_for_alt_gesture:
                    if predicted_index in alt_labels_dict:
                        predicted_character = alt_labels_dict[predicted_index]
                        current_time = time.time()
                        if current_time - last_prediction_time < cooldown_time:
                            continue  
                        if predicted_character != 'waiting':
                                sum += str(predicted_character)
                                placeholder.markdown(f"### Expression: `{sum}`")
                        last_prediction = predicted_character
                        last_prediction_time = current_time 
                        labels_dict = default_labels_dict  
                        waiting_for_alt_gesture = False  
                    else:
                        continue  

                elif not waiting_for_alt_gesture and predicted_character != last_prediction:
                    current_time = time.time()
                    if current_time - last_prediction_time < cooldown_time:
                            continue  
                    if predicted_character != 'waiting':
                                if predicted_character == 'answer':
                                    calc()
                                    break
                                else:
                                    sum += str(predicted_character)
                                    placeholder.markdown(f"### Expression: `{sum}`")
                    last_prediction = predicted_character
                    last_prediction_time = current_time
                    last_prediction = predicted_character 

                # display the bounding box and prediction text
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
                            
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # exit with 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  
        
        cap.release()
        cv2.destroyAllWindows()



if s:
    active()
