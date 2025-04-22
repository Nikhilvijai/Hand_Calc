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
        # Loading time
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

                
        # Load the trained model
        model_dict = pickle.load(open('./model.p', 'rb'))
        model = model_dict['model']


        # Initialize MediaPipe Hands
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

        # Default Labels
        default_labels_dict = {
            0: 'waiting', 1: '0', 2: '1', 3: '2', 4: '3', 5: '4', 6: '5', 7: '6', 8: '7', 9: '8', 
            10: '9',11: 'next function',12:'answer'
        }

        # Alternate Labels (Only `0, 2, 3, 4` are valid)
        alt_labels_dict = {
            0: 'waiting', 2: '+', 3: '-', 4: '*',5:'/',6:'^'  # Only these values are valid
        }

        labels_dict = default_labels_dict  # Start with default labels
        last_prediction = None  # Stores the last recognized gesture
        waiting_for_alt_gesture = False  # Flag to track if waiting for an alternate gesture

        placeholder = st.empty()

        while True:
            data_aux = []
            x_ = []
            y_ = []

            ret, frame = cap.read()
            if not ret:
                break  # Exit loop if frame is not captured
            
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
                        continue  # Avoid errors if no hand landmarks are detected

                    for landmark in hand_landmarks.landmark:
                        data_aux.append(landmark.x - min(x_))
                        data_aux.append(landmark.y - min(y_))

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                # Make a prediction
                prediction = model.predict(np.array(data_aux).reshape(1, -1))  # Ensure correct input shape
                predicted_index = int(prediction[0])
                predicted_character = labels_dict.get(predicted_index, 'Unknown')

                # If "next function" is detected, switch to alt labels and wait for a valid gesture
                if predicted_character == 'next function' and not waiting_for_alt_gesture:
                    labels_dict = alt_labels_dict  # Switch to alternate labels
                    waiting_for_alt_gesture = True  # Now we wait for a valid alternate gesture
                    #print("Switched to alternate labels. Waiting for a valid alt gesture...")
                    current_time = time.time()
                    cooldown_time = 1.0
                    if current_time - last_prediction_time < cooldown_time:
                        continue  
                    last_prediction = predicted_character
                    last_prediction_time = current_time 

                # If waiting for an alt gesture, allow only `0, 2, 3, 4` to be detected
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
                        labels_dict = default_labels_dict  # Switch back to default labels
                        waiting_for_alt_gesture = False  # Reset flag after switching back
                    else:
                        continue  # Ignore other gestures and keep waiting

                # Print normally if we're in default mode
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
                    last_prediction = predicted_character  # Update last prediction

                # Display the bounding box and prediction text
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
                            
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Exit with 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  
        
        cap.release()
        cv2.destroyAllWindows()



if s:
    active()