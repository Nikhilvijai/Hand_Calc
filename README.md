
![screenshot](TITLE.png)

# ✋🤖 Hand Gesture Calculator

This project is a **hand-gesture-based calculator** built with **Streamlit**, **OpenCV**, **MediaPipe**, and **scikit-learn**. The application captures hand gestures via webcam and interprets them to perform mathematical operations — no keyboard needed!

---

## ⚙ Installation

following libraries must be installed

<pre> pip install opencv-python mediapipe scikit-learn streamlit matplotlib </pre>


## 🧠 Features

- Real-time hand gesture recognition using **MediaPipe**
- Machine learning classification with **Random Forest**
- Dynamic switching between number and operator gestures
- Live video preview inside  **Streamlit** 
- Fully functional math parser: `+`, `-`, `*`, `/`, `^`

---

## ⚒ Working

### step 1:
  - run the collect_data.py to collect the required hand gestures based on our preferences which then is stored in directory "data".
  - these gestures must identify numbers from (0-9),"next function"( to change to operands),"result"(to print the result")
### step 2:
  - run create_dataset.py to create the dataset for identifying landmarks of your hand
  - different guestures are overlooked on the data directory to create the landmarks and fetch coordinates
  - the result is stored in "data.pickle"
### step 3:
  - run train_model.py to train the model
  - the model stored in "model.p"
### step 4:
  - run stream.py to open streamlit
  - assign labels to your gestures as your wish 
  - you can test your model here  

## 📁 Project Structure

```bash
.
├── data/                    # Collected gesture image dataset
├── data.pickle              # Extracted landmark data and labels
├── model.p                  # Trained ML model (Random Forest)
├── TITLE.png                # Banner image for Streamlit UI
├── collect_data.py          # Script to collect gesture images
├── extract_landmarks.py     # Extracts hand landmark features
├── train_model.py           # Trains and saves the ML model
├── stream.py                   # Streamlit  for the calculator
└── README.md
```

