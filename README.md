
![screenshot](TITLE.png)

# ✋🤖 Hand Gesture Calculator

This project is a **hand-gesture-based calculator** built with **Streamlit**, **OpenCV**, **MediaPipe**, and **scikit-learn**. The application captures hand gestures via webcam and interprets them to perform mathematical operations — no keyboard needed!

---

## 📸 Demo

> Simply show your hand gestures to the camera to input numbers and operators. Perform calculations by making a gesture for "answer".

---

## ⚙ Installation

following libraries must be installed

<pre> ```bash pip install opencv-python mediapipe scikit-learn streamlit matplotlib``` </pre>


## 🧠 Features

- Real-time hand gesture recognition using **MediaPipe**
- Machine learning classification with **Random Forest**
- Dynamic switching between number and operator gestures
- Live video preview inside  **Streamlit** 
- Fully functional math parser: `+`, `-`, `*`, `/`, `^`

---

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
├── app.py                   # Streamlit app for the calculator
└── README.md

