import cv2
import numpy as np
from keras.models import load_model

emotion_model_path = "emotion-model/fer2013_big_XCEPTION.54-0.66.hdf5"  # Place the pretrained model here
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_classifier = load_model(emotion_model_path)


def detect_emotion(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    emotions_detected = []
    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (64, 64))
        roi_gray = roi_gray.astype("float") / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        preds = emotion_classifier.predict(roi_gray)[0]
        emotion = emotion_labels[np.argmax(preds)]
        emotions_detected.append((emotion, (x, y, w, h)))

    return emotions_detected
