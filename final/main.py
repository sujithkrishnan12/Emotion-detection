import cv2
import time
import numpy as np
from utils.emotion_utils import detect_emotion
from utils.posture_utils import detect_posture
from reportlab.pdfgen import canvas
import os
import mediapipe as mp

# Setup snapshot folder
snapshots_dir = "report/snapshots"
os.makedirs(snapshots_dir, exist_ok=True)


def save_snapshot(frame, label, count):
    path = os.path.join(snapshots_dir, f"{label}_{count}.jpg")
    cv2.imwrite(path, frame)
    return path


def generate_report(data):
    c = canvas.Canvas("report/final_report.pdf")
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 800, "Emotion & Posture Analysis Report")
    y = 770
    for i, item in enumerate(data):
        c.setFont("Helvetica", 12)
        c.drawString(100, y, f"Frame {i + 1}")
        y -= 20
        c.drawString(120, y, f"Emotion: {item['emotion']}")
        y -= 20
        c.drawString(120, y, f"Posture: {item['posture']}")
        y -= 20
        c.drawString(120, y, f"Looking at Screen: {item['looking_at_screen']}")
        y -= 30
        if y < 150:
            c.showPage()
            y = 770
        try:
            c.drawImage(item["snapshot"], 120, y - 100, width=120, height=90)
            y -= 110
        except:
            continue
    c.save()


def is_looking_at_screen(landmarks):
    # Extract the left and right eye landmarks
    left_eye_landmarks = [landmarks[i] for i in range(33, 133)]  # FACEMESH_LEFT_EYE is a range of indices
    right_eye_landmarks = [landmarks[i] for i in range(133, 233)]  # FACEMESH_RIGHT_EYE is another range

    # Calculate the center of the left and right eye
    left_eye_center = (
        sum([landmarks[i].x for i in range(33, 133)]) / len(left_eye_landmarks),
        sum([landmarks[i].y for i in range(33, 133)]) / len(left_eye_landmarks)
    )

    right_eye_center = (
        sum([landmarks[i].x for i in range(133, 233)]) / len(right_eye_landmarks),
        sum([landmarks[i].y for i in range(133, 233)]) / len(right_eye_landmarks)
    )

    # Checking if both eyes are aligned (as a simple heuristic for looking at the screen)
    eye_distance = abs(left_eye_center[0] - right_eye_center[0])  # Horizontal difference between eyes
    threshold = 0.02  # A threshold to determine if eyes are centered

    if eye_distance < threshold:
        return "Yes"  # Looking at the screen
    return "No"  # Not looking at the screen


def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    mp_face_mesh = mp.solutions.face_mesh

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution for better FPS
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_data = []
    snapshot_count = 0
    frame_count = 0

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    prev_time = time.time()
    emotions = []
    dominant_emotion = "Unknown"

    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
            mp_holistic.Holistic(static_image_mode=False, model_complexity=1) as holistic:

        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Run emotion detection every 3 frames
            if frame_count % 3 == 0:
                emotions = detect_emotion(frame)
                dominant_emotion = emotions[0][0] if emotions else "Unknown"

            # Draw emotion results
            for emotion, (x, y, w, h) in emotions:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # Posture detection
            posture_status, results = detect_posture(frame, holistic)  # Pass holistic here
            cv2.putText(frame, f"Posture: {posture_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
            cv2.putText(frame, f"Emotion: {dominant_emotion}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 100),
                        2)

            # Face mesh landmarks for screen check
            if results.face_landmarks:
                looking_at_screen = is_looking_at_screen(results.face_landmarks.landmark)

            # Calculate and display FPS
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)

            # Show frame in window
            cv2.imshow("Real-time Emotion & Posture Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                snap_path = save_snapshot(frame, dominant_emotion, snapshot_count)
                frame_data.append({
                    "emotion": dominant_emotion,
                    "posture": posture_status,
                    "looking_at_screen": looking_at_screen,
                    "snapshot": snap_path
                })
                print(f"[✓] Snapshot {snapshot_count} saved.")
                snapshot_count += 1
            elif key == ord('q'):
                break

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

        if frame_data:
            generate_report(frame_data)
            print("[✓] PDF report generated at report/final_report.pdf")
        else:
            print("[!] No snapshots were saved. Press 's' to save snapshots.")


if __name__ == "__main__":
    main()
