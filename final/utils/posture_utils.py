import cv2
import mediapipe as mp
import math

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def angle_between(p1, p2):
    return math.degrees(math.atan2(p2.y - p1.y, p2.x - p1.x))

def detect_posture(frame, holistic):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)

    posture_status = "Neutral"

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        def get_point(landmark_id):
            return landmarks[landmark_id] if landmarks[landmark_id].visibility > 0.6 else None

        left_shoulder = get_point(mp_holistic.PoseLandmark.LEFT_SHOULDER)
        right_shoulder = get_point(mp_holistic.PoseLandmark.RIGHT_SHOULDER)
        left_hip = get_point(mp_holistic.PoseLandmark.LEFT_HIP)
        right_hip = get_point(mp_holistic.PoseLandmark.RIGHT_HIP)
        left_wrist = get_point(mp_holistic.PoseLandmark.LEFT_WRIST)
        right_wrist = get_point(mp_holistic.PoseLandmark.RIGHT_WRIST)
        nose = get_point(mp_holistic.PoseLandmark.NOSE)
        left_ear = get_point(mp_holistic.PoseLandmark.LEFT_EAR)
        right_ear = get_point(mp_holistic.PoseLandmark.RIGHT_EAR)

        # 1. Slouching based on neck to hip angle
        if nose and left_hip:
            neck_hip_angle = angle_between(nose, left_hip)
            if neck_hip_angle < 80 or neck_hip_angle > 100:
                posture_status = "Slouching"

        # 2. Head tilt detection
        if left_ear and right_ear:
            if abs(left_ear.y - right_ear.y) > 0.1:
                posture_status = "Head Tilted"

        # 3. Arms crossed detection
        if left_wrist and right_wrist:
            if abs(left_wrist.x - right_wrist.x) < 0.1 and abs(left_wrist.y - right_wrist.y) < 0.1:
                posture_status = "Arms Crossed"

        # 4. Touching face
        if left_wrist and nose:
            if abs(left_wrist.x - nose.x) < 0.1 and abs(left_wrist.y - nose.y) < 0.1:
                posture_status = "Touching Face"
        if right_wrist and nose:
            if abs(right_wrist.x - nose.x) < 0.1 and abs(right_wrist.y - nose.y) < 0.1:
                posture_status = "Touching Face"

        # 5. Leaning detection
        if left_shoulder and right_shoulder and left_hip and right_hip:
            shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
            hip_diff = abs(left_hip.y - right_hip.y)
            if shoulder_diff > 0.1 or hip_diff > 0.1:
                posture_status = "Leaning"

        # 6. Confident posture as fallback
        if posture_status == "Neutral":
            if left_shoulder and left_hip and right_shoulder and right_hip:
                if left_shoulder.y < left_hip.y and right_shoulder.y < right_hip.y:
                    posture_status = "Confident"

    return posture_status, results
