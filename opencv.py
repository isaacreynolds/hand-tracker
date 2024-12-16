import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
if ret:
    h, w, _ = frame.shape

prev_x, prev_y = 0, 0
smooth_factor = 0.2

def is_index_finger_raised(hand_landmarks):
    
    landmarks = hand_landmarks.landmark

    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    index_dip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP].y

    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP].y

    is_index_extended = index_tip < index_dip
    is_others_not_extended = (
        index_tip < middle_tip and
        index_tip < ring_tip and
        index_tip < pinky_tip
    )

    return is_index_extended and is_others_not_extended

def is_pinch_pose(hand_landmarks):
    
    landmarks = hand_landmarks.landmark
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    distance = np.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
    return distance < 0.05

def is_click_pose(hand_landmarks):
    
    landmarks = hand_landmarks.landmark
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    distance = np.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
    return distance < 0.05

dragging = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_results = hands.process(frame_rgb)
    

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            screen_width, screen_height = pyautogui.size()
            screen_x = int(x / w * screen_width)
            screen_y = int(y / h * screen_height)
            smooth_x = prev_x + (screen_x - prev_x) * smooth_factor
            smooth_y = prev_y + (screen_y - prev_y) * smooth_factor
            pyautogui.moveTo(smooth_x, smooth_y)
            prev_x, prev_y = smooth_x, smooth_y

            if is_pinch_pose(hand_landmarks):
                
                if not dragging:
                    pyautogui.mouseDown()
                    dragging = True
            else:
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False

            if is_click_pose(hand_landmarks):
                
                pyautogui.click()

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Tracking', frame)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
