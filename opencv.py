import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# Set OpenCV to use CUDA if available
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    print("CUDA is available. Using GPU acceleration.")
    cap.set(cv2.CAP_PROP_BACKEND, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
else:
    print("CUDA is not available. Checking for Apple Silicon (Metal).")
    try:
        cv2.ocl.setUseOpenCL(True)
        if cv2.ocl.haveOpenCL():
            print("Apple Silicon (Metal) is available. Using GPU acceleration.")
            cap.set(cv2.CAP_PROP_BACKEND, cv2.CAP_AVFOUNDATION)
        else:
            raise Exception("Apple Silicon (Metal) is not available.")
    except Exception as e:
        print(f"Apple Silicon (Metal) is not available. Using CPU. Error: {e}")

ret, frame = cap.read()
if ret:
    h, w, _ = frame.shape

smooth_factor = 0.2
drawing = False
draw_color = (255, 255, 255)
draw_thickness = 15  # Increase thickness for better visibility
canvas = None
drawings = []

def is_index_finger_extended(hand_landmarks):
    landmarks = hand_landmarks.landmark
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    return index_tip < index_mcp and index_tip < middle_tip

def is_thumbs_up(hand_landmarks):
    landmarks = hand_landmarks.landmark
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP].y
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP].y
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    return thumb_tip < thumb_ip and thumb_tip < index_tip

def draw_on_frame(frame, x, y):
    global drawing, drawings
    if drawing:
        timestamp = time.time()
        drawings.append((x, y, timestamp))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    if canvas is None:
        canvas = np.zeros_like(frame)

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_results = hands.process(frame_rgb)
    
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            if is_thumbs_up(hand_landmarks):
                canvas = np.zeros_like(frame)
                drawings = []
                drawing = False
            elif is_index_finger_extended(hand_landmarks):
                drawing = True
                draw_on_frame(canvas, x, y)
            else:
                drawing = False

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    current_time = time.time()
    drawings = [(x, y, t) for x, y, t in drawings if current_time - t < 5]

    for x, y, t in drawings:
        cv2.circle(canvas, (x, y), draw_thickness, draw_color, -1)

    combined_frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    cv2.namedWindow('Hand Tracking', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Hand Tracking', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Hand Tracking', combined_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Exiting...")
        break
    elif key == ord('c'):  # Clear the canvas when 'c' is pressed
        canvas = np.zeros_like(frame)
        drawings = []
        print("cleared canvas")

cap.release()
cv2.destroyAllWindows()
