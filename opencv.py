import cv2
import numpy as np
import time
import mediapipe as mp

# Initialize Mediapipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Check if OpenCV is built with CUDA support
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    use_cuda = True
    print("CUDA is available. Using GPU for processing.")
else:
    use_cuda = False
    print("CUDA is not available. Using CPU for processing.")

# Function to draw landmarks and lines on the frame
def draw_landmarks(frame, landmarks):
    for i, landmark in enumerate(landmarks):
        x, y = int(landmark[0]), int(landmark[1])
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Increase circle size for better visibility
        if i > 0:
            prev_x, prev_y = int(landmarks[i-1][0]), int(landmarks[i-1][1])
            cv2.line(frame, (prev_x, prev_y), (x, y), (255, 0, 0), 2)  # Draw lines between landmarks

def is_index_finger_extended(landmarks):
    index_tip = landmarks[8][1]
    index_mcp = landmarks[5][1]
    return index_tip < index_mcp

def draw_on_frame(frame, x, y):
    global drawings
    timestamp = time.time()
    drawings.append((x, y, timestamp))

def toggle_drawing():
    global drawing
    drawing = not drawing

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
if ret:
    h, w, _ = frame.shape

drawing = False
draw_enabled = True  # Variable to track drawing state
draw_color = (0, 255, 0)
draw_thickness = 5  # Increase thickness for better visibility
canvas = None
drawings = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    if canvas is None:
        canvas = np.zeros_like(frame)

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if use_cuda:
        # Upload frame to GPU
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame_rgb)
        # Perform hand detection on GPU
        gpu_frame_rgb = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2RGB)
        frame_rgb = gpu_frame_rgb.download()

    # Perform hand detection
    results = hands.process(frame_rgb)
    hand_landmarks = []

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            for lm in hand_landmark.landmark:
                hand_landmarks.append([lm.x * w, lm.y * h])

    if len(hand_landmarks) > 0:
        index_finger_tip = hand_landmarks[8]
        x, y = int(index_finger_tip[0]), int(index_finger_tip[1])

        if draw_enabled and is_index_finger_extended(hand_landmarks):
            draw_on_frame(canvas, x, y)

        draw_landmarks(frame, hand_landmarks)

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
    elif key == ord('d'):  # Toggle drawing when 'd' is pressed
        toggle_drawing()

cap.release()
cv2.destroyAllWindows()
