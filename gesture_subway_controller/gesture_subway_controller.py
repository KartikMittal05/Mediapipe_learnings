import cv2
import mediapipe as mp
import pyautogui
import time

# Setup
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

last_trigger_time = 0
gesture_cooldown = 1.0  # seconds

def fingers_status(landmarks):
    finger_tips = [8, 12, 16, 20]  # index, middle, ring, pinky
    fingers = []

    for tip in finger_tips:
        if landmarks[tip].y < landmarks[tip - 2].y:
            fingers.append(1)  # open
        else:
            fingers.append(0)  # closed

    thumb_open = landmarks[4].x > landmarks[3].x  # right hand check
    return fingers, thumb_open

def detect_custom_gesture(fingers, thumb):
    total = sum(fingers)

    if total == 0:
        return "FIST"
    elif total == 4:
        return "PALM_OPEN"
    elif fingers == [1, 0, 0, 0]:
        return "INDEX_ONLY"
    elif fingers == [1, 0, 0, 0] and thumb:
        return "InDEX_MIDDLE"
    else:
        return None

print("🟢 Gesture control started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    current_time = time.time()

    if result.multi_hand_landmarks:
        handLms = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
        lm = handLms.landmark

        fingers, thumb_open = fingers_status(lm)
        gesture = detect_custom_gesture(fingers, thumb_open)

        if gesture and current_time - last_trigger_time > gesture_cooldown:
            if gesture == "FIST":
                pyautogui.press("down")
                print("⬇️ SLIDE")
            elif gesture == "PALM_OPEN":
                pyautogui.press("up")
                print("⬆️ JUMP")
            elif gesture == "INDEX_MIDDLE":
                pyautogui.press("left")
                print("⬅️ LEFT")
            elif gesture == "INDEX_ONLY":
                pyautogui.press("right")
                print("➡️ RIGHT")
            last_trigger_time = current_time

    cv2.imshow("Custom Gesture Controller", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
