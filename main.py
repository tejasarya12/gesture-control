import cv2
import mediapipe as mp
import pyautogui
import time
import math


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)
pyautogui.PAUSE = 0.02


WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
INDEX_PIP = 6
MIDDLE_TIP = 12
MIDDLE_PIP = 10
RING_TIP = 16
RING_PIP = 14
PINKY_TIP = 20
PINKY_PIP = 18


last_scroll_time = 0
scroll_delay = 0.08

pinch_state = "open"
pinch_count = 0
last_pinch_time = 0
last_screenshot_time = 0

fullscreen_on = False
last_fullscreen_time = 0

prev_index_x = None
tab_locked = False

last_win_ctrl_time = 0
win_ctrl_delay = 1.2

print("Gesture Controller Started")


def finger_up(tip, pip, lm):
    return lm.landmark[tip].y < lm.landmark[pip].y - 0.015


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    now = time.time()
    gesture_text = ""

    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        
        index = finger_up(INDEX_TIP, INDEX_PIP, lm)
        middle = finger_up(MIDDLE_TIP, MIDDLE_PIP, lm)
        ring = finger_up(RING_TIP, RING_PIP, lm)
        pinky = finger_up(PINKY_TIP, PINKY_PIP, lm)

        fingers = [index, middle, ring, pinky]
        four_up = all(fingers)

        thumb_y = lm.landmark[THUMB_TIP].y * h
        finger_ys = [lm.landmark[i].y * h for i in
                     [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]]

        avg_finger_y = sum(finger_ys) / 4

       
        scrolling = False

        if four_up:
            if avg_finger_y < thumb_y - 25:
                scrolling = True
                if now - last_scroll_time > scroll_delay:
                    pyautogui.scroll(60)
                    gesture_text = "SCROLL UP"
                    last_scroll_time = now

            elif avg_finger_y > thumb_y + 25:
                scrolling = True
                if now - last_scroll_time > scroll_delay:
                    pyautogui.scroll(-60)
                    gesture_text = "SCROLL DOWN"
                    last_scroll_time = now

        
        pinch_dist = math.hypot(
            lm.landmark[INDEX_TIP].x - lm.landmark[THUMB_TIP].x,
            lm.landmark[INDEX_TIP].y - lm.landmark[THUMB_TIP].y
        )

        
        if not scrolling:
            if pinch_dist < 0.032 and pinch_state == "open":
                pinch_state = "closed"

            if pinch_dist > 0.070 and pinch_state == "closed":
                pinch_state = "open"
                pinch_count += 1
                last_pinch_time = now

        if now - last_pinch_time > 2:
            pinch_count = 0

        if pinch_count == 3 and now - last_screenshot_time > 2:
            pyautogui.screenshot(f"screenshot_{int(now)}.png")
            gesture_text = "SCREENSHOT"
            pinch_count = 0
            last_screenshot_time = now

      
        spread = abs(lm.landmark[INDEX_TIP].x - lm.landmark[PINKY_TIP].x)

        if four_up and spread > 0.35 and not fullscreen_on and now - last_fullscreen_time > 1.2:
            pyautogui.press("f11")
            fullscreen_on = True
            last_fullscreen_time = now
            gesture_text = "FULL SCREEN"

       
        fist = not index and not middle and not ring and not pinky

        
        if four_up and fullscreen_on and now - last_fullscreen_time > 1.2:
            pyautogui.press("f11")
            fullscreen_on = False
            last_fullscreen_time = now
            gesture_text = "EXIT FULL SCREEN"

       
        only_thumb_index = index and not middle and not ring and not pinky

        index_x = lm.landmark[INDEX_TIP].x

        if only_thumb_index:
            if prev_index_x is None:
                prev_index_x = index_x

            delta_x = index_x - prev_index_x

            if not tab_locked:
                if delta_x > 0.07:
                    pyautogui.hotkey("ctrl", "tab")
                    gesture_text = "NEXT TAB"
                    tab_locked = True

                elif delta_x < -0.07:
                    pyautogui.hotkey("ctrl", "shift", "tab")
                    gesture_text = "PREVIOUS TAB"
                    tab_locked = True

            prev_index_x = index_x
        else:
            prev_index_x = None
            tab_locked = False


        pinky_thumb_dist = math.hypot(
            lm.landmark[PINKY_TIP].x - lm.landmark[THUMB_TIP].x,
            lm.landmark[PINKY_TIP].y - lm.landmark[THUMB_TIP].y
        )

        if pinky_thumb_dist < 0.035 and now - last_win_ctrl_time > win_ctrl_delay:
            pyautogui.hotkey("win", "h")
            gesture_text = "WIN + h"
            last_win_ctrl_time = now

  
    if gesture_text:
        cv2.putText(
            frame, gesture_text, (40, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3
        )

    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
