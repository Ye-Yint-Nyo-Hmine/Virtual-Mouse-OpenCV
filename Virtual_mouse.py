import cv2
import mediapipe as mp
import time
import math
import numpy as np
import pyautogui

cap = cv2.VideoCapture(0)

width_ = 960
height_ = 540
cap.set(3, width_)
cap.set(4, height_)

width = width_
height = height_

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pTime = 0
cTime = 0
looper = []


def landmarks(hand_landmarks):
    mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
    differ_x = abs(middle_finger[0] * 2 - middle_finger_l[0] * 2)  # less than 30 then, close together
    differ_y = abs(middle_finger[1] * 2 - middle_finger_l[1] * 2)  # less than 50

    if differ_x >= 30:
        if differ_y >= 250:
            pyautogui.click(index_finger[0] * 2, index_finger[1] * 2)

    if index_finger[0] * 2 <= 1918 and index_finger[0] * 2 >= 2:
        if index_finger[1] * 2 <= 1078 and index_finger[1] * 2 >= 2:
            pyautogui.moveTo(index_finger[0] * 2, index_finger[1] * 2)

    return looper


while True:
    success, img = cap.read()
    frame = cv2.flip(img, 1)

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        # specific points to be using for calculation
        thumb = int(results.multi_hand_landmarks[0].landmark[mpHands.HandLandmark.THUMB_TIP].x * width), \
                int(results.multi_hand_landmarks[0].landmark[mpHands.HandLandmark.THUMB_TIP].y * height)

        index_finger = int(results.multi_hand_landmarks[0].landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x * width), \
                       int(results.multi_hand_landmarks[0].landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y * height)

        middle_finger = int(results.multi_hand_landmarks[0].landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP].x * width), \
                        int(results.multi_hand_landmarks[0].landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP].y * height)

        middle_finger_l = int(
            results.multi_hand_landmarks[0].landmark[mpHands.HandLandmark.MIDDLE_FINGER_MCP].x * width), \
                          int(results.multi_hand_landmarks[0].landmark[
                                  mpHands.HandLandmark.MIDDLE_FINGER_MCP].y * height)

        pinky_mcp = int(results.multi_hand_landmarks[0].landmark[mpHands.HandLandmark.PINKY_MCP].x * width), \
                    int(results.multi_hand_landmarks[0].landmark[mpHands.HandLandmark.PINKY_MCP].y * height)

        wrist = int(results.multi_hand_landmarks[0].landmark[mpHands.HandLandmark.WRIST].x * width), \
                int(results.multi_hand_landmarks[0].landmark[mpHands.HandLandmark.WRIST].y * height)

        looper = [landmarks(hand_landmarks) for hand_landmarks in results.multi_hand_landmarks]

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(frame, 'FPS ' + str(int(fps)), (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 140, 0), 2,
                cv2.LINE_AA)

    cv2.imshow("Image", frame)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
