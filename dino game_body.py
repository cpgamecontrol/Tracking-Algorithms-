import cv2
import mediapipe as mp
import time
import pyautogui
import os

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
pTime = 0  # previous time


cap = cv2.VideoCapture(0)

#opening up game

os.system("open /Applications/Google\ Chrome.app")
pyautogui.moveTo(200,85,1)
pyautogui.click()
pyautogui.write('https://poki.com/en/g/dinosaur-game', interval=0.01)
pyautogui.press('enter') 
pyautogui.moveTo(1000,650,2)
pyautogui.click()

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.5) as holistic:

    prev_shoulder_x = None  # To track the previous shoulder x-coordinate

    while cap.isOpened():
        ret, frame = cap.read()

        # recolor and flip feed
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

        # processes and prints coords
        results = holistic.process(image)
        print(results.pose_landmarks)

        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                 )

        # check for shoulder landmarks
        if results.pose_landmarks:
            left_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]

            if left_shoulder and right_shoulder:
                # Get x-coordinates of left and right shoulders
                left_shoulder_x = left_shoulder.x
                right_shoulder_x = right_shoulder.x
                

                # Define a threshold for detecting significant movement
                movement_threshold = 0.01  # just an estimate really

                # Initialize the current swaying direction
                current_direction = "Still"

                # swaying direction based on shoulder movement - continues to move right until you go back to the middele
                if prev_shoulder_x is not None: 
                    shoulder_x_diff = right_shoulder_x - prev_shoulder_x #rolling difference of movement 

                    # Check if the absolute difference is greater than the movement threshold
                    if abs(shoulder_x_diff) > movement_threshold:
                        #and if difference is a positive - move right
                        if shoulder_x_diff > 0:
                            new_direction = "Right"
                       #and if difference is a negative - move left
                        else:
                            new_direction = "Left"

                        # Check if there's a change in direction
                        if new_direction != current_direction:
                            current_direction = new_direction
                            
                            # Send key press based on the new direction
                            if current_direction == "Right":
                                pyautogui.press('right')
                            elif current_direction == "Left":
                                pyautogui.press('left')
                    else:
                        current_direction = "Still"
                else:
                    current_direction = "Still"

                    prev_shoulder_x = right_shoulder_x

                # Display the swaying direction on the frame
                cv2.putText(image, current_direction, (70, 100), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255), 2)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        #show body tracking
        # cv2.imshow('Body Tracking', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
  
cap.release()
cv2.destroyAllWindows()
