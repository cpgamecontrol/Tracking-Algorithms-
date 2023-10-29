import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
import sys

# opening camera
cap = cv2.VideoCapture(0)

# Initialize Pygame
pygame.init()

# Pygame constants
WINDOW_WIDTH = 1440
WINDOW_HEIGHT = 900
pygame_screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Shoulder Tracking")

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Game variables
cursor_radius = 20

# Initialize cursor positions for left and right shoulders
cursor_x_left = cursor_x_right = WINDOW_WIDTH // 2  # Start with the cursors at the window center
cursor_y = WINDOW_HEIGHT // 2  # Keep the cursors vertically centered


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
pTime = 0  # previous time

# Initialize 'text' with a default value
text = "NA"
with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.5) as holistic:

    prev_shoulder_x = None  # To track the previous shoulder x-coordinate

    while cap.isOpened():
        ret, frame = cap.read()

        # recolor and flip feed
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

        # processes and prints coords
        results = holistic.process(image)

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
            # Update cursor positions for left and right shoulders
            if left_shoulder_x:
                cursor_x_left = int(left_shoulder_x * WINDOW_WIDTH)
            if right_shoulder_x:
                cursor_x_right = int(right_shoulder_x * WINDOW_WIDTH)
                
                # Define a threshold for detecting significant movement
                movement_threshold = 0.008  # just an estimate really

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
                            text = new_direction
                        
                    else:
                        text = "Still"
                else:
                    text = "Still"

                    prev_shoulder_x = right_shoulder_x

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
                
        # Display the text message
        font = pygame.font.Font(None, 36)
        text_surface = font.render(text, True, (255, 255, 255))
        pygame_screen.blit(text_surface, (WINDOW_WIDTH // 2 - text_surface.get_width() // 2, 20))

        pygame.display.flip()  # Update the Pygame display  

        pygame_screen.fill((0, 0, 0))  # Clear the Pygame window

        # Draw the left cursor
        pygame.draw.circle(pygame_screen, WHITE, (cursor_x_left, cursor_y), cursor_radius)

        # Draw the right cursor
        pygame.draw.circle(pygame_screen, WHITE, (cursor_x_right, cursor_y), cursor_radius)

         # Draw a line connecting the left and right cursors
        pygame.draw.line(pygame_screen, WHITE, (cursor_x_left, cursor_y), (cursor_x_right, cursor_y), 2)


        # Check for quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()  # Release the camera
                pygame.quit()  # Quit Pygame
                sys.exit()  # Exit the Python program


# Release the camera
cap.release()
cv2.destroyAllWindows()


