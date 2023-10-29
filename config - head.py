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
pygame.display.set_caption("Head Tracking")

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Game variables
cursor_radius = 20

# Cursor starts in the middle
cursor_x, cursor_y = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Initialize 'text' with a default value
text = "No face detected"

while cap.isOpened():
    success, image = cap.read()
    start = time.time()

    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False

    # Get the result
    results = face_mesh.process(image)

    # To improve performance
    image.flags.writeable = True

    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, _ = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        cursor_x = int(lm.x * img_w)
                        cursor_y = int(lm.y * img_h)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])

            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]])

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # See where the user's head tilting
            if y < -2:
                text = "Looking Left"
            elif y > 3:
                text = "Looking Right"
            elif x < -1:
                text = "Looking Down"
            elif x > 4:
                text = "Looking Up"
            else:
                text = "Forward"
                
    # Display the text message
    font = pygame.font.Font(None, 36)
    text_surface = font.render(text, True, (255, 255, 255))
    pygame_screen.blit(text_surface, (WINDOW_WIDTH // 2 - text_surface.get_width() // 2, 20))

    pygame.display.flip()  # Update the Pygame display  

    pygame_screen.fill((0, 0, 0))  # Clear the Pygame window

    # Draw the cursor
    pygame.draw.circle(pygame_screen, WHITE, (cursor_x, cursor_y), cursor_radius)
    
    # Position and sie for the X symbol
    x_pos = WINDOW_WIDTH // 2
    y_pos = WINDOW_HEIGHT // 2
    x_size = 10
        
    #Draw the cross
    pygame.draw.line(pygame_screen, RED, (x_pos - x_size, y_pos - x_size), (x_pos + x_size, y_pos + x_size), 5)
    pygame.draw.line(pygame_screen, RED, (x_pos + x_size, y_pos - x_size), (x_pos - x_size, y_pos + x_size), 5)
    

    # Check for quit event
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()  # Release the camera
            pygame.quit()  # Quit Pygame
            sys.exit()  # Exit the Python program


# Release the camera
cap.release()
cv2.destroyAllWindows()


