#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:41:37 2023

@author: kjj
"""




import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui
import os

#opening camera

cap = cv2.VideoCapture(0)


#opening up game

os.system("open /Applications/Google\ Chrome.app")
pyautogui.moveTo(200,85,1)
pyautogui.click()
pyautogui.write('https://www.crazygames.com/game/rocket-pants-runner-3d', interval=0.01)
pyautogui.press('enter') 
pyautogui.moveTo(550,550,2)
pyautogui.click()
pyautogui.moveTo(1000,710,7)
pyautogui.click()
pyautogui.moveTo(700,300,1)
pyautogui.click()
pyautogui.moveTo(700,800,1)


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


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

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

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

            cam_matrix = np.array([ [focal_length, 0, img_h / 2],
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
            if y < -3:
                text = "Looking Left"
                # pyautogui.press('left') 
                pyautogui.keyDown('left')
                pyautogui.keyUp('left')
                #pyautogui.moveRel(-20,0)
            elif y > 3:
                text = "Looking Right"
                # pyautogui.press('right')
                pyautogui.keyDown('right')
                pyautogui.keyUp('right')
                #pyautogui.moveRel(20,0)
            elif x < -2:
                text = "Looking Down"
                pyautogui.keyDown('down')
                pyautogui.keyUp('down')
                #pyautogui.moveRel(0,20)
            elif x > 4:
                text = "Looking Up"
                pyautogui.keyDown('up')
                pyautogui.keyUp('up')
                #pyautogui.moveRel(0,-20)
            else:
                text = "Forward"


cap.release()
cv2.destroyWindow('Head Pose Estimation')

