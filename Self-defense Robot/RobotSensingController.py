#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 23:37:21 2022

@author: benderang
"""

import numpy as np
import gym
import matplotlib as mpl
import mediapipe as mp
import cv2 
import sim
import sys

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

sim.simxFinish(-1)

clientID=sim.simxStart("127.0.0.1",19999,True,True,5000,5) 


# def getHeadFollowtheTArget():
#     print('test')


    
def blockLeftHandAttack(targetjointangle1,targetjointangle4):
    err_code, left_joint_handle1 = sim.simxGetObjectHandle(clientID, 'Baxter_leftArm_joint1', sim.simx_opmode_blocking)
    err_code = sim.simxSetJointTargetPosition(clientID, left_joint_handle1, targetjointangle1, sim.simx_opmode_streaming)
    err_code, left_joint_handle4 = sim.simxGetObjectHandle(clientID, 'Baxter_leftArm_joint4', sim.simx_opmode_blocking)
    err_code = sim.simxSetJointTargetPosition(clientID, left_joint_handle4, targetjointangle4, sim.simx_opmode_blocking)



def blockRightHandAttack(righttargetjointangle1, righttargetjointangle4):
    err_code, right_joint_handle2 = sim.simxGetObjectHandle(clientID, 'Baxter_rightArm_joint1', sim.simx_opmode_blocking)
    err_code = sim.simxSetJointTargetPosition(clientID, right_joint_handle2, righttargetjointangle1, sim.simx_opmode_streaming)
    err_code, right_joint_handle4 = sim.simxGetObjectHandle(clientID, 'Baxter_rightArm_joint4', sim.simx_opmode_blocking)
    err_code = sim.simxSetJointTargetPosition(clientID, right_joint_handle4, righttargetjointangle4, sim.simx_opmode_blocking)
        

def forearmToBlockLeftChopping(targetjointangle2,targetjointangle6):
    err_code, left_joint_handle1 = sim.simxGetObjectHandle(clientID, 'Baxter_leftArm_joint2', sim.simx_opmode_blocking)
    err_code = sim.simxSetJointTargetPosition(clientID, left_joint_handle1, targetjointangle2, sim.simx_opmode_streaming)
    err_code, left_joint_handle4 = sim.simxGetObjectHandle(clientID, 'Baxter_leftArm_joint6', sim.simx_opmode_blocking)
    err_code = sim.simxSetJointTargetPosition(clientID, left_joint_handle4, targetjointangle6, sim.simx_opmode_blocking)

     
def forearmToBlockRightChopping(righttargetjointangle2, righttargetjointangle6):
    err_code, left_joint_handle1 = sim.simxGetObjectHandle(clientID, 'Baxter_rightArm_joint2', sim.simx_opmode_blocking)
    err_code = sim.simxSetJointTargetPosition(clientID, left_joint_handle1, righttargetjointangle2, sim.simx_opmode_streaming)
    err_code, left_joint_handle4 = sim.simxGetObjectHandle(clientID, 'Baxter_rightArm_joint6', sim.simx_opmode_blocking)
    err_code = sim.simxSetJointTargetPosition(clientID, left_joint_handle4, righttargetjointangle6, sim.simx_opmode_blocking)

    


def sensingOffensiveMoves():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    
    
    #function to calculate the angle
    def calculateUppercutAngle(a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
        
        return angle
    
    #this is to calculate the kick angle
    def calculateKickAngle(hip, knee, ankle):
        hip = np.array(hip)
        knee = np.array(knee)
        ankle = np.array(ankle)
        
        radians = np.arctan2(ankle[1]-knee[1], ankle[0]-knee[0]) - np.arctan2(hip[1]-knee[1], hip[0]-knee[0])
        kickangle = np.abs(radians*180.0/np.pi)
        
        if kickangle > 180.0:
            kickangle = 360-kickangle
            
        return kickangle
    
    
    def calculateChopingAngle(Selbow, Sshoulder, Ship):
        Selbow = np.array(Selbow)
        Sshoulder = np.array(Sshoulder)
        Ship = np.array(Ship)
        
        radians = np.arctan2(Ship[1]-Sshoulder[1], Ship[0]-Sshoulder[0]) - np.arctan2(Selbow[1]-Sshoulder[1], Selbow[0]-Sshoulder[0])
        shoulderangle = np.abs(radians*180.0/np.pi)
        
        if shoulderangle > 180.0:
           shoulderangle = 360-shoulderangle
        
        return shoulderangle
        
    
        
        
    
    
    
    
    # this is for static images
    IMAGE_FILES = []
    BG_COLOR = (192, 192, 192) # gray
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5) as pose:
      for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)
        image_height, image_width, _ = image.shape
        # it will change the BGR to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
          continue
        print(
            f'Nose coordinates: ('
            f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
            f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
        )

        annotated_image = image.copy()
        # Draw segmentation on the image.
       
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        annotated_image = np.where(condition, annotated_image, bg_image)
        # Draw pose landmarks on the image.
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
        # Plot pose world landmarks.
        mp_drawing.plot_landmarks(
            results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

    # For webcam input:
    cap = cv2.VideoCapture(0)
    
    #gonna punch?
    stage = None
    Offensive_move = False
    #baxterrobotarmangle = 0
    
    with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
        
            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            try:
                
                landmarks = results.pose_landmarks.landmark
                
                
                #left arm coordinates
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                #right arm coordinates
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                #calculate the left and right arms to check whether an uppercut action is executing or not
                left_uppercut_angle = calculateUppercutAngle(left_shoulder, left_elbow, left_wrist)
                right_uppercut_angle = calculateUppercutAngle(right_shoulder, right_elbow, right_wrist)
                
                #visualize the arms angle
                cv2.putText(image, str(left_uppercut_angle), tuple(np.multiply(left_elbow, [640, 480]).astype(int)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
                cv2.putText(image, str(right_uppercut_angle), tuple(np.multiply(right_shoulder, [640, 480]).astype(int)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
                
                
                #left kick coordinates
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
                #calculate left kick angle
                left_kick_angle = calculateKickAngle(left_hip, left_knee, left_ankle)
                cv2.putText(image, str(left_kick_angle), tuple(np.multiply(left_hip, [640, 480]).astype(int)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
                
                #right kick coordinates
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                
                #calculate right kick angle
                right_kick_angle = calculateKickAngle(right_hip, right_knee, right_ankle)
                cv2.putText(image, str(right_kick_angle), tuple(np.multiply(right_hip, [640, 480]).astype(int)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
                
                #define the left shoulder to determine whether the offensive action is choping the head or not
                choping_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                choping_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                choping_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                
                choping_angle = calculateChopingAngle(choping_elbow, choping_shoulder, choping_hip)
                cv2.putText(image, str(choping_angle), tuple(np.multiply(choping_shoulder, [640, 480]).astype(int)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
                
                #define the right shoulder to determine whether the offensive action 
                
                choping_right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                choping_right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                choping_right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                
                right_choping_angle = calculateChopingAngle(choping_right_elbow, choping_right_shoulder, choping_right_hip)
                cv2.putText(image, str(right_choping_angle), tuple(np.multiply(choping_right_shoulder, [640, 480]).astype(int)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
                
                
                # gonna punch me logic
                #print (left_angle)
                #print(right_angle)
                if left_uppercut_angle > 160 and right_uppercut_angle > 160 and left_kick_angle > 160 and right_kick_angle > 160 and choping_angle <20:
                    Offensive_move = False
                    print ("down")
                    print('left arm angle=' ,left_uppercut_angle)
                    print('right arm angle=', right_uppercut_angle)
                    #print('chopping angle is:', choping_angle)
                    stage = "down"
                    blockLeftHandAttack(0, 0)
                    blockRightHandAttack(0, 0)
                    forearmToBlockLeftChopping(0, 0)
                    forearmToBlockRightChopping(0, 0)
                    
                    # callTheBaxterController(0)
                    # callTheBaxterRightArm(180)
                    #stopMotor()
                
                if choping_angle > 30 or right_choping_angle > 30 and stage == 'down':
                    stage="up"
                    if choping_angle > 30:
                        blockLeftHandAttack(-180, -90)
                        forearmToBlockLeftChopping(-120, -90)
                    if right_choping_angle > 30:
                        blockRightHandAttack(180, -90)
                        forearmToBlockRightChopping(-120, -90)
                
                if left_uppercut_angle < 30 or right_uppercut_angle < 30 or left_kick_angle < 30 or right_kick_angle < 30 and stage == "down":
                    stage="up"
                    print ("up")
                    print('left arm angle=' ,left_uppercut_angle)
                    print('right arm angle=', right_uppercut_angle)
                    print('chopping angle is:', choping_angle)
                    if left_uppercut_angle < 30:
                        # blockLeftHandAttack(-270, 90)
                        blockRightHandAttack(270, 90)
                    if right_uppercut_angle < 30:
                        blockLeftHandAttack(-270, 90)
                        # blockRightHandAttack(270, 90)
                   
                    if left_kick_angle < 30 or right_kick_angle<30:
                        
                         blockLeftHandAttack(-270, 90)
                         blockRightHandAttack(270, 90)
                         forearmToBlockLeftChopping(0, 0)
                         forearmToBlockRightChopping(0, 0)
                    
                    Offensive_move = True
                
                # if left_kick_angle > 160:
                #     Offensive_move = False
                    
                
                
     
                
                
                
                
            except:
                pass
            
            cv2.rectangle(image, (0,0), (255, 73), (255, 0, 0))
            cv2.putText(image, "Offensive Action", (15,12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))
            cv2.putText(image, stage, (10, 60), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,0))
            
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
            cv2.imshow('Baxter robot offensive action detection', image)
            
            if cv2.waitKey(5) & 0xFF == 27:
                    break
            
    cap.release()                
                
                
            



if clientID != -1:
    print("Successfully connected")
    #sensingOffensiveMoves()
    
else:
    print("Not connected")
    sys.exit("Could not connect")
    
    
sensingOffensiveMoves()

    
    
    
    
    
    
    
    
    
    
    
