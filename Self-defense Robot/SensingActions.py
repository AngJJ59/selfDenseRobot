import numpy as np
import gym
import matplotlib as mpl
import mediapipe as mp
import cv2 
import sim


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

sim.simxFinish(-1)

clientID=sim.simxStart("127.0.0.1",19997,True,True,5000,5) 




def MotorOn():
    err_code,l_motor_handle = sim.simxGetObjectHandle(clientID,"bubbleRob_leftMotor", sim.simx_opmode_blocking)
    err_code,r_motor_handle = sim.simxGetObjectHandle(clientID,"bubbleRob_rightMotor", sim.simx_opmode_blocking)
    err_code = sim.simxSetJointTargetVelocity(clientID, l_motor_handle, 1.0, sim.simx_opmode_streaming)
    err_code = sim.simxSetJointTargetVelocity(clientID, r_motor_handle, 1.0, sim.simx_opmode_streaming)
    print ("run motor")



def checkTheLandmarks():
    for i in mp_pose.PoseLandmark:
        print(i)

def calculateAngle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle


# For static images:
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
    # Convert the BGR image to RGB before processing.
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
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
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
    
    
    
    #Extract the landmarks 
    try:
        landmarks = results.pose_landmarks.landmark
        
        
        
        
        #left arm coordinates
        #shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].X, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        
        
        
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        
        
        #calculate the angle
        left_angle = calculateAngle(left_shoulder, left_elbow, left_wrist)
        right_angle = calculateAngle(right_shoulder, right_elbow, right_wrist)
        
        #visualize the angle
        cv2.putText(image, str(left_angle), tuple(np.multiply(left_elbow, [640, 480]).astype(int)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
        cv2.putText(image, str(right_angle), tuple(np.multiply(right_shoulder, [640, 480]).astype(int)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
        #cv2.putText(img, text, org, fontFace, fontScale, color)
        
        
        # gonna punch me logic
        if left_angle > 160 and right_angle > 160:
            stage = "down"
        if left_angle < 30 or right_angle < 30 and stage == "down":
            stage="up"
            Offensive_move = True
            if Offensive_move == True:
                
                
                
                
                print(Offensive_move)
            
        
        
        
        
        #print(landmarks)
        
        
        
        
        
        
    except:
        pass
    
    cv2.rectangle(image, (0,0), (255, 73), (255, 0, 0))
    cv2.putText(image, "Offensive Action", (15,12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))
    cv2.putText(image, stage, (10, 60), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,0))
    #cv2.putText(image, str(Offensive_move), (10,60), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))
    
    
    
    
    
    
    
    
    
    #Show statusbox 
    # cv2.rectangle(image, (0,0), (255, 73), (255, 0, 0))
    # cv2.putText(image, statusoutput, (15,12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))
    
    
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    cv2.imshow('MediaPipe Pose', image)
    
    #checkTheLandmarks()
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()


        








