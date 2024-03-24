import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
# Variables for data output (placeholders)
direction_num = 0
direction_facing = "Unknown"
last_update_time = 0  # Variable to track the time of the last update
max_shoulder_size = 0
tickCheck = 0
user_height = 0 #cm
user_depth = 150 #cm
wait_for_update = 0
once = True
left_shoulder_z = 0
left_elbow_z = 0
user_weight = 58.967 #kg
forearm = (user_height*0.01) * 0.216
upperarm = (user_height*0.01) * 0.173
cfg = forearm * 0.432
b = forearm * 0.11
weightForearm = user_weight * 0.023
weightAdded = 0
developer_mode = False  # Developer mode state
force_data = 0
angle_data = 0
leftArmAngle = 0
left_arm_bicep_force = 0
time_to_get_position_var = 0
time_simulation_var = 0
start_time = 0
twoStepDone = False
twoStepCountDown = 10
isGraphOn = False








def calculate_distance(landmark1, landmark2):
    """
    Calculate the Euclidean distance between two landmarks.
    """
    return np.sqrt((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2)

def get_distance_right_eye_outer_to_ear():
    """
    Get the distance between the right eye outer and the right ear using MediaPipe Pose.
    """
    distance = None
    
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """

    if results.pose_landmarks:
        right_eye_outer = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE_OUTER]
        right_ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]

        # Calculate the distance
        distance = calculate_distance(right_eye_outer, right_ear)

    return distance


def get_distance_left_eye_outer_to_ear():
    """
    Get the distance between the left eye outer and the left ear using MediaPipe Pose.
    """
    distance = None
    
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """

    if results.pose_landmarks:
        left_eye_outer = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE_OUTER]
        left_ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]

        # Calculate the distance
        distance = calculate_distance(left_eye_outer, left_ear)

    return distance


def get_distance_right_hip_to_right_shoulder():
    """
    Get the distance between the right hip and the right shoulder using MediaPipe Pose.
    """
    distance = None
    
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """

    if results.pose_landmarks:
        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Calculate the distance
        distance = calculate_distance(right_hip, right_shoulder)

    return distance

def get_distance_left_hip_to_left_shoulder():
    """
    Get the distance between the left hip and the left shoulder using MediaPipe Pose.
    """
    distance = None
    
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """

    if results.pose_landmarks:
        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]

        # Calculate the distance
        distance = left_hip.y - left_shoulder.y

    return distance


def get_distance_right_shoulder_to_left_shoulder():
    """
    Get the distance between the right shoulder and the left shoulder using MediaPipe Pose.
    """
    distance = None
    
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """

    if results.pose_landmarks:
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]

        # Calculate the distance
        distance = calculate_distance(right_shoulder, left_shoulder)

    return distance


def get_distance_right_hip_to_left_hip():
    """
    Get the distance between the right hip and the left hip using MediaPipe Pose.
    """
    distance = None
    
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """

    if results.pose_landmarks:
        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]

        # Calculate the distance
        distance = calculate_distance(right_hip, left_hip)

    return distance


def get_distance_right_shoulder_to_right_elbow():
    """
    Get the distance between the right shoulder and the right elbow using MediaPipe Pose.
    """
    distance = None
    
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """

    if results.pose_landmarks:
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]

        # Calculate the distance
        distance = calculate_distance(right_shoulder, right_elbow)

    return distance


def get_distance_left_shoulder_to_left_elbow():
    """
    Get the distance between the left shoulder and the left elbow using MediaPipe Pose.
    """
    distance = None
    
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """

    if results.pose_landmarks:
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]

        # Calculate the distance
        distance = calculate_distance(left_shoulder, left_elbow)

    return distance


def get_distance_right_elbow_to_right_wrist():
    """
    Get the distance between the right elbow and the right wrist using MediaPipe Pose.
    """
    distance = None
    
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """
    
    if results.pose_landmarks:
        right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

        # Calculate the distance
        distance = calculate_distance(right_elbow, right_wrist)

    return distance


def get_distance_left_elbow_to_left_wrist():
    """
    Get the distance between the left elbow and the left wrist using MediaPipe Pose.
    """
    distance = None
    
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """

    if results.pose_landmarks:
        left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]

        # Calculate the distance
        distance = calculate_distance(left_elbow, left_wrist)

    return distance


def get_head_width():
    """
    Get the distance between the right ear and the left ear using MediaPipe Pose.
    """
    distance = None
    
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """

    if results.pose_landmarks:
        right_ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]
        left_ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]

        # Calculate the distance
        distance = calculate_distance(right_ear, left_ear)

    return distance


def get_height_diff_right_shoulder_to_right_hip():
    """
    Get the distance between the right hip and the left hip using MediaPipe Pose.
    """
    distance = None
    
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """

    if results.pose_landmarks:
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

        # Calculate the distance
        distance = right_hip.y - right_shoulder.y

    return distance


def get_distance_fingertip_to_fingertip():
    """
    Get the distance between the left index finger and the right index finger using MediaPipe Pose.
    """
    distance = None
    
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """

    if results.pose_landmarks:
        left_index_finger = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX]
        right_index_finger = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX]

        # Calculate the distance
        distance = calculate_distance(left_index_finger, right_index_finger)

    return distance

def get_distance_left_fingertip_to_elbow():
    """
    Get the distance between the left index finger and the right index finger using MediaPipe Pose.
    """
    distance = None
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """

    if results.pose_landmarks:
        left_index_finger = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX]
        left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]

        # Calculate the distance
        distance = calculate_distance(left_index_finger, left_elbow)

    return distance

def get_distance_right_fingertip_to_elbow():
    """
    Get the distance between the left index finger and the right index finger using MediaPipe Pose.
    """
    distance = None
    
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """


    if results.pose_landmarks:
        right_index_finger = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX]
        right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]

        # Calculate the distance
        distance = calculate_distance(right_index_finger, right_elbow)

    return distance


def get_left_hip_x():
    x = 0
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """

    if results.pose_landmarks:
        left_hip_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * m_to_mpu_ratio
        #left_hip_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * m_to_mpu_ratio
        #left_hip_z = user_depth
        
        # Calculate the distance
        x = left_hip_x

    return x

def get_left_hip_y():
    y = 0
    
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """

    if results.pose_landmarks:
        #left_hip_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * m_to_mpu_ratio
        left_hip_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * m_to_mpu_ratio
        #left_hip_z = user_depth
        
        # Calculate the distance
        y = left_hip_y

    return y


def get_left_hip_z():
    z = 0
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """

    if results.pose_landmarks:
        #left_hip_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * m_to_mpu_ratio
        #left_hip_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * m_to_mpu_ratio
        left_hip_z = user_depth
        
        # Calculate the distance
        z = left_hip_z

    return z

def get_left_hip_x_y_z():
    global left_hip_z
    xyz = [0,0,0]
    
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """

    if results.pose_landmarks:
        left_hip_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * m_to_mpu_ratio
        left_hip_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * m_to_mpu_ratio
        left_hip_z = user_depth
        
        # Calculate the distance
        xyz = [left_hip_x,left_hip_y,left_hip_z]

    return xyz


def get_left_shoulder_x_y_z():
    global left_hip_z, left_shoulder_z
    xyz = [0,0,0]
    
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """
    
    #print("inisde left shoulder")


    if results.pose_landmarks:
        if developer_mode:
            print("shoulder")

        left_shoulder_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * m_to_mpu_ratio
        left_shoulder_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * m_to_mpu_ratio
        left_shoulder_z = calculate_z_angle(left_hip_z, init_left_distance_hip_shoulder, body_pitch)

        # Calculate the distance
        xyz = [left_shoulder_x, left_shoulder_y, left_shoulder_z]
        if developer_mode:
            print(xyz)
    
    return xyz

def get_left_elbow_x_y_z():
    global left_shoulder_z, left_elbow_z
    xyz = [0,0,0]
    
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """
    

    #print("inisde left shoulder")


    if results.pose_landmarks:
        if developer_mode:
            print("elbow")

        left_elbow_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * m_to_mpu_ratio
        left_elbow_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * m_to_mpu_ratio
        left_elbow_z = calculate_z(left_shoulder_z, init_left_shoulder_to_left_elbow, get_distance_left_shoulder_to_left_elbow(), 0, body_pitch, hipShoElb)
        
        # Calculate the distance
        xyz = [left_elbow_x, left_elbow_y, left_elbow_z]
        if developer_mode:
            print(xyz)

    return xyz

def get_left_wrist_x_y_z():
    global left_elbow_z, left_wrist_z
    xyz = [0,0,0]
    
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """
    #print("inisde left shoulder")


    if results.pose_landmarks:
        if developer_mode:
            print("wrist")

        left_wrist_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * m_to_mpu_ratio
        left_wrist_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * m_to_mpu_ratio
        left_wrist_z = calculate_z(left_elbow_z, init_left_elbow_to_left_wrist, get_distance_left_elbow_to_left_wrist(), 0, body_pitch, hipShoElb)
        
        # Calculate the distance
        xyz = [left_wrist_x, left_wrist_y, left_wrist_z]
        if developer_mode:
            print(xyz)

    return xyz


# Test the function with an image
# image = cv2.imread("path_to_your_image.jpg")
# distance = get_distance_right_eye_outer_to_ear(image)
# print("Distance:", distance)


def calculate_z(z_init, max_length, actual_length, angle, pitch, hipShoElb):
    z = 0
    max_len = max_length*m_to_mpu_ratio
    max_len = max_len + (max_len*0.9*abs(pitch/90)) + (max_len*0.47*abs((90-hipShoElb)/90))
    act_len = actual_length*m_to_mpu_ratio
    if angle > 0:
        if act_len >= max_len: 
            act_len = max_len
        if developer_mode:    
            print("z_init: " + str(z_init) + ", max_length: " + str(max_len) + ", actual_length: " + str(act_len) + ", angle: " + str(angle) + ", max mpu: " + str(init_user_max_mpu))
        z = z_init + np.sqrt(abs((max_len)**2 - (act_len)**2))

        return z
    else:
        if act_len >= max_len: 
            act_len = max_len
        z = z_init + np.sqrt((max_len)**2 - (act_len)**2)
        if developer_mode:
            print("z_init: " + str(z_init) + ", max_length: " + str(max_len) + ", actual_length: " + str(act_len) + ", angle: " + str(angle) + ", max mpu: " + str(init_user_max_mpu) + ", z = zinit + " + str(np.sqrt((max_len)**2 - (act_len)**2)) + ", z = " + str(z))
        return z
    
def calculate_z_angle(z_init, max_length, angle):
    #print("calc ran")

    z = 0
    if angle > 0: #Backward
        if developer_mode:
            print("z_init: " + str(z_init) + ", max_length: " + str(max_length*m_to_mpu_ratio) + ", angle: " + str(angle/90) + ", z + : " + str(((max_length*m_to_mpu_ratio)*(angle/90))))
        z = z_init + ((max_length*m_to_mpu_ratio)*(angle/90))

        return z
    else: # Forward
        if developer_mode:
            print("z_init: " + str(z_init) + ", max_length: " + str(max_length*m_to_mpu_ratio) + ", angle: " + str(angle/90) + ", z + : " + str(((max_length*m_to_mpu_ratio)*(angle/90))))
        z = z_init + ((max_length*m_to_mpu_ratio)*(angle/90))        
        
        return z


def calculate_direction(distance_right, distance_left):
    """
    Calculate the direction the person is facing based on the distances.
    """
    if distance_right is not None and distance_left is not None and distance_right != 0 and distance_left != 0:
        direction_num = distance_right / distance_left
        direction_facing = "Right" if direction_num <= 1 else "Left"
        return direction_num, direction_facing
    return None, "Unknown"


def calculate_body_yaw(distance_shoulder, distance_hip_shoulder, direction_facing, init_val):
    """
    Calculate the body's yaw based on the distance between the shoulders over the distance between hips and shoulders and the 
    direction facing(left or right) to notate whether the user is turning to the left or right.
    """
    if distance_shoulder is not None and distance_hip_shoulder is not None and distance_hip_shoulder != 0:
        if direction_facing == "Right":
            return round(90-(((distance_shoulder / distance_hip_shoulder)/init_val)*90),4) #init_val = 0.55
        else:
            return round((((distance_shoulder / distance_hip_shoulder)/init_val)*90)-90, 4) #init_val = 0.55
    return 0


def calculate_body_pitch(head_width, height_diff_hip_shoulder, init_val, eye_ear_angle, init_eye_ear_angle):
    """
    Calculate the body's pitch based on height difference between the right should and the right hip over the width of the head 
    and the direction facing(up or down) to notate whether the user is leaning forward or backward.
    """
    if head_width is not None and height_diff_hip_shoulder is not None and height_diff_hip_shoulder != 0:
            if eye_ear_angle <= init_eye_ear_angle:
                if developer_mode:
                    print("up")
                return round((180-(np.arcsin(abs(height_diff_hip_shoulder / init_val))*90)), 4) #init_val = ? add arcsin - took out *2 after *90)
            else:
                if developer_mode:
                    print("down")
                return round(((np.arcsin(abs(height_diff_hip_shoulder / init_val))*90))-180, 4) #init_val = ? - took out *2 after *90)
    return 0


def calculate_body_roll():
    """
    Calculate the angle between the line connecting the shoulders and the horizontal line.
    """

    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    """


    if results.pose_landmarks:
        # Get landmarks for shoulders
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Calculate the slope and the angle
        delta_x = right_shoulder.x - left_shoulder.x
        delta_y = right_shoulder.y - left_shoulder.y  # Y value decreases upwards in image coordinates
        
        angle_radians = math.atan2(delta_y, delta_x)  # Angle with respect to the horizontal line
        angle_degrees = math.degrees(angle_radians)
        
        # Adjusting the angle to horizontal, 0 degrees means the shoulders are perfectly horizontal
        
        if (angle_degrees > 0):
            shoulder_angle = -(angle_degrees)+180
        else:
            shoulder_angle = -((angle_degrees)+180)
        return shoulder_angle
    return None


def calculate_angle(a,b,c):
    """
    This method is a intermediary function for the rest of the specific angle calculations
    """
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle


def calculate_left_arm_angle():
    """
    Basic calculation to find the angle between the left shoulder, left elbow, and left wrist.
    """

    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    landmarks = results.pose_landmarks.landmark
    """
    
    if results.pose_landmarks:
        # Get landmarks for shoulders
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
 
        angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

        return angle
    return None


def calculate_nose_eyeInR_earR():
    """
    Calculates the angle between the nose, right inner eye, and right ear.
    *Used to calculate whether the user is facing up or down 
    """

    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    landmarks = results.pose_landmarks.landmark
    """


    if results.pose_landmarks:
        # Get landmarks for shoulders
        nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y]
        right_eye_inner = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].y]
        right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
        angle = calculate_angle(nose, right_eye_inner, right_ear)

        return angle
    return None


def calculate_left_hip_shoulder_elbow_angle():
    """
    Calculates the angle between the nose, right inner eye, and right ear.
    *Used to calculate whether the user is facing up or down 
    """

    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    landmarks = results.pose_landmarks.landmark
    """

    if results.pose_landmarks:
        # Get landmarks for shoulders
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]

        angle = calculate_angle(left_hip, left_shoulder, left_elbow)
        return angle
    return None


def dot_prod_angle(matrixA, matrixB, matrixC):
    aTimesB = 0
    #vectorA = [matrixB[0] - matrixA[0], matrixB[1] - matrixA[1], matrixB[2] - matrixA[2]]
    #vectorB = [matrixC[0] - matrixB[0], matrixC[1] - matrixB[1], matrixC[2] - matrixB[2]]
    aTimesB = (((matrixB[0]-matrixA[0])*(matrixC[0]-matrixB[0]))+((matrixB[1]-matrixA[1])*(matrixC[1]-matrixB[1]))+((matrixB[2]-matrixA[2])*(matrixC[2]-matrixB[2])))
    aMag = np.sqrt(((matrixB[0]-matrixA[0])**2) + ((matrixB[1]-matrixA[1])**2) + ((matrixB[2]-matrixA[2])**2))
    bMag = np.sqrt(((matrixC[0]-matrixB[0])**2) + ((matrixC[1]-matrixB[1])**2) + ((matrixC[2]-matrixB[2])**2))
    theta = np.arccos(aTimesB/(aMag*bMag))
    
    if developer_mode:
        print(str(theta * (180/np.pi)))
    return theta * (180/np.pi)


def calculate_arm_force(thetaUpper, thetaArm, weightAdded):

    thetaB = 180 - ((b - upperarm * np.cos(thetaUpper))/ (np.sqrt(b**2 + upperarm**2 - 2 * b * upperarm * np.cos(thetaUpper))) )
    leverArmFA = cfg * np.sin(thetaUpper + thetaArm - 90)
    leverArmAdd = forearm * np.sin(thetaUpper + thetaArm - 90)
    leverArmBic = b * np.sin(thetaB)
    if developer_mode:
        print("ThetaB: " + str(thetaB) + ", leverArmFA: " + str(leverArmFA) + "leverArmAdd: " + str(leverArmAdd) + "leverArmBic: " + str(leverArmBic))
    force = abs((weightForearm*9.81 * leverArmFA + weightAdded*9.81 * leverArmAdd) / leverArmBic)
    if developer_mode:
        print("Bicep Force: " + str(force))
    return force

def init_data_update(image):
    """
    This method is called once before the program begins updating calculations so that initial values can be found for the user's specifc body ratios
    """
    global init_distance_shoulder, init_distance_hip_shoulder, init_left_distance_hip_shoulder, init_height_diff_right_shoulder_to_right_hip, init_head_width, init_nose_eye_ear_angle, init_right_shoulder_to_right_elbow, init_right_elbow_to_right_wrist, init_left_shoulder_to_left_elbow, init_left_elbow_to_left_wrist, init_user_max_mpu, m_to_mpu_ratio, image_rgb, results, landmarks
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    landmarks = results.pose_landmarks.landmark


    init_distance_shoulder = get_distance_right_shoulder_to_left_shoulder()

    init_distance_hip_shoulder = get_distance_right_hip_to_right_shoulder()
    init_left_distance_hip_shoulder = get_distance_left_hip_to_left_shoulder()
    init_height_diff_right_shoulder_to_right_hip = get_height_diff_right_shoulder_to_right_hip()
    init_head_width = get_head_width()
    init_nose_eye_ear_angle = calculate_nose_eyeInR_earR()
    init_right_shoulder_to_right_elbow = get_distance_right_shoulder_to_right_elbow()
    init_right_elbow_to_right_wrist = get_distance_right_elbow_to_right_wrist()
    init_left_shoulder_to_left_elbow = get_distance_left_shoulder_to_left_elbow()
    init_left_elbow_to_left_wrist = get_distance_left_elbow_to_left_wrist()
    #init_user_max_mpu = get_distance_left_fingertip_to_elbow(image) + get_distance_left_shoulder_to_left_elbow(image) + get_distance_right_shoulder_to_left_shoulder(image) + get_distance_right_shoulder_to_right_elbow(image) + get_distance_right_fingertip_to_elbow(image)
    init_user_max_mpu = get_distance_fingertip_to_fingertip()
    m_to_mpu_ratio = user_height/init_user_max_mpu #cm per mpu

def init2_data_update(image):
    """
    This method is called once before the program begins updating calculations so that initial values can be found for the user's specifc body ratios
    """
    global init_distance_shoulder2, init_distance_hip_shoulder2, init_left_distance_hip_shoulder2, init_height_diff_right_shoulder_to_right_hip2, init_head_width2, init_nose_eye_ear_angle2, init_right_shoulder_to_right_elbow2, init_right_elbow_to_right_wrist2, init_left_shoulder_to_left_elbow2, init_left_elbow_to_left_wrist2, init_user_max_mpu2, image_rgb, results, landmarks
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    landmarks = results.pose_landmarks.landmark


    init_distance_shoulder2 = get_distance_right_shoulder_to_left_shoulder()

    init_distance_hip_shoulder2 = get_distance_right_hip_to_right_shoulder()
    init_left_distance_hip_shoulder2 = get_distance_left_hip_to_left_shoulder()
    init_height_diff_right_shoulder_to_right_hip2 = get_height_diff_right_shoulder_to_right_hip()
    init_head_width2 = get_head_width()
    init_nose_eye_ear_angle2 = calculate_nose_eyeInR_earR()
    init_right_shoulder_to_right_elbow2 = get_distance_right_shoulder_to_right_elbow()
    init_right_elbow_to_right_wrist2 = get_distance_right_elbow_to_right_wrist()
    init_left_shoulder_to_left_elbow2 = get_distance_left_shoulder_to_left_elbow()
    init_left_elbow_to_left_wrist2 = get_distance_left_elbow_to_left_wrist()
    #init_user_max_mpu = get_distance_left_fingertip_to_elbow(image) + get_distance_left_shoulder_to_left_elbow(image) + get_distance_right_shoulder_to_left_shoulder(image) + get_distance_right_shoulder_to_right_elbow(image) + get_distance_right_fingertip_to_elbow(image)
    init_user_max_mpu2 = get_distance_fingertip_to_fingertip()
    


def data_update(image):
    global user_height, user_depth, left_arm_bicep_force, leftArmAngle
    """
    This method updates all of the input and output data every time its called
    """
    global direction_num, direction_facing, body_yaw, body_roll, body_pitch, test_num, left_hip_x, left_hip_y, left_hip_z, hipShoElb, left_arm_bicep_force, image_rgb, results, landmarks
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    landmarks = results.pose_landmarks.landmark
    
    distance_right = get_distance_right_eye_outer_to_ear()
    distance_left = get_distance_left_eye_outer_to_ear()
    distance_shoulder = get_distance_right_shoulder_to_left_shoulder()
    distance_hip_shoulder = get_distance_right_hip_to_right_shoulder()
    head_width = get_head_width()
    height_diff_shoulder_hip = get_height_diff_right_shoulder_to_right_hip()
    nose_eye_ear_angle = calculate_nose_eyeInR_earR()
    direction_num, direction_facing = calculate_direction(distance_right, distance_left)
    body_yaw = calculate_body_yaw(distance_shoulder, distance_hip_shoulder, direction_facing, (init_distance_shoulder/init_distance_hip_shoulder))
    body_roll = calculate_body_roll()  # Calculate shoulder angle
    body_pitch = calculate_body_pitch(head_width, height_diff_shoulder_hip, init_height_diff_right_shoulder_to_right_hip, nose_eye_ear_angle, init_nose_eye_ear_angle)
    hipShoElb = calculate_left_hip_shoulder_elbow_angle()
    left_hip_x = get_left_hip_x()
    left_hip_y = get_left_hip_y()
    left_hip_z = get_left_hip_z()
    leftShoulderAngle = dot_prod_angle(get_left_elbow_x_y_z(), get_left_shoulder_x_y_z(), get_left_hip_x_y_z())
    leftArmAngle = dot_prod_angle(get_left_wrist_x_y_z(), get_left_elbow_x_y_z(), get_left_shoulder_x_y_z()) 
    print("bicep force input: left shoulder angle: "+str(leftShoulderAngle)+", left arm angle: "+str(leftArmAngle)+", weight added: " + str(weightAdded))
    left_arm_bicep_force = calculate_arm_force(leftShoulderAngle, leftArmAngle, weightAdded)
    test_num = leftArmAngle
    #print("test num updated")


    user_max_mpu = get_distance_fingertip_to_fingertip()
    m_to_mpu_ratio = user_height/user_max_mpu #cm per mpu
    #print("ratio: " + str(m_to_mpu_ratio))


    #(((init_distance_hip_shoulder/init_distance_shoulder))-(((init_distance_hip_shoulder/init_distance_shoulder) * (abs(body_rotation_y-90))/90)))
    

def update_labels():
    """
    This method updates the labels every time it's called
    """
    if twoStepDone:
        if developer_mode:
            direction_facing_label.config(text=f"Direction Facing: {direction_facing}")
            rot_mtx_label.config(text=f"Torso Rotation (Pitch, Yaw, Roll): ({body_pitch if body_pitch else 'N/A'}°, {body_yaw if body_yaw else 'N/A'}°, {round(body_roll,4) if body_roll else 'N/A'}°)")
            body_roll_label.config(text=f"Torso Roll: {body_roll:.2f}°" if body_roll is not None else "Torso Roll: N/A")
            body_yaw_label.config(text=f"Torso Yaw: {body_yaw:.2f}°" if body_yaw else "Torso Yaw: N/A")
            body_pitch_label.config(text=f"Torso Pitch: {body_pitch:.2f}°" if body_pitch else "Torso Pitch: N/A")
        bicep_force_label.config(text=f"Bicep Force: {left_arm_bicep_force if left_arm_bicep_force else 'N/A'}")
        test_num_label.config(text=f"Left Arm Angle: {test_num if test_num else 'N/A'}")
        #test_num_label.config(text=f"Left Hip (X, Y, Z): ({left_hip_x if left_hip_x else 'N/A'}cm, {left_hip_y if left_hip_y else 'N/A'}cm, {left_hip_z if left_hip_z else 'N/A'}cm)")


# Function to update the pose image and data
def update_image():
    ret, frame = cap.read()
    global last_update_time, do_once, wait_for_update, once, leftArmAngle, left_arm_bicep_force, start_time, twoStepDone, isGraphOn

    do_once = False
    # Start the timer when update_image is first called
    if start_time == 0:
        start_time = time.time()

    # Calculate the elapsed time since the start
    elapsed_time = time.time() - start_time

    # Stop the function after 10 seconds
    if elapsed_time > time_simulation_active:
        print(str(time_simulation_active) + " seconds have elapsed, stopping the update.")
        cap.release()
        cv2.destroyAllWindows()
        #root.destroy()
        return # Exit the function to stop the loop

    if last_update_time == 0:  # Check if this is the first time update_image is called
        time.sleep(1)  # Wait for 1 seconds
        last_update_time = time.time()  # Update last_update_time to current time
        do_once = True

    if root.winfo_exists():
        if ret:
            # Process the image and detect the pose
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if wait_for_update > 30:
                if once:
                    init_data_update(image)
                    if developer_mode:
                        print("---------------------------------------------------------------Init Ran------------------------------------------------------------")
                    once = False
                current_time = time.time()
                
                if current_time - last_update_time >= 0.5:  # Check if 0.5 second has passed
                    if elapsed_time <= 5:
                        init2_data_update(image)                   
                    elif  elapsed_time <= 10:
                        instruct_label.config(text=f"Please Step 1 Foot Back To Starting Position\nSimulation Starting in {(round(twoStepCountDown - elapsed_time, 1))}")
                    else:
                        instruct_label.config(text=f"Simulation Started")
                        twoStepDone = True

                    if developer_mode:
                        print("---------------------------------------------------------------Update Ran------------------------------------------------------------")
                    data_update(image) #Updating data to new vals   
                    update_labels() # Update data labels
                    last_update_time = current_time  # Update the last update time

            wait_for_update += wait_for_update + 1

            # Draw the pose annotations
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Convert the image to ImageTk format
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image_tk = ImageTk.PhotoImage(image=image)

            # Update the image on the label
            video_label.config(image=image_tk)
            video_label.image = image_tk
            if twoStepDone:
                if isGraphOn:
                    plot_graph()
                    
        root.after(10, update_image)  # Repeat after an interval











"""
Start up sequence
\/ \/ \/
"""
# Main window setup
main_window = tk.Tk()
main_window.title("Muscle Force Calculator")
main_window.geometry("450x350")



def show_data_input():
    # Hide the start frame and show the data input frame
    start_frame.pack_forget()
    data_input_frame.pack(fill='both', expand=True)

def show_settings():
    # Hide the data input frame and show the settings frame
    data_input_frame.pack_forget()
    settings_frame.pack(fill='both', expand=True)

# Countdown frame setup
countdown_frame = ttk.Frame(main_window)
countdown_label = ttk.Label(countdown_frame, text="Starting in 0 seconds", font=("Helvetica", 16))
countdown_label.pack(padx=20, pady=20)


def on_settings_submit():
    global time_to_get_in_position, time_simulation_active

    time_to_get_in_position = int(time_to_get_position_var.get())
    time_simulation_active = int(time_simulation_var.get())
    print(f"Time to Get in Position: {time_to_get_in_position} seconds, Time Simulation Active: {time_simulation_active} seconds")

    # Hide the current settings frame and show the countdown frame
    settings_frame.pack_forget()
    countdown_frame.pack(fill='both', expand=True)
    
    # Start the countdown
    countdown(time_to_get_in_position)

def countdown(time_left):
    if time_left > 0:
        countdown_label.config(text=f"Starting in {time_left} seconds")
        countdown_frame.after(1000, countdown, time_left - 1)
    else:
        countdown_label.config(text="Starting now!")
        main_window.after(1000, main_window.destroy)  # Close the window after 1 second


# Default value for time_to_get_in_position
# Add a new frame for additional settings (initially hidden)
settings_frame = ttk.Frame(main_window)

# Time to Get in Position Dropdown
time_to_get_position_var = tk.StringVar()
time_to_get_position_label = ttk.Label(settings_frame, text="Time To Get In Position(sec):")
time_to_get_position_label.pack(padx=10, pady=5)
time_to_get_position_dropdown = ttk.Combobox(settings_frame, textvariable=time_to_get_position_var, 
                                             values=[5, 10, 20, 30, 60])
time_to_get_position_dropdown.pack(padx=10, pady=5)

# Time Simulation Active Dropdown
time_simulation_var = tk.StringVar()
time_simulation_label = ttk.Label(settings_frame, text="Time Simulation Active(sec):")
time_simulation_label.pack(padx=10, pady=5)
time_simulation_dropdown = ttk.Combobox(settings_frame, textvariable=time_simulation_var,
                                        values=[30, 60, 120, 300, 1200])
time_simulation_dropdown.pack(padx=10, pady=5)

# Settings Submit Button
settings_submit_button = ttk.Button(settings_frame, text="Submit", command=on_settings_submit)
settings_submit_button.pack(padx=10, pady=15)


# Modify the on_submit function to show the settings frame
def on_submit():
    try:
        global user_weight, user_height, user_depth, weightAdded, developer_mode, isGraphOn, forearm, upperarm, cfg, b, weightForearm
        user_weight = float(weight_entry.get())
        user_height = float(height_entry.get()) * 0.9588 #Mediapipe measures from kunckle to kunckle not fingertip to fingertip
        user_depth = float(depth_entry.get())
        weightAdded = float(weight_holding_entry.get())
        developer_mode = dev_mode_var.get() == 1
        isGraphOn = graph_on_var.get() == 1
        print(f"User Weight: {user_weight} kg, User Height: {user_height} cm, User Depth: {user_depth} cm, Weight Holding: {weightAdded} kg")
        show_settings()  # Show the settings frame instead of destroying the window

        #Updating initial values
        forearm = (user_height*0.01) * 0.216
        upperarm = (user_height*0.01) * 0.173
        cfg = forearm * 0.432
        b = forearm * 0.11
        weightForearm = user_weight * 0.023

    except ValueError:
        print("Please enter valid numbers for weight, height, and depth.")

# Start frame
start_frame = ttk.Frame(main_window)
start_frame.pack(fill='both', expand=True)

title_label = ttk.Label(start_frame, text="Muscle Force Calculator", font=("Helvetica", 16))
title_label.pack(padx=20, pady=20)

start_button = ttk.Button(start_frame, text="Start", command=show_data_input)
start_button.pack(side=tk.BOTTOM, padx=10, pady=10)

# Data input frame setup
data_input_frame = ttk.Frame(main_window)

# Left column for height and depth
left_column = ttk.Frame(data_input_frame)
left_column.pack(side=tk.LEFT, fill='both', expand=True, padx=10, pady=10)

ttk.Label(left_column, text="Enter Your Height (cm):").pack(padx=10, pady=5)
height_entry = ttk.Entry(left_column)
height_entry.pack(padx=10, pady=5)

ttk.Label(left_column, text="Enter Your Distance from Camera (cm):").pack(padx=10, pady=5)
depth_entry = ttk.Entry(left_column)
depth_entry.pack(padx=10, pady=5)

# Right column for user weight, weight holding, and developer mode
right_column = ttk.Frame(data_input_frame)
right_column.pack(side=tk.LEFT, fill='both', expand=True, padx=10, pady=10)

ttk.Label(right_column, text="Enter Your Weight (kg):").pack(padx=10, pady=5)
weight_entry = ttk.Entry(right_column)
weight_entry.pack(padx=10, pady=5)

ttk.Label(right_column, text="Enter The Weight You're Holding (kg):").pack(padx=10, pady=5)
weight_holding_entry = ttk.Entry(right_column)
weight_holding_entry.pack(padx=10, pady=5)

dev_mode_var = tk.IntVar()
dev_mode_check = ttk.Checkbutton(right_column, text="Developer Mode", variable=dev_mode_var)
dev_mode_check.pack(padx=10, pady=5)

graph_on_var = tk.IntVar()
graph_on_check = ttk.Checkbutton(right_column, text="Enable Graph", variable=graph_on_var)
graph_on_check.pack(padx=10, pady=5)

submit_button = ttk.Button(right_column, text="Next", command=on_submit)
submit_button.pack(side=tk.BOTTOM, padx=10, pady=10)

# Start the Tkinter main loop
main_window.mainloop()


"""
/\ /\ /\ 
Start Up Sequence
"""













# Set up the main Tkinter window
root = tk.Tk()
root.title("Pose Detection with Data Output")

# Create the main frame
main_frame = ttk.Frame(root)
main_frame.pack(padx=10, pady=10, fill='both', expand=True)

instruct_label = ttk.Label(main_frame, text="Please Step 1 Foot Forward")
instruct_label.pack(side=tk.BOTTOM, fill='both', expand=True, padx=10, pady=10)

# Create a label in the main frame for video feed
video_label = ttk.Label(main_frame)
video_label.pack(side=tk.LEFT, fill='both', expand=True, padx=10, pady=10)

# Create a frame for data output
data_frame = ttk.LabelFrame(main_frame, text="Data Output")
data_frame.pack(side=tk.RIGHT, fill='both', expand=False, padx=20, pady=10)  # Apply padx and pady here
data_frame.pack_propagate(False)  # Prevent the frame from resizing to fit its content
data_frame.config(width=400, height=600)  # Set the width and height of the frame

graph_frame = ttk.Frame(data_frame)
graph_frame.pack(side=tk.BOTTOM, fill='both', expand=False, padx=10, pady=10)
graph_frame.config(width=400, height=300)  # Set the width and height of the frame




# Create a matplotlib figure and a canvas
fig = plt.Figure(figsize=(5, 4), dpi=100)
canvas = FigureCanvasTkAgg(fig, master=graph_frame)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
# Initialize the plot data
I_values = np.arange(0, 180)  # Elbow Angle values from -90 to 90
current_index = 0  # Start with the first point
plotted_points = []  # List to store the points that have been plotted

plotted_points = []  # Initialize outside of the plot_graph function to store all points

def plot_graph():
    global leftArmAngle, left_arm_bicep_force

    # Check if there's new data to plot and add it to the list
    if leftArmAngle != 0 and left_arm_bicep_force != 0:
        plotted_points.append((leftArmAngle, left_arm_bicep_force))

    # Clear the previous figure
    fig.clear()

    # Create a new plot
    ax = fig.add_subplot(111)

    # Plot all points in the list
    for point in plotted_points:
        ax.scatter(*point, color='blue')

    ax.set_title("Elbow Angle vs Bicep Force")
    ax.set_xlabel("Elbow Angle")
    ax.set_ylabel("Bicep Force")
    ax.set_xlim(0, 180)  # Set the x-axis limits based on your data
    ax.set_ylim(0, max([8100] + [force for _, force in plotted_points]))  # Set the y-axis limit based on the maximum force

    # Draw the updated graph
    canvas.draw()

    # Reset the current data point
    leftArmAngle = 0
    left_arm_bicep_force = 0
    # Plot the graph as soon as the program runs






# Labels for displaying data
if developer_mode:

    direction_facing_label = ttk.Label(data_frame, text="Direction Facing: N/A")
    direction_facing_label.pack(anchor=tk.W)

    rot_mtx_label = ttk.Label(data_frame, text="Rotation Matrix: (x,y,z)")
    rot_mtx_label.pack(anchor=tk.W)

    body_pitch_label = ttk.Label(data_frame, text="Torso Pitch: N/A")
    body_pitch_label.pack(anchor=tk.W)

    body_roll_label = ttk.Label(data_frame, text="Body Rotation (Z-Axis): N/A")
    body_roll_label.pack(anchor=tk.W)

    body_yaw_label = ttk.Label(data_frame, text="Body Yaw: N/A")
    body_yaw_label.pack(anchor=tk.W)

bicep_force_label = ttk.Label(data_frame, text="Force: N/A")
bicep_force_label.pack(anchor=tk.W)

test_num_label = ttk.Label(data_frame, text="Test Num: N/A")
test_num_label.pack(anchor=tk.W)


# Add User Height Input UI Elements
user_height_frame = ttk.Frame(main_frame)
user_height_frame.pack(fill='x', expand=True, pady=5)


"""
def on_confirm_height():
    global user_height, user_depth
    try:
        user_height = float(user_height_entry.get())
        user_depth = float(user_depth_entry.get())
        print(f"User Height: {user_height} cm")
        # You can now use `user_height` for further calculations or display
    except ValueError:
        print("Please enter a valid number for height.")

confirm_height_button = ttk.Button(user_height_frame, text="Confirm", command=on_confirm_height)
confirm_height_button.pack(side=tk.BOTTOM, padx=5, pady=15)

user_height_label = ttk.Label(user_height_frame, text="Enter User Height (cm):")
user_height_label.pack(side=tk.BOTTOM, padx=5, pady=5)

user_height_entry = ttk.Entry(user_height_frame)
user_height_entry.pack(side=tk.BOTTOM, padx=5, pady=5)

user_depth_label = ttk.Label(user_height_frame, text="Enter User Depth (cm):")
user_depth_label.pack(side=tk.BOTTOM, padx=5, pady=5)

user_depth_entry = ttk.Entry(user_height_frame)
user_depth_entry.pack(side=tk.BOTTOM, padx=5, pady=5)
"""


# Open the webcam
cap = cv2.VideoCapture(0)

# Start the periodic update of the image and data
update_image()


# Start the Tkinter main loop
root.mainloop()

# Release resources
cap.release()
cv2.destroyAllWindows()