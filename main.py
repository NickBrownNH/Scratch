import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import time
import math



# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
# Variables for data output (placeholders)
direction_num = 0
direction_facing = "Unknown"
last_update_time = 0  # Variable to track the time of the last update
body_rotation_z = 0
max_shoulder_size = 0
tickCheck = 0



def calculate_3d_angle(A, B, C):
    """
    Calculate the 3D angle between vectors BA and BC using their 3D coordinates.

    Parameters:
    - A, B, C: The 3D coordinates (x, y, z) of points A, B, and C.

    Returns:
    - angle_deg: The angle in degrees between vectors BA and BC.
    """
    BA = np.array([A.x - B.x, A.y - B.y, A.z - B.z])
    BC = np.array([C.x - B.x, C.y - B.y, C.z - B.z])
    cosine_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    angle_deg = np.degrees(angle)
    return angle_deg

def calculate_distance(landmark1, landmark2):
    """
    Calculate the Euclidean distance between two landmarks.
    """
    return np.sqrt((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2)

def get_distance_right_eye_outer_to_ear(image):
    """
    Get the distance between the right eye outer and the right ear using MediaPipe Pose.
    """
    distance = None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        right_eye_outer = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE_OUTER]
        right_ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]

        # Calculate the distance
        distance = calculate_distance(right_eye_outer, right_ear)

    return distance

def get_distance_left_eye_outer_to_ear(image):
    """
    Get the distance between the left eye outer and the left ear using MediaPipe Pose.
    """
    distance = None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        left_eye_outer = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE_OUTER]
        left_ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]

        # Calculate the distance
        distance = calculate_distance(left_eye_outer, left_ear)

    return distance

def get_distance_right_hip_to_right_shoulder(image):
    """
    Get the distance between the right hip and the right shoulder using MediaPipe Pose.
    """
    distance = None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Calculate the distance
        distance = calculate_distance(right_hip, right_shoulder)

    return distance

def get_distance_right_shoulder_to_left_shoulder(image):
    """
    Get the distance between the right shoulder and the left shoulder using MediaPipe Pose.
    """
    distance = None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]

        # Calculate the distance
        distance = calculate_distance(right_shoulder, left_shoulder)

    return distance

def get_distance_right_hip_to_left_hip(image):
    """
    Get the distance between the right hip and the left hip using MediaPipe Pose.
    """
    distance = None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]

        # Calculate the distance
        distance = calculate_distance(right_hip, left_hip)

    return distance

def get_head_width(image):
    """
    Get the distance between the right hip and the left hip using MediaPipe Pose.
    """
    distance = None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        right_ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]
        left_ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]

        # Calculate the distance
        distance = calculate_distance(right_ear, left_ear)

    return distance

def get_height_diff_right_shoulder_to_right_hip(image):
    """
    Get the distance between the right hip and the left hip using MediaPipe Pose.
    """
    distance = None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

        # Calculate the distance
        distance = right_hip.y - right_shoulder.y

    return distance





# Test the function with an image
# image = cv2.imread("path_to_your_image.jpg")
# distance = get_distance_right_eye_outer_to_ear(image)
# print("Distance:", distance)

def calculate_direction(distance_right, distance_left):
    """
    Calculate the direction the person is facing based on the distances.
    """
    if distance_right is not None and distance_left is not None and distance_right != 0 and distance_left != 0:
        direction_num = distance_right / distance_left
        direction_facing = "Right" if direction_num <= 1 else "Left"
        return direction_num, direction_facing
    return None, "Unknown"

def calculate_body_rotation(distance_shoulder, distance_hip_shoulder, direction_facing, init_val):
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
                print("up")
                return round(-(90-((height_diff_hip_shoulder / head_width)/(init_val)*90)), 4) #init_val = ?
            else:
                print("down")
                return round((90-((height_diff_hip_shoulder / head_width)/(init_val)*90)), 4) #init_val = ?
    return 0






def calculate_shoulder_angle(image):
    """
    Calculate the angle between the line connecting the shoulders and the horizontal line.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
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

def calculate_left_arm_angle(image):
    """
    Basic calculation to find the angle between the left shoulder, left elbow, and left wrist.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    landmarks = results.pose_landmarks.landmark
    
    if results.pose_landmarks:
        # Get landmarks for shoulders
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
 
        angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

        return angle
    return None

def calculate_nose_eyeInR_earR(image):
    """
    Calculates the angle between the nose, right inner eye, and right ear.
    *Used to calculate whether the user is facing up or down 
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    landmarks = results.pose_landmarks.landmark
    
    if results.pose_landmarks:
        # Get landmarks for shoulders
        nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y]
        right_eye_inner = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].y]
        right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
        angle = calculate_angle(nose, right_eye_inner, right_ear)

        return angle
    return None

def calculate_arm_3d(image):
    """
    Calculates the angle between the nose, right inner eye, and right ear.
    *Used to calculate whether the user is facing up or down 
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    landmarks = results.pose_landmarks.landmark
    
    if results.pose_landmarks:
        # Get landmarks for shoulders
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]

        angle = left_wrist.z
        return angle
    return None





def init_data_update(image):
    """
    This method is called once before the program begins updating calculations so that initial values can be found for the user's specifc body ratios
    """
    global init_distance_shoulder, init_distance_hip_shoulder, init_height_diff_right_shoulder_to_right_hip, init_head_width, init_nose_eye_ear_angle
    init_distance_shoulder = get_distance_right_shoulder_to_left_shoulder(image)
    init_distance_hip_shoulder = get_distance_right_hip_to_right_shoulder(image)
    init_height_diff_right_shoulder_to_right_hip = get_height_diff_right_shoulder_to_right_hip(image)
    init_head_width = get_head_width(image)
    init_nose_eye_ear_angle = calculate_nose_eyeInR_earR(image)


def data_update(image):
    """
    This method updates all of the input and output data every time its called
    """
    global direction_num, direction_facing, body_rotation_y, body_rotation_z, body_pitch, test_num
    distance_right = get_distance_right_eye_outer_to_ear(image)
    distance_left = get_distance_left_eye_outer_to_ear(image)
    distance_shoulder = get_distance_right_shoulder_to_left_shoulder(image)
    distance_hip_shoulder = get_distance_right_hip_to_right_shoulder(image)
    head_width = get_head_width(image)
    height_diff_shoulder_hip = get_height_diff_right_shoulder_to_right_hip(image)
    nose_eye_ear_angle = calculate_nose_eyeInR_earR(image)
    direction_num, direction_facing = calculate_direction(distance_right, distance_left)
    body_rotation_y = calculate_body_rotation(distance_shoulder, distance_hip_shoulder, direction_facing, (init_distance_shoulder/init_distance_hip_shoulder))
    body_rotation_z = calculate_shoulder_angle(image)  # Calculate shoulder angle
    body_pitch = calculate_body_pitch(head_width, height_diff_shoulder_hip, (init_height_diff_right_shoulder_to_right_hip/init_head_width), nose_eye_ear_angle, init_nose_eye_ear_angle)
    test_num = calculate_arm_3d(image)
    #(((init_distance_hip_shoulder/init_distance_shoulder))-(((init_distance_hip_shoulder/init_distance_shoulder) * (abs(body_rotation_y-90))/90)))
            


def update_labels():
    """
    This method updates the labels every time it's called
    """
    direction_facing_label.config(text=f"Direction Facing: {direction_facing}")
    rot_mtx_label.config(text=f"Torso Rotation (Pitch, Yaw, Roll): ({body_pitch if body_pitch else 'N/A'}, {body_rotation_y if body_rotation_y else 'N/A'}, {round(body_rotation_z,4) if body_rotation_z else 'N/A'})")
    body_rot_z_num_label.config(text=f"Torso Roll: {body_rotation_z:.2f}°" if body_rotation_z is not None else "Torso Roll: N/A")
    body_rot_y_num_label.config(text=f"Torso Yaw: {body_rotation_y:.2f}°" if body_rotation_y else "Torso Yaw: N/A")
    body_rot_x_num_label.config(text=f"Torso Pitch: {body_pitch:.2f}°" if body_pitch else "Torso Pitch: N/A")
    tets_num_label.config(text=f"TestNum: {test_num if test_num else 'N/A'}")



# Function to update the pose image and data
def update_image():
    ret, frame = cap.read()
    global last_update_time, do_once
    do_once = False



    if last_update_time == 0:  # Check if this is the first time update_image is called
        time.sleep(5)  # Wait for 5 seconds
        last_update_time = time.time()  # Update last_update_time to current time
        do_once = True



    if ret:
        # Process the image and detect the pose
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        if do_once:
            init_data_update(image)
            print("---------------------------------------------------------------Init Ran------------------------------------------------------------")

        current_time = time.time()
        if current_time - last_update_time >= 0.5:  # Check if 0.5 second has passed
            data_update(image) #Updating data to new vals   
            update_labels() # Update data labels
            last_update_time = current_time  # Update the last update time            

        # Draw the pose annotations
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Convert the image to ImageTk format
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image_tk = ImageTk.PhotoImage(image=image)

        # Update the image on the label
        video_label.config(image=image_tk)
        video_label.image = image_tk
                
    root.after(10, update_image)  # Repeat after an interval

# Set up the main Tkinter window
root = tk.Tk()
root.title("Pose Detection with Data Output")

# Create the main frame
main_frame = ttk.Frame(root)
main_frame.pack(padx=10, pady=10, fill='both', expand=True)

# Create a label in the main frame for video feed
video_label = ttk.Label(main_frame)
video_label.pack(side=tk.LEFT, padx=10, pady=10)

# Create a frame for data output
data_frame = ttk.LabelFrame(main_frame, text="Data Output")
data_frame.pack(side=tk.RIGHT, fill='both', expand=False, padx=100, pady=10)  # Apply padx and pady here
data_frame.pack_propagate(False)  # Prevent the frame from resizing to fit its content
data_frame.config(width=300, height=200)  # Set the width and height of the frame


# Labels for displaying data
direction_facing_label = ttk.Label(data_frame, text="Direction Facing: N/A")
direction_facing_label.pack(anchor=tk.W)

rot_mtx_label = ttk.Label(data_frame, text="Rotation Matrix: (x,y,z)")
rot_mtx_label.pack(anchor=tk.W)

body_rot_x_num_label = ttk.Label(data_frame, text="Body Rotation (X-Axis): N/A")
body_rot_x_num_label.pack(anchor=tk.W)

body_rot_z_num_label = ttk.Label(data_frame, text="Body Rotation (Z-Axis): N/A")
body_rot_z_num_label.pack(anchor=tk.W)

body_rot_y_num_label = ttk.Label(data_frame, text="Body Rotation (Y-Axis): N/A")
body_rot_y_num_label.pack(anchor=tk.W)

tets_num_label = ttk.Label(data_frame, text="Test Num: N/A")
tets_num_label.pack(anchor=tk.W)


# Open the webcam
cap = cv2.VideoCapture(0)


# Start the periodic update of the image and data
update_image()



# Start the Tkinter main loop
root.mainloop()

# Release resources
cap.release()
cv2.destroyAllWindows()
