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

def calculate_body_rotation(distance_shoulder, distance_hip_shoulder, direction_facing):
    """
    Calculate the body rotation based on the distances and the direction facing.
    """
    if distance_shoulder is not None and distance_hip_shoulder is not None and distance_hip_shoulder != 0:
        if direction_facing == "Right":
            return round((((distance_shoulder / distance_hip_shoulder)/0.55)*90), 4)
        else:
            return round(180-(((distance_shoulder / distance_hip_shoulder)/0.55)*90),4)
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

def data_update(image):
    global direction_num, direction_facing, body_rotation_y, body_rotation_z
    distance_right = get_distance_right_eye_outer_to_ear(image)
    distance_left = get_distance_left_eye_outer_to_ear(image)
    distance_shoulder = get_distance_right_shoulder_to_left_shoulder(image)
    distance_hip_shoulder = get_distance_right_hip_to_right_shoulder(image)
    direction_num, direction_facing = calculate_direction(distance_right, distance_left)
    body_rotation_y = calculate_body_rotation(distance_shoulder, distance_hip_shoulder, direction_facing)
    body_rotation_z = calculate_shoulder_angle(image)  # Calculate shoulder angle
            


def update_labels():
    direction_num_label.config(text=f"Direction Num: {round(direction_num, 4) if direction_num else 'N/A'}")
    direction_facing_label.config(text=f"Direction Facing: {direction_facing}")
    body_rot_y_num_label.config(text=f"Body Rotation (Y-Axis): {body_rotation_y if body_rotation_y else 'N/A'}")
    rot_mtx_label.config(text=f"Rotation Matrix (X, Y, Z): (55, {round(body_rotation_y,4) if body_rotation_y else 'N/A'}, {round(body_rotation_z,4) if body_rotation_z else 'N/A'})")
    body_rot_z_num_label.config(text=f"Shoulder Angle: {body_rotation_z:.2f}°" if body_rotation_z is not None else "Shoulder Angle: N/A")


# Function to update the pose image and data
def update_image():
    ret, frame = cap.read()
    global last_update_time
    if ret:
        # Process the image and detect the pose
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

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
direction_num_label = ttk.Label(data_frame, text="Direction Num: N/A")
direction_num_label.pack(anchor=tk.W)

direction_facing_label = ttk.Label(data_frame, text="Direction Facing: N/A")
direction_facing_label.pack(anchor=tk.W)

body_rot_y_num_label = ttk.Label(data_frame, text="Body Rotation (Y-Axis): N/A")
body_rot_y_num_label.pack(anchor=tk.W)

rot_mtx_label = ttk.Label(data_frame, text="Rotation Matrix: (x,y,z)")
rot_mtx_label.pack(anchor=tk.W)

body_rot_z_num_label = ttk.Label(data_frame, text="Body Rotation (Z-Axis): N/A")
body_rot_z_num_label.pack(anchor=tk.W)



# Open the webcam
cap = cv2.VideoCapture(0)

# Start the periodic update of the image and data
update_image()

# Start the Tkinter main loop
root.mainloop()

# Release resources
cap.release()
cv2.destroyAllWindows()
