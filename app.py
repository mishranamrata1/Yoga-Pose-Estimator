from flask import Flask, render_template, send_from_directory, Response, jsonify
from PIL import Image
import cv2
import math
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

app = Flask(__name__)

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle

def rescale_frame(frame, percent=50):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

# Function to calculate angle between three points
def calculate_angle(point1, point2, point3):
    vector1 = np.array(point1) - np.array(point2)
    vector2 = np.array(point3) - np.array(point2)
    cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle = math.degrees(np.arccos(cosine_angle))
    return angle

# Function to check if an angle is within a specified range
def is_angle_in_range(angle, min_angle, max_angle):
    return min_angle <= angle <= max_angle

@app.route('/')
def index():
    image_files = ['rep.jpg', 'squat.jpg', 'cobra_pose.jpg', 'DownwardFacingDog.jpg', 'Warrior_pose.jpg', 'Chair-pose.png', 'Plank-pose.jpg', 'Mountain-Pose.jpg', 'Forward-bend.jpg']  # Add more filenames as needed

    # Resize images to a specific dimension
    resized_images = []
    target_size = (150, 150)  # Change to your desired dimension

    for filename in image_files:
        image = Image.open(f'static/img/{filename}')
        image = image.resize(target_size)
        resized_filename = f'{filename}'
        image.save(f'static/resized_poses/{resized_filename}')
        resized_images.append(resized_filename)

    return render_template('index.html', images=resized_images)

def generate_frames_rep():
    global camera
    cap = cv2.VideoCapture(0)
    camera = cap
    counter = 0
    stage = None
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame. Exiting...")
                break
            frame = cv2.flip(frame, 1)
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)

                # Visualize angle
                cv2.putText(image, str(angle),
                            tuple(np.multiply(elbow, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Curl counter logic
                if angle > 160:
                    stage = "down"
                if angle < 40 and stage == 'down':
                    stage = "up"
                    counter += 1
                    print(counter)

            except:
                pass

            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

            # Rep data
            cv2.putText(image, 'REPS', (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter),
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Stage data
            cv2.putText(image, 'STAGE', (65, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage,
                        (60, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

            # Convert the image to bytes and yield for streaming
            frame_bytes = cv2.imencode('.jpg', image)[1].tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    cap.release()

def generate_frames_Squat():
    global camera
    cap = cv2.VideoCapture(0)
    camera = cap
    counter = 0
    stage = None
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame. Exiting...")
                break
            frame = cv2.flip(frame, 1)
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                
                # Get coordinates
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
            
                
                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)
                
                angle_knee = calculate_angle(hip, knee, ankle) #Knee joint angle
                
                angle_hip = calculate_angle(shoulder, hip, knee)
                hip_angle = 180-angle_hip
                knee_angle = 180-angle_knee
                                   
                cv2.putText(image, str(angle_knee), 
                            tuple(np.multiply(knee, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (79, 121, 66), 2, cv2.LINE_AA
                                    )
                
                # Curl counter logic
                if angle_knee > 169:
                    stage = "UP"
                if angle_knee <= 90 and stage =='UP':
                    stage="DOWN"
                    counter +=1
                    print(counter)
            except:
                pass
            
            # Render squat counter
            # Setup status box
            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
            
            # Rep data
            cv2.putText(image, 'Squats', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (65,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (60,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            
            # Convert the image to bytes and yield for streaming
            frame_bytes = cv2.imencode('.jpg', image)[1].tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

        cap.release()

def generate_frames_cobra():
    global camera
    cap = cv2.VideoCapture(0)
    camera = cap
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    frame_images=[]
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break  # Break the loop when the video ends
        frame = cv2.flip(frame, 1)
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Calculate angles between specified connections
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]

            shoulder_elbow_wrist_angle = calculate_angle(shoulder, elbow, wrist)
            shoulder_hip_knee_angle = calculate_angle(shoulder, hip, knee)

            # Check if angles are within specified ranges
            is_shoulder_elbow_wrist_correct = is_angle_in_range(shoulder_elbow_wrist_angle, 130, 175)
            is_shoulder_hip_knee_correct = is_angle_in_range(shoulder_hip_knee_angle, 107, 130)

            # Determine if the pose is "correct" or "incorrect"
            pose_status = "Correct" if is_shoulder_elbow_wrist_correct and is_shoulder_hip_knee_correct else "Incorrect"
            
            # Determine if the pose is "correct" or "incorrect"
            if is_shoulder_elbow_wrist_correct and is_shoulder_hip_knee_correct:
                pose_status = "Correct"
                text_color = (0, 255, 0)  # Green
            else:
                pose_status = "Incorrect"
                text_color = (0, 0, 255)  # Blue

            # Visualize angle
            cv2.putText(image, f'Shoulder_elbow_wrist Angle: {shoulder_elbow_wrist_angle:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(image, f'Shoulder_hip_knee Angle: {shoulder_hip_knee_angle:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            cv2.putText(image, f'Pose Status: {pose_status}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        # Render detections (optional)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    ) 

        frame_images.append(image)

        # Convert the image to bytes and yield for streaming
        frame_bytes = cv2.imencode('.jpg', image)[1].tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
    # Release the video capture and close OpenCV windows
    cap.release()

def generate_frames_DFD():
    global camera
    cap = cv2.VideoCapture(0)
    camera = cap
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break  # Break the loop when the video ends
        frame = cv2.flip(frame, 1)
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Calculate angles between specified connections
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]

            shoulder_elbow_wrist_angle = calculate_angle(shoulder, elbow, wrist)
            shoulder_hip_knee_angle = calculate_angle(shoulder, hip, knee)

            # Check if angles are within specified ranges
            is_shoulder_elbow_wrist_correct = is_angle_in_range(shoulder_elbow_wrist_angle, 150, 175)
            is_shoulder_hip_knee_correct = is_angle_in_range(shoulder_hip_knee_angle, 50, 80)

            # Determine if the pose is "correct" or "incorrect"
            pose_status = "Correct" if is_shoulder_elbow_wrist_correct and is_shoulder_hip_knee_correct else "Incorrect"
            
            # Determine if the pose is "correct" or "incorrect"
            if is_shoulder_elbow_wrist_correct and is_shoulder_hip_knee_correct:
                pose_status = "Correct"
                text_color = (0, 255, 0)  # Green
            else:
                pose_status = "Incorrect"
                text_color = (0, 0, 255)  # Blue


            # Display the pose status
            # Visualize angle
            cv2.putText(image, f'Shoulder_elbow_wrist Angle: {shoulder_elbow_wrist_angle:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(image, f'Shoulder_hip_knee Angle: {shoulder_hip_knee_angle:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            cv2.putText(image, f'Pose Status: {pose_status}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        # Render detections (optional)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    ) 
        
        # Convert the image to bytes and yield for streaming
        frame_bytes = cv2.imencode('.jpg', image)[1].tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
        
    # Release the video capture and close OpenCV windows
    cap.release()

def generate_frames_warrior():
    global camera
    cap = cv2.VideoCapture(0)
    camera = cap
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break  # Break the loop when the video ends
        frame = cv2.flip(frame, 1)
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Calculate angles between specified connections
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]
            
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]
            
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
            
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]

            left_shoulder_elbow_wrist_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_shoulder_elbow_wrist_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            left_hip_knee_ankle_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_hip_knee_ankle_angle = calculate_angle(right_hip, right_knee, right_ankle)

            shoulder_hip_arm_angle = calculate_angle(left_hip, left_shoulder, left_elbow)

            # Check if angles are within specified ranges
            is_left_shoulder_elbow_wrist_correct = is_angle_in_range(left_shoulder_elbow_wrist_angle, 172, 180)
            is_right_shoulder_elbow_wrist_correct = is_angle_in_range(right_shoulder_elbow_wrist_angle, 172, 180)
            
            is_left_hip_knee_ankle_correct = is_angle_in_range(left_hip_knee_ankle_angle, 120, 145)
            is_right_hip_knee_ankle_correct = is_angle_in_range(right_hip_knee_ankle_angle, 170, 180)

            is_shoulder_hip_arm_correct = is_angle_in_range(shoulder_hip_arm_angle, 100, 115)

            # Determine if the pose is "correct" or "incorrect"
            if is_left_shoulder_elbow_wrist_correct and is_right_shoulder_elbow_wrist_correct and \
            is_left_hip_knee_ankle_correct and is_right_hip_knee_ankle_correct and is_shoulder_hip_arm_correct:
                pose_status = "Correct"
                text_color = (0, 255, 0)  # Green
            else:
                pose_status = "Incorrect"
                text_color = (0, 0, 255)  # Blue

            # Display the pose status with the determined text color
            cv2.putText(image, f'Pose Status: {pose_status}', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            cv2.putText(image, f'left_shoulder_elbow_wrist_angle: {left_shoulder_elbow_wrist_angle:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(image, f'right_shoulder_elbow_wrist_angle: {right_shoulder_elbow_wrist_angle:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(image, f'left_hip_knee_ankle_angle: {left_hip_knee_ankle_angle:.2f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(image, f'right_hip_knee_ankle_angle: {right_hip_knee_ankle_angle:.2f}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(image, f'shoulder_hip_arm_angle: {shoulder_hip_arm_angle:.2f}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            
        # Render detections (optional)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

        # Convert the image to bytes and yield for streaming
        frame_bytes = cv2.imencode('.jpg', image)[1].tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
        
    cap.release()

def generate_frames_chair():
    global camera
    cap = cv2.VideoCapture(0)
    camera = cap
    # Initialize the MediaPipe Pose model
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_detection:
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break
            image = cv2.flip(image, 1)
            # Detect Body Pose Landmarks
            results = pose_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Define the landmarks for the Yoga pose
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
                left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]

                # Calculate angles
                def calculate_angle(a, b, c):
                    angle = math.degrees(math.acos((b.x - a.x) * (c.x - b.x)) + (b.y - a.y) * (c.y - b.y) /
                                (math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2) * math.sqrt((c.x - b.x) ** 2 + (c.y - b.y) ** 2)))
                    return angle

                angle_shoulder_hip_knee = calculate_angle(left_shoulder, left_hip, left_knee)
                angle_hip_knee_feet = calculate_angle(left_hip, left_knee, left_ankle)

                # Define the angle thresholds for correct chair pose
                shoulder_hip_knee_threshold = (135, 145)
                hip_knee_feet_threshold = (132, 138)

                # Check if the pose is correct
                if (shoulder_hip_knee_threshold[0] <= angle_shoulder_hip_knee <= shoulder_hip_knee_threshold[1] and
                    hip_knee_feet_threshold[0] <= angle_hip_knee_feet <= hip_knee_feet_threshold[1]):
                    correctness_text = "Correct Chair Pose (Utkatasana)"
                else:
                    correctness_text = "Incorrect Pose"

                # Display angles and pose correctness on the frame
                cv2.putText(image, f"Shoulder-Hip-Knee Angle: {angle_shoulder_hip_knee:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(image, f"Hip-Knee-Feet Angle: {angle_hip_knee_feet:.2f} degrees", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(image, correctness_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw landmarks and connections on the frame
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            # Convert the image to bytes and yield for streaming
            frame_bytes = cv2.imencode('.jpg', image)[1].tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
            
        cap.release()

def generate_frames_mountain():
    global camera
    cap = cv2.VideoCapture(0)
    camera = cap
    # Load the MediaPipe Pose model
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        # Convert frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform pose estimation
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            # Extract landmarks for left and right wrists (landmarks[9] and landmarks[10])
            left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

            if left_wrist and right_wrist:
                # Calculate the Euclidean distance between the wrists
                wrist_distance = math.dist((left_wrist.x, left_wrist.y), (right_wrist.x, right_wrist.y))

                # Define a threshold for correct Mountain Pose wrist distance
                threshold_distance = 0.1  # Adjust this threshold as needed

                # Assess the correctness of the pose based on wrist distance
                if wrist_distance < threshold_distance:
                    pose_correct = True
                    correctness_text = "Correct Mountain Pose"
                else:
                    pose_correct = False
                    correctness_text = "Incorrect Mountain Pose"

                # Display correctness text on the frame
                cv2.putText(frame, correctness_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, 8)

        # Draw landmarks and connections on the frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Convert the image to bytes and yield for streaming
        frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    cap.release()

def generate_frames_FBP():
    global camera
    cap = cv2.VideoCapture(0)
    camera = cap
    # Initialize the MediaPipe Pose model
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_detection:        
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            # Detect Body Pose Landmarks
            results = pose_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Define the landmarks for the Yoga pose
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]

                # Calculate the angle
                def calculate_angle(a, b, c):
                    angle = math.degrees(math.acos((b.x - a.x) * (c.x - b.x)) + (b.y - a.y) * (c.y - b.y) /
                                (math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2) * math.sqrt((c.x - b.x) ** 2 + (c.y - b.y) ** 2)))
                    return angle

                angle_shoulder_hip_knee = calculate_angle(left_shoulder, left_hip, left_knee)

                # Define the angle threshold for correct Uttanasana
                shoulder_hip_knee_threshold = (32, 42)  # Adjust the range as needed

                # Check if the pose is correct
                if shoulder_hip_knee_threshold[0] <= angle_shoulder_hip_knee <= shoulder_hip_knee_threshold[1]:
                    correctness_text = "Correct Uttanasana (Standing Forward Bend)"
                else:
                    correctness_text = "Incorrect Uttanasana (Standing Forward Bend)"

                # Display the angle and pose correctness on the frame
                cv2.putText(image, f"Shoulder-Hip-Knee Angle: {angle_shoulder_hip_knee:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(image, correctness_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Draw landmarks and connections on the frame
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            # Convert the image to bytes and yield for streaming
            frame_bytes = cv2.imencode('.jpg', image)[1].tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    cap.release()

def generate_frames_plank():
    global camera
    cap = cv2.VideoCapture(0)
    camera = cap
    # Initialize the MediaPipe Pose model
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_detection:
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            # Detect Body Pose Landmarks
            results = pose_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Define the landmarks for the Yoga pose
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]

                # Calculate the angles
                def calculate_angle(a, b, c):
                    angle = math.degrees(math.acos((b.x - a.x) * (c.x - b.x)) + (b.y - a.y) * (c.y - b.y) /
                                (math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2) * math.sqrt((c.x - b.x) ** 2 + (c.y - b.y) ** 2)))
                    return angle

                angle_shoulder_elbow_wrist = calculate_angle(left_shoulder, left_elbow, left_wrist)
                angle_elbow_shoulder_hip = calculate_angle(left_elbow, left_shoulder, left_hip)
                angle_knee_hip_shoulder = calculate_angle(left_knee, left_hip, left_shoulder)

                # Define the angle thresholds for the plank pose
                shoulder_elbow_wrist_threshold = (85, 100)
                elbow_shoulder_hip_threshold = (67, 75)
                knee_hip_shoulder_threshold = (85, 95)

                # Check if the pose is correct
                if (shoulder_elbow_wrist_threshold[0] <= angle_shoulder_elbow_wrist <= shoulder_elbow_wrist_threshold[1] and
                    elbow_shoulder_hip_threshold[0] <= angle_elbow_shoulder_hip <= elbow_shoulder_hip_threshold[1] and
                    knee_hip_shoulder_threshold[0] <= angle_knee_hip_shoulder <= knee_hip_shoulder_threshold[1]):
                    correctness_text = "Correct Plank Pose"
                else:
                    correctness_text = "Incorrect Pose"

                # Display the angles and pose correctness on the frame
                cv2.putText(image, f"Shoulder-Elbow-Wrist Angle: {angle_shoulder_elbow_wrist:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(image, f"Elbow-Shoulder-Hip Angle: {angle_elbow_shoulder_hip:.2f} degrees", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(image, f"Knee-Hip-Shoulder Angle: {angle_knee_hip_shoulder:.2f} degrees", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(image, correctness_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Draw landmarks and connections on the frame
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            # Convert the image to bytes and yield for streaming
            frame_bytes = cv2.imencode('.jpg', image)[1].tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
            
        cap.release()


@app.route('/stop_camera')
def stop_camera():
    global camera
    if camera is not None:
        camera.release()  # Release the camera resources
        camera = None
        return jsonify({'message': 'Camera stopped successfully'})
    else:
        return jsonify({'message': 'Camera not started'})



@app.route('/video_feed/<image_id>')
def video_feed(image_id):
    if image_id=="rep":
        return Response(generate_frames_rep(), mimetype='multipart/x-mixed-replace; boundary=frame')
    elif image_id=="squat":
        return Response(generate_frames_Squat(), mimetype='multipart/x-mixed-replace; boundary=frame')
    elif image_id=="cobra":
        return Response(generate_frames_cobra(), mimetype='multipart/x-mixed-replace; boundary=frame')
    elif image_id=="DFD":
        return Response(generate_frames_DFD(), mimetype='multipart/x-mixed-replace; boundary=frame')
    elif image_id=="warrior":
        return Response(generate_frames_warrior(), mimetype='multipart/x-mixed-replace; boundary=frame')
    elif image_id=="chair":
        return Response(generate_frames_chair(), mimetype='multipart/x-mixed-replace; boundary=frame')
    elif image_id=="mountain":
        return Response(generate_frames_mountain(), mimetype='multipart/x-mixed-replace; boundary=frame')
    elif image_id=="FBP":
        return Response(generate_frames_FBP(), mimetype='multipart/x-mixed-replace; boundary=frame')
    elif image_id=="plank":
        return Response(generate_frames_plank(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/input_feed/<image_id>')
def input_feed(image_id):
    return render_template('input_feed.html', image_id=image_id)

@app.route('/lastpage')
def lastpage():
    return render_template('lastpage.html')

@app.route('/resized_poses/<filename>')
def resized_image(filename):
    return send_from_directory('static/resized_poses', filename)

if __name__ == '__main__':
    app.run(debug=True)

