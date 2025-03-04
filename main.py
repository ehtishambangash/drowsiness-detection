import cv2
import dlib
import numpy as np
import time
import datetime
import csv
from scipy.spatial import distance as dist
from collections import deque

class DrowsinessDetectionSystem:
    def __init__(self):
        self.EYE_AR_THRESH = 0.25        # Eye aspect ratio threshold
        self.EYE_AR_CONSEC_FRAMES = 48   # Number of consecutive frames to trigger eye alarm
        self.MOUTH_AR_THRESH = 0.6       # Mouth aspect ratio threshold
        self.MOUTH_AR_CONSEC_FRAMES = 30  # Number of consecutive frames to trigger yawn alarm
        
        self.HEAD_TILT_THRESH = 15       # Degrees threshold for head tilt
        
        # Initialize blink parameters
        self.BLINK_THRESH = 0.25         # Blink threshold
        self.MIN_BLINK_FRAMES = 3        # Minimum number of frames for a blink
        self.BLINKS_PER_MINUTE_THRESH = [10, 30]  # Normal range [min, max]
        
        # Initialize counters
        self.eye_counter = 0
        self.mouth_counter = 0
        self.head_counter = 0
        self.blink_counter = 0
        self.frame_counter = 0
        
        # Initialize blink tracking
        self.blinks = 0
        self.last_minute_blinks = deque(maxlen=60)  # Store last minute of blink data
        self.blink_started = False
        
        # Initialize alert flags
        self.eye_alarm_on = False
        self.mouth_alarm_on = False
        self.head_alarm_on = False
        self.blink_alarm_on = False
        
        # Initialize timestamps
        self.start_time = time.time()
        self.last_blink_check = time.time()
        
        # Initialize detectors
        print("Loading facial landmark predictor...")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # Initialize log file
        self.log_filename = f"drowsiness_log.csv"
        with open(self.log_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Event', 'EAR', 'MAR', 'Head_Tilt', 'Blink_Rate'])
    
    def eye_aspect_ratio(self, eye):
        # Calculate the vertical distances
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        
        # Calculate the horizontal distance
        C = dist.euclidean(eye[0], eye[3])
        
        # Calculate the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear
    
    def mouth_aspect_ratio(self, mouth):
        # Calculate the vertical distances
        A = dist.euclidean(mouth[2], mouth[10])
        B = dist.euclidean(mouth[4], mouth[8])
        C = dist.euclidean(mouth[0], mouth[6])
        
        # Calculate the mouth aspect ratio
        mar = (A + B) / (2.0 * C)
        return mar
    
    def calculate_head_pose(self, shape):
        # Get facial landmarks
        image_points = np.array([
            shape[30],  # Nose tip
            shape[8],   # Chin
            shape[36],  # Left eye left corner
            shape[45],  # Right eye right corner
            shape[48],  # Left mouth corner
            shape[54],  # Right mouth corner
        ], dtype="double")
        
        nose_chin_vector = np.array(shape[8]) - np.array(shape[30])
        vertical_angle = np.degrees(np.arctan2(nose_chin_vector[1], nose_chin_vector[0]) - np.pi/2)
        
        return abs(vertical_angle)
    
    def log_event(self, event, ear, mar, head_tilt, blink_rate):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, event, ear, mar, head_tilt, blink_rate])
    
    def check_blink_rate(self):
        # Check blink rate every minute
        current_time = time.time()
        if current_time - self.last_blink_check >= 60:
            blink_rate = sum(self.last_minute_blinks)
            self.last_blink_check = current_time
            self.last_minute_blinks.append(self.blinks)
            self.blinks = 0
            
            # Check if blink rate is abnormal
            if blink_rate < self.BLINKS_PER_MINUTE_THRESH[0] or blink_rate > self.BLINKS_PER_MINUTE_THRESH[1]:
                self.blink_alarm_on = True
                self.log_event("Abnormal Blink Rate", 0, 0, 0, blink_rate)
                return True
        return False
    
    def process_frame(self, frame):
        # Resize frame
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.detector(gray, 0)
        
        # Create alarm frame (red border) for visual alerts
        alarm_frame = frame.copy()
        
        if len(faces) > 0:
            face = faces[0]
            
            # Get facial landmarks
            shape = self.predictor(gray, face)
            shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
            
            # Extract eye coordinates
            left_eye = shape[42:48]
            right_eye = shape[36:42]
            
            # Extract mouth coordinates
            mouth = shape[48:68]
            left_ear = self.eye_aspect_ratio(left_eye)
            right_ear = self.eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            
            mar = self.mouth_aspect_ratio(mouth)
            head_tilt = self.calculate_head_pose(shape)
            
            # Check for eye closure
            if ear < self.EYE_AR_THRESH:
                self.eye_counter += 1
                
                # Check for blink
                if not self.blink_started and self.eye_counter >= self.MIN_BLINK_FRAMES:
                    self.blink_started = True
                
                # Check for prolonged eye closure
                if self.eye_counter >= self.EYE_AR_CONSEC_FRAMES:
                    if not self.eye_alarm_on:
                        self.eye_alarm_on = True
                        self.log_event("Eyes Closed", ear, mar, head_tilt, 0)
                    
                    # Visual alert - red border
                    cv2.rectangle(alarm_frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 20)
                    cv2.putText(alarm_frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    frame = alarm_frame
            else:
                # Reset eye counter if eyes are open
                if self.blink_started:
                    self.blinks += 1
                    self.blink_started = False
                    
                self.eye_counter = 0
                self.eye_alarm_on = False
            
            # Check for yawning
            if mar > self.MOUTH_AR_THRESH:
                self.mouth_counter += 1
                
                if self.mouth_counter >= self.MOUTH_AR_CONSEC_FRAMES:
                    if not self.mouth_alarm_on:
                        self.mouth_alarm_on = True
                        self.log_event("Yawning", ear, mar, head_tilt, 0)
                    
                    # Visual alert - red border
                    cv2.rectangle(alarm_frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 20)
                    cv2.putText(alarm_frame, "YAWNING - TAKE A BREAK!", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    frame = alarm_frame
            else:
                self.mouth_counter = 0
                self.mouth_alarm_on = False
            
            # Check for head tilt
            if head_tilt > self.HEAD_TILT_THRESH:
                self.head_counter += 1
                
                if self.head_counter >= 15:  # Approx half a second at 30fps
                    if not self.head_alarm_on:
                        self.head_alarm_on = True
                        self.log_event("Head Tilting", ear, mar, head_tilt, 0)
                    
                    # Visual alert - red border
                    cv2.rectangle(alarm_frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 20)
                    cv2.putText(alarm_frame, "HEAD TILT DETECTED!", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    frame = alarm_frame
            else:
                self.head_counter = 0
                self.head_alarm_on = False
            
            # Check blink rate periodically
            if self.check_blink_rate():
                cv2.rectangle(alarm_frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 20)
                cv2.putText(alarm_frame, "ABNORMAL BLINK RATE!", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                frame = alarm_frame
            
            # Draw facial landmarks
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                
            # Draw eyes
            self.draw_eye_contours(frame, left_eye)
            self.draw_eye_contours(frame, right_eye)
            
            # Draw mouth
            self.draw_mouth_contours(frame, mouth)
            
            # Display EAR and MAR values
            cv2.putText(frame, f"EAR: {ear:.2f}", (500, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (500, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Tilt: {head_tilt:.2f}", (500, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Every 5 seconds, log normal state data
            self.frame_counter += 1
            if self.frame_counter % 150 == 0:  # Assuming 30fps
                self.log_event("Normal", ear, mar, head_tilt, 0)
        else:
            cv2.putText(frame, "No Face Detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display elapsed time
        elapsed_time = time.time() - self.start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))
        cv2.putText(frame, f"Drive Time: {time_str}", (10, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def draw_eye_contours(self, frame, eye):
        eye_hull = cv2.convexHull(eye)
        cv2.drawContours(frame, [eye_hull], -1, (0, 255, 0), 1)
    
    def draw_mouth_contours(self, frame, mouth):
        mouth_hull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouth_hull], -1, (0, 255, 0), 1)

def main():
    dds = DrowsinessDetectionSystem()
    print("Starting video stream...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        processed_frame = dds.process_frame(frame)
        cv2.imshow("Driver Drowsiness Detection", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()