import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

class PitchAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def calculate_angle(self, a, b, c):
        """Calculate the angle between three points using dot product."""
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        # Handle potential numerical errors
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    def calculate_distance(self, a, b):
        """Calculate the distance between two points."""
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        return np.linalg.norm(a - b)

    def process_video(self, video_path):
        """Process the video and analyze pitching mechanics."""
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create video writer for output
        output_path = 'output_' + video_path
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Draw pose landmarks
                self.mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )
                
                # Extract key joint positions
                landmarks = results.pose_landmarks.landmark
                
                # Calculate important angles
                # Arm angles
                left_arm_angle = self.calculate_angle(
                    landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                    landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value],
                    landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
                )
                
                right_arm_angle = self.calculate_angle(
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
                )
                
                # Hip angles
                left_hip_angle = self.calculate_angle(
                    landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
                    landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
                    landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
                )
                
                right_hip_angle = self.calculate_angle(
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value],
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                )
                
                # Knee angles
                left_knee_angle = self.calculate_angle(
                    landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
                    landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
                    landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
                )
                
                right_knee_angle = self.calculate_angle(
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value],
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                )

                # Calculate stride length (distance between feet)
                stride_length = self.calculate_distance(
                    landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value],
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                ) * width  # Convert to pixels

                # Draw joint positions
                for landmark in [
                    self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                    self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    self.mp_pose.PoseLandmark.LEFT_ELBOW,
                    self.mp_pose.PoseLandmark.RIGHT_ELBOW,
                    self.mp_pose.PoseLandmark.LEFT_WRIST,
                    self.mp_pose.PoseLandmark.RIGHT_WRIST,
                    self.mp_pose.PoseLandmark.LEFT_HIP,
                    self.mp_pose.PoseLandmark.RIGHT_HIP,
                    self.mp_pose.PoseLandmark.LEFT_KNEE,
                    self.mp_pose.PoseLandmark.RIGHT_KNEE,
                    self.mp_pose.PoseLandmark.LEFT_ANKLE,
                    self.mp_pose.PoseLandmark.RIGHT_ANKLE
                ]:
                    x = int(landmarks[landmark.value].x * width)
                    y = int(landmarks[landmark.value].y * height)
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                    cv2.putText(frame, landmark.name.split('_')[1], 
                              (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Display the measurements on the frame
                cv2.putText(frame, f'Left Arm: {left_arm_angle:.1f}', 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f'Right Arm: {right_arm_angle:.1f}', 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f'Left Hip: {left_hip_angle:.1f}', 
                          (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f'Right Hip: {right_hip_angle:.1f}', 
                          (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f'Left Knee: {left_knee_angle:.1f}', 
                          (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f'Right Knee: {right_knee_angle:.1f}', 
                          (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f'Stride Length: {stride_length:.1f}px', 
                          (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Write the frame to output video
            out.write(frame)
            
            # Display the frame
            cv2.imshow('Pitch Analysis', frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    analyzer = PitchAnalyzer()
    video_path = "pitch1.mp4"  # Updated to match the actual video file
    analyzer.process_video(video_path) 