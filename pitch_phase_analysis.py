import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

class PitchPhaseAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize phase detection parameters
        self.window_size = 10  # Number of frames to keep for velocity calculation
        self.wrist_positions = deque(maxlen=self.window_size)
        self.phase = "Initial"
        self.phases = []
        
        # Thresholds for phase detection
        self.stride_threshold = 0.3  # Normalized stride length threshold
        self.arm_angle_threshold = 90  # Degrees
        self.velocity_threshold = 0.1  # Normalized velocity threshold

        # Ideal mechanics thresholds
        self.IDEAL_STRIDE_RATIO = 0.8  # 80% of height
        self.IDEAL_ELBOW_ANGLE = 90
        self.IDEAL_WRIST_VELOCITY = 0.05  # Adjust based on testing
        self.IDEAL_HIP_ROTATION = 45  # Degrees
        self.IDEAL_KNEE_FLEXION = 90  # Degrees

    def calculate_angle(self, a, b, c):
        """Calculate the angle between three points using dot product."""
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    def calculate_distance(self, a, b):
        """Calculate the distance between two points."""
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        return np.linalg.norm(a - b)

    def calculate_velocity(self, current_pos, prev_pos):
        """Calculate velocity between two positions."""
        if prev_pos is None:
            return 0
        return np.linalg.norm(current_pos - prev_pos)

    def generate_feedback(self, stride_length, arm_angle, wrist_velocity, hip_angle, knee_angle):
        """Generate feedback based on current mechanics."""
        feedback = []
        
        # Stride length feedback
        if stride_length < self.IDEAL_STRIDE_RATIO:
            feedback.append("Increase stride length for more power")
        elif stride_length > self.IDEAL_STRIDE_RATIO * 1.2:
            feedback.append("Stride length too long - may affect balance")
            
        # Arm angle feedback
        if arm_angle < self.IDEAL_ELBOW_ANGLE:
            feedback.append("Improve external shoulder rotation")
        elif arm_angle > self.IDEAL_ELBOW_ANGLE * 1.2:
            feedback.append("Arm slot too high - adjust mechanics")
            
        # Wrist velocity feedback
        if wrist_velocity < self.IDEAL_WRIST_VELOCITY:
            feedback.append("Increase arm speed at release")
            
        # Hip rotation feedback
        if hip_angle < self.IDEAL_HIP_ROTATION:
            feedback.append("Increase hip rotation for more power")
            
        # Knee flexion feedback
        if knee_angle < self.IDEAL_KNEE_FLEXION:
            feedback.append("Increase knee flexion for better drive")
            
        return feedback if feedback else ["Good mechanics!"]

    def detect_phase(self, stride_length, arm_angle, wrist_velocity):
        """Detect the current phase of the pitch based on metrics."""
        if stride_length < self.stride_threshold and arm_angle < self.arm_angle_threshold:
            return "Windup"
        elif stride_length >= self.stride_threshold and arm_angle < self.arm_angle_threshold:
            return "Stride"
        elif wrist_velocity > self.velocity_threshold:
            return "Arm Acceleration"
        elif arm_angle > self.arm_angle_threshold:
            return "Follow Through"
        else:
            return "Initial"

    def process_video(self, video_path):
        """Process the video and analyze pitching phases."""
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create video writer for output
        output_path = 'output_phase_' + video_path
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        frame_count = 0
        prev_wrist_pos = None
        
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
                
                # Calculate metrics
                # Stride length
                stride_length = self.calculate_distance(
                    landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value],
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                )
                
                # Arm angle
                arm_angle = self.calculate_angle(
                    landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                    landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value],
                    landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
                )
                
                # Hip rotation
                hip_angle = self.calculate_angle(
                    landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                    landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
                )
                
                # Knee flexion
                knee_angle = self.calculate_angle(
                    landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
                    landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
                    landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
                )
                
                # Wrist velocity
                current_wrist_pos = np.array([
                    landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                    landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y
                ])
                self.wrist_positions.append(current_wrist_pos)
                
                if len(self.wrist_positions) > 1:
                    wrist_velocity = self.calculate_velocity(
                        current_wrist_pos,
                        self.wrist_positions[-2]
                    )
                else:
                    wrist_velocity = 0
                
                # Detect current phase
                current_phase = self.detect_phase(stride_length, arm_angle, wrist_velocity)
                if current_phase != self.phase:
                    self.phase = current_phase
                    self.phases.append((frame_count, current_phase))
                
                # Generate feedback
                feedback = self.generate_feedback(
                    stride_length, arm_angle, wrist_velocity,
                    hip_angle, knee_angle
                )
                
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
                
                # Display metrics and phase
                cv2.putText(frame, f'Phase: {self.phase}', 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f'Stride Length: {stride_length:.2f}', 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f'Arm Angle: {arm_angle:.1f}', 
                          (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f'Wrist Velocity: {wrist_velocity:.2f}', 
                          (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display feedback
                for i, text in enumerate(feedback):
                    cv2.putText(frame, text, 
                              (10, 150 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Write the frame to output video
            out.write(frame)
            
            # Display the frame
            cv2.imshow('Pitch Phase Analysis', frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Print phase summary
        print("\nPitch Phase Summary:")
        for frame, phase in self.phases:
            print(f"Frame {frame}: {phase}")

if __name__ == "__main__":
    analyzer = PitchPhaseAnalyzer()
    video_path = "pitch1.mp4"  # Replace with your video path
    analyzer.process_video(video_path) 