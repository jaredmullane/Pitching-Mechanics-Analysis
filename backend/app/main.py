from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import mediapipe as mp
import numpy as np
import os
from typing import List, Dict
import tempfile
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MediaPipe Pose first
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

class PitchAnalyzer:
    def __init__(self):
        self.IDEAL_STRIDE_RATIO = 0.8
        self.IDEAL_ELBOW_ANGLE = 90
        self.IDEAL_WRIST_VELOCITY = 0.05
        self.IDEAL_HIP_ROTATION = 45
        self.IDEAL_KNEE_FLEXION = 90

    def calculate_angle(self, a, b, c):
        try:
            a = np.array([a.x, a.y])
            b = np.array([b.x, b.y])
            c = np.array([c.x, c.y])

            ba = a - b
            bc = c - b

            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            angle = np.arccos(cosine_angle)
            return np.degrees(angle)
        except Exception as e:
            logger.error(f"Error calculating angle: {str(e)}")
            return 0.0

    def calculate_distance(self, a, b):
        try:
            a = np.array([a.x, a.y])
            b = np.array([b.x, b.y])
            return np.linalg.norm(a - b)
        except Exception as e:
            logger.error(f"Error calculating distance: {str(e)}")
            return 0.0

    def analyze_frame(self, frame, pose_landmarks):
        if not pose_landmarks:
            logger.warning("No pose landmarks detected in frame")
            return None

        try:
            # Extract key joint positions
            left_hip = pose_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = pose_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            left_knee = pose_landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
            left_ankle = pose_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            right_ankle = pose_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            left_shoulder = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            left_elbow = pose_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            left_wrist = pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

            # Calculate metrics
            stride_length = self.calculate_distance(left_ankle, right_ankle)
            arm_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            hip_angle = self.calculate_angle(left_shoulder, left_hip, right_hip)
            knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)

            # Generate feedback
            feedback = []
            if stride_length < self.IDEAL_STRIDE_RATIO:
                feedback.append("Increase stride length for more power")
            if arm_angle < self.IDEAL_ELBOW_ANGLE:
                feedback.append("Improve external shoulder rotation")
            if hip_angle < self.IDEAL_HIP_ROTATION:
                feedback.append("Increase hip rotation for more power")
            if knee_angle < self.IDEAL_KNEE_FLEXION:
                feedback.append("Increase knee flexion for better drive")

            return {
                "stride_length": float(stride_length),
                "arm_angle": float(arm_angle),
                "hip_angle": float(hip_angle),
                "knee_angle": float(knee_angle),
                "feedback": feedback if feedback else ["Good mechanics!"]
            }
        except Exception as e:
            logger.error(f"Error analyzing frame: {str(e)}")
            return None

# Create FastAPI app
app = FastAPI(title="Pitching Mechanics Analysis API")

# Configure CORS with more specific settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002", 
                  "http://localhost:3003", "http://localhost:3004", "http://localhost:3005", 
                  "http://localhost:3006", "http://localhost:3007", "http://localhost:3008",
                  "http://localhost:3009", "http://localhost:3010"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600
)

# Create an instance of PitchAnalyzer
pitch_analyzer = PitchAnalyzer()

@app.post("/api/analyze-pitch")
async def analyze_pitch(file: UploadFile = File(...)):
    temp_path = None
    try:
        # Log file details
        logger.info(f"Received file: {file.filename}, content_type: {file.content_type}")
        
        # Validate file type
        if not file.content_type.startswith('video/'):
            logger.error(f"Invalid file type: {file.content_type}")
            raise HTTPException(
                status_code=422,
                detail=f"File must be a video. Received content type: {file.content_type}"
            )

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            content = await file.read()
            if not content:
                logger.error("Empty file received")
                raise HTTPException(
                    status_code=422,
                    detail="Empty file received. Please upload a valid video file."
                )
            temp_file.write(content)
            temp_path = temp_file.name
            logger.info(f"Saved video to temporary file: {temp_path}")

        # Process video frames
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            logger.error("Failed to open video file")
            raise HTTPException(
                status_code=422,
                detail="Failed to open video file. Please ensure the file is a valid video format."
            )

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video properties: {width}x{height} @ {fps}fps, {frame_count} frames")

        if frame_count == 0:
            logger.error("Video has no frames")
            raise HTTPException(
                status_code=422,
                detail="Video has no frames. Please upload a valid video file."
            )

        frame_count = 0
        phases = []
        results = []

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                logger.warning(f"Failed to read frame at count {frame_count}")
                break

            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_pose = pose.process(frame_rgb)
            
            if results_pose.pose_landmarks:
                # Extract key landmarks for phase detection
                landmarks = results_pose.pose_landmarks.landmark

                # Simple phase detection logic (can be enhanced)
                if frame_count == 0:
                    phases.append({"frame": frame_count, "phase": "Setup"})
                elif frame_count == 30:  # Example frame numbers
                    phases.append({"frame": frame_count, "phase": "Wind-up"})
                elif frame_count == 60:
                    phases.append({"frame": frame_count, "phase": "Stride"})
                elif frame_count == 90:
                    phases.append({"frame": frame_count, "phase": "Release"})

                analysis = pitch_analyzer.analyze_frame(frame, landmarks)
                if analysis:
                    results.append(analysis)
                    logger.info(f"Successfully analyzed frame {frame_count}")
                else:
                    logger.warning(f"Failed to analyze frame {frame_count}")

            frame_count += 1

        cap.release()

        if not results:
            logger.error("No pose landmarks detected in the video")
            raise HTTPException(
                status_code=422,
                detail="No pose landmarks detected in the video. Please ensure the video shows a clear view of the pitcher."
            )

        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

        return JSONResponse(content={
            "status": "success",
            "analysis": results,
            "summary": {
                "average_stride_length": float(np.mean([r["stride_length"] for r in results])),
                "average_arm_angle": float(np.mean([r["arm_angle"] for r in results])),
                "average_hip_angle": float(np.mean([r["hip_angle"] for r in results])),
                "average_knee_angle": float(np.mean([r["knee_angle"] for r in results])),
                "common_feedback": list(set([f for r in results for f in r["feedback"]]))
            },
            "phases": phases,
            "total_frames": frame_count
        })

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing video: {str(e)}"
        )

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"message": "Pitching Analysis API is running"} 